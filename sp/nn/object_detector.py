import dataclasses
import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sp.configs import Config
from sp.configs import Model
from sp.configs import TokenizerType
from sp.data_types import ObjectDetectionBatch
from sp.loaders import load_dimensions
from sp.yolox.head import YOLOXHead
from sp.yolox.neck import YOLOXNeck

States = tuple[torch.Tensor, torch.Tensor]  # (hidden_state, cell_state)


@dataclasses.dataclass
class ObjectDetectorOutputs:
    labelled_indices: torch.Tensor
    yolox_outputs_all: dict[str, torch.Tensor] | None
    yolox_outputs_labelled: dict[str, torch.Tensor] | None


class ObjectDetector(pl.LightningModule):
    def __init__(self, config: Config, encoder: nn.Module, hidden_size: int, num_classes: int):
        super().__init__()
        self.batch_size = config.batch_size

        height, width = load_dimensions(config.dataset)
        grid_size = self.get_grid_size(config)
        patch_size = config.patch_size
        if config.tokenizer == TokenizerType.none:
            self.num_rows = math.ceil(height / grid_size)
            self.num_cols = math.ceil(width / grid_size)
            fpn2_stride = grid_size
        else:
            self.num_rows = math.ceil(math.ceil(height / patch_size) / grid_size)
            self.num_cols = math.ceil(math.ceil(width / patch_size) / grid_size)
            fpn2_stride = patch_size * grid_size
            strides = [patch_size * grid_size]

        fpn1_stride = fpn2_stride // 2
        fpn3_stride = 2 * fpn2_stride
        strides = [fpn1_stride, fpn2_stride, fpn3_stride]

        self.hidden_size = hidden_size
        self.encoder = encoder
        self.rnn = RNN(hidden_size=hidden_size, num_rows=self.num_rows, num_cols=self.num_cols)

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Identity()

        self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.neck = YOLOXNeck(
            in_features=("fpn1", "fpn2", "fpn3"),
            in_channels=[hidden_size, hidden_size, hidden_size],
        )

        self.head = YOLOXHead(
            num_classes=num_classes,
            in_channels=[hidden_size, hidden_size, hidden_size],
            strides=strides,
        )

        self.rnn_states = {}

        self._float_dtype = None

    def forward(self, batch: ObjectDetectionBatch) -> ObjectDetectorOutputs:
        total_steps = batch.sequence_lengths.sum().item()
        encoder_outputs = self.encoder(batch.inputs, total_steps)
        assert encoder_outputs.shape[0] == total_steps, (
            f"Expected {total_steps} steps, but got {encoder_outputs.shape[0]} steps from the encoder."
        )

        sequence_lengths = batch.sequence_lengths
        zero = torch.zeros((1,), dtype=sequence_lengths.dtype, device=sequence_lengths.device)
        cumulative_lengths = sequence_lengths.cumsum(dim=0)[:-1]
        batch_offsets = torch.cat([zero, cumulative_lengths])
        max_length = sequence_lengths.max().item()

        batch_indices = batch.batch_indices[batch_offsets]

        if self.has_state(batch.worker_id):
            state = self.get_state(batch.worker_id)
        else:
            state = self.init_state(batch.worker_id)

        fpn_inputs = torch.empty(
            total_steps,
            self.num_rows,
            self.num_cols,
            self.hidden_size,
            device=self.device,
            dtype=self.float_dtype,
        )

        for length in range(1, max_length + 1):
            batch_mask = length <= sequence_lengths
            sequence_lengths = sequence_lengths[batch_mask]
            batch_offsets = batch_offsets[batch_mask]
            step_indices = batch_offsets + (length - 1)
            batch_indices = batch.batch_indices[step_indices]

            reset = batch.reset[step_indices]
            if reset.any():
                batch_reset_indices = batch_indices[reset]
                reset_state = self.reset_state(batch.worker_id, batch_reset_indices)
                state[0][batch_reset_indices] = reset_state[0]
                state[1][batch_reset_indices] = reset_state[1]

            rnn_inputs = encoder_outputs[step_indices]
            rnn_state = (state[0][batch_indices], state[1][batch_indices])
            rnn_outputs, rnn_state = self.rnn(rnn_inputs, rnn_state)
            fpn_inputs[step_indices] = rnn_outputs
            state[0][batch_indices] = rnn_state[0]
            state[1][batch_indices] = rnn_state[1]

        if batch.worker_id != -1:  # No need to save state in random-access mode
            self.save_state(batch.worker_id, state)

        labelled_indices = torch.arange(total_steps, device=self.device)[batch.has_labels]

        fpn_inputs = fpn_inputs.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)

        if self.training:
            yolox_outputs_all = None
        else:
            fpn_features = self.pad_fpn_features(
                {
                    "fpn1": self.fpn1(fpn_inputs),
                    "fpn2": self.fpn2(fpn_inputs),
                    "fpn3": self.fpn3(fpn_inputs),
                }
            )
            neck_outputs = self.neck(fpn_features)
            yolox_outputs_all = self.head(neck_outputs)

        if labelled_indices.size(0) == 0:
            yolox_outputs_labelled = None
        else:
            fpn_inputs_labelled = fpn_inputs[labelled_indices]
            fpn_features_labelled = self.pad_fpn_features(
                {
                    "fpn1": self.fpn1(fpn_inputs_labelled),
                    "fpn2": self.fpn2(fpn_inputs_labelled),
                    "fpn3": self.fpn3(fpn_inputs_labelled),
                }
            )
            neck_outputs_labelled = self.neck(fpn_features_labelled)
            yolox_outputs_labelled = self.head(neck_outputs_labelled)

        return ObjectDetectorOutputs(
            labelled_indices=labelled_indices,
            yolox_outputs_all=yolox_outputs_all,
            yolox_outputs_labelled=yolox_outputs_labelled,
        )

    @property
    def float_dtype(self) -> torch.dtype:
        if self._float_dtype is None:
            match self.trainer.precision:
                case "16-mixed":
                    self._float_dtype = torch.float16
                case "32-true":
                    self._float_dtype = torch.float32
                case _:
                    raise ValueError(f"Unsupported precision: {self.trainer.precision}")
        return self._float_dtype

    def get_grid_size(self, config: Config) -> int:
        match config.model:
            case Model.gnn:
                return config.gnn.grid_size
            case Model.pcn:
                return config.pcn.grid_size
            case Model.transformer:
                return config.transformer.grid_size
            case _:
                raise ValueError(f"Unsupported model type: {config.model}")

    def pad_fpn_features(self, fpn_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Ensure that the YOLOXNeck can upsample low-resolution feature maps
        to the same size as the high-resolution feature maps."""
        desired_height = 2 * fpn_features["fpn3"].size(2)
        desired_width = 2 * fpn_features["fpn3"].size(3)
        outputs = {"fpn3": fpn_features["fpn3"]}
        for key in ("fpn2", "fpn1"):
            value = fpn_features[key]
            actual_height = value.size(2)
            actual_width = value.size(3)
            if actual_height != desired_height or actual_width != desired_width:
                pad_height = desired_height - actual_height
                pad_width = desired_width - actual_width
                value = F.pad(value, (0, pad_width, 0, pad_height))
            desired_height = 2 * value.size(2)
            desired_width = 2 * value.size(3)
            outputs[key] = value
        return outputs

    def get_state(self, worker_id: int) -> States:
        return self.rnn_states[worker_id]

    def has_state(self, worker_id: int) -> bool:
        return worker_id in self.rnn_states

    def init_state(self, worker_id: int) -> States:
        state_shape = (self.batch_size, self.num_rows, self.num_cols, self.hidden_size)
        hidden_state = torch.zeros(*state_shape, device=self.device, dtype=self.float_dtype)
        cell_state = torch.zeros(*state_shape, device=self.device, dtype=self.float_dtype)
        state = (hidden_state, cell_state)
        self.rnn_states[worker_id] = state
        return state

    def reset_state(self, worker_id: int, batch_indices: torch.Tensor) -> States:
        batch_size = batch_indices.size(0)
        state_shape = (batch_size, self.num_rows, self.num_cols, self.hidden_size)
        hidden_state = torch.zeros(*state_shape, device=self.device, dtype=self.float_dtype)
        cell_state = torch.zeros(*state_shape, device=self.device, dtype=self.float_dtype)
        self.rnn_states[worker_id][0][batch_indices] = hidden_state
        self.rnn_states[worker_id][1][batch_indices] = cell_state
        return hidden_state, cell_state

    def save_state(self, worker_id: int, state: States):
        hidden_state = state[0].detach()
        cell_state = state[1].detach()
        self.rnn_states[worker_id] = hidden_state, cell_state


class RNN(nn.Module):
    def __init__(self, hidden_size: int, num_rows: int, num_cols: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=hidden_size,
            hidden_size=hidden_size,
        )

    def forward(self, encoder_outputs: torch.Tensor, prev_states: States) -> tuple[torch.Tensor, States]:
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, height, width, hidden_size)
            prev_states: Tuple of (hidden_state, cell_state) each of shape (batch_size, height, width, hidden_size)
        """
        batch_size = encoder_outputs.size(0)
        num_cells = batch_size * self.num_rows * self.num_cols
        lstm_inputs = encoder_outputs.reshape(num_cells, 1, self.hidden_size)
        prev_hidden_state, prev_cell_state = prev_states
        prev_hidden_state = prev_hidden_state.view(1, num_cells, self.hidden_size)
        prev_cell_state = prev_cell_state.view(1, num_cells, self.hidden_size)
        prev_states = (prev_hidden_state, prev_cell_state)

        outputs, (hidden_state, cell_state) = self.lstm(lstm_inputs, prev_states)

        output_shape = (batch_size, self.num_rows, self.num_cols, self.hidden_size)
        hidden_state = hidden_state.view(*output_shape)
        cell_state = cell_state.view(*output_shape)
        states = (hidden_state, cell_state)
        outputs = outputs.view(*output_shape)

        return outputs, states
