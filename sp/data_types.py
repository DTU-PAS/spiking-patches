import dataclasses

import numpy as np
import torch
import torch_geometric

from sp.events import Events


@dataclasses.dataclass
class Tokens:
    prediction_time: int
    pos_x: np.ndarray
    pos_y: np.ndarray
    pos_t: np.ndarray
    events_x: list[np.ndarray]
    events_y: list[np.ndarray]
    events_t: list[np.ndarray]
    events_p: list[np.ndarray]


@dataclasses.dataclass
class TokensBatch:
    batch_size: int
    prediction_time: torch.Tensor
    pos_x: torch.Tensor
    pos_y: torch.Tensor
    pos_t: torch.Tensor
    tokens: torch.Tensor  # Shape: (batch_size, seq_length, 2, B, P, P)
    padding_mask: torch.Tensor

    def to(self, device: torch.device):
        self.prediction_time = self.prediction_time.to(device)
        self.pos_x = self.pos_x.to(device)
        self.pos_y = self.pos_y.to(device)
        self.pos_t = self.pos_t.to(device)
        self.tokens = self.tokens.to(device)
        self.padding_mask = self.padding_mask.to(device)
        return self


@dataclasses.dataclass
class ClassificationEventsData:
    events: Events
    label: torch.Tensor
    id: str


@dataclasses.dataclass
class ClassificationBatch:
    inputs: TokensBatch | torch_geometric.data.Batch
    batch_size: int
    labels: torch.Tensor
    ids: list[str]

    def to(self, device: torch.device):
        self.inputs.to(device)
        self.labels = self.labels.to(device)
        return self


@dataclasses.dataclass
class ObjectDetectionTokensData:
    batch_index: list[int] | None
    inputs: list[Tokens] | list[torch_geometric.data.Data]
    prediction_time: list[int]
    prophesee_labels: list[np.ndarray | None]
    reset: list[bool]
    sequence_id: list[int]
    yolox_labels: list[torch.Tensor | None]
    worker_id: int


@dataclasses.dataclass
class ObjectDetectionBatch:
    batch_indices: torch.Tensor
    inputs: TokensBatch | torch_geometric.data.Batch
    has_labels: torch.Tensor
    prediction_times: torch.Tensor
    prophesee_labels: list[np.ndarray | None]
    reset: torch.Tensor
    sequence_ids: list[int]
    sequence_lengths: torch.Tensor
    yolox_labels: list[torch.Tensor | None]
    worker_id: int

    def to(self, device: torch.device):
        self.batch_indices = self.batch_indices.to(device)
        self.inputs = self.inputs.to(device)
        self.has_labels = self.has_labels.to(device)
        self.prediction_times = self.prediction_times.to(device)
        self.reset = self.reset.to(device)
        self.sequence_lengths = self.sequence_lengths.to(device)
        self.yolox_labels = [label.to(device) if label is not None else None for label in self.yolox_labels]
        return self


PROPHESEE_BBOX_DTYPE = np.dtype(
    {
        "names": ["t", "x", "y", "w", "h", "class_id", "class_confidence"],
        "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<f4"],
    }
)


class ObjectDetectionPrediction:
    """Represents the predictions at a specific time step."""

    def __init__(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        class_id: torch.Tensor,
        class_confidence: torch.Tensor,
    ):
        self.length = t.numel()
        self.t = t.cpu().numpy()
        self.x = x.cpu().numpy()
        self.y = y.cpu().numpy()
        self.w = w.cpu().numpy()
        self.h = h.cpu().numpy()
        self.class_id = class_id.cpu().numpy()
        self.class_confidence = class_confidence.cpu().numpy()

    def numpy(self):
        output = list(zip(self.t, self.x, self.y, self.w, self.h, self.class_id, self.class_confidence, strict=True))
        output = np.array(output, dtype=PROPHESEE_BBOX_DTYPE)
        return output
