from typing import Literal

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn.utils.rnn import pad_sequence

from sp.aggregators import MetricAggregator
from sp.configs import Config
from sp.configs import Model
from sp.configs import ObjectDetectionEvaluatorConfig
from sp.configs import Split
from sp.data_types import ObjectDetectionBatch
from sp.data_types import ObjectDetectionPrediction
from sp.data_types import TokensBatch
from sp.evaluators import ObjectDetectionEvaluator
from sp.nn.gnn_detector import GNNDetector
from sp.nn.object_detector import ObjectDetector
from sp.nn.object_detector import ObjectDetectorOutputs
from sp.nn.pcn_detector import PCNDetector
from sp.nn.transformer_detector import TransformerDetector
from sp.timers import CudaTimer
from sp.yolox.losses import YOLOXLoss
from sp.yolox.utils import postprocess


class ObjectDetectionModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        evaluation: ObjectDetectionEvaluatorConfig,
        test_split: Split = Split.test,
    ):
        super().__init__()

        self.batch_size = config.batch_size
        self.batch_size_random = config.object_detection.batch_size_random
        self.config = config
        self.evaluation = evaluation
        self.num_classes = evaluation.num_classes
        self.optimizer = config.optimizer
        self.test_split = test_split

        if config.model == Model.transformer:
            hidden_size = config.transformer.hidden_size
            encoder = TransformerDetector(config)
            self.merge_inputs = self.merge_tokens
        elif config.model == Model.gnn:
            hidden_size = config.gnn.dim3
            encoder = GNNDetector(config)
            self.merge_inputs = self.merge_graphs
        elif config.model == Model.pcn:
            hidden_size = config.pcn.dim3
            encoder = PCNDetector(config)
            self.merge_inputs = self.merge_clouds
        else:
            raise ValueError(f"Unsupported model type: {config.model}")

        self.model = ObjectDetector(
            config=config,
            encoder=encoder,
            hidden_size=hidden_size,
            num_classes=self.num_classes,
        )

        self.loss = YOLOXLoss(num_classes=self.num_classes)
        self.aggregator = MetricAggregator()

        self.evaluator = ObjectDetectionEvaluator(evaluation)

    def configure_optimizers(self):
        if self.optimizer is None:
            raise ValueError("Missing an optimizer configuration.")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=self.optimizer.div_factor,
            final_div_factor=self.optimizer.final_div_factor,
            max_lr=self.optimizer.lr,
            optimizer=optimizer,
            pct_start=self.optimizer.pct_start,
            total_steps=self.optimizer.steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": True,
                "name": "learning_rate",
            },
        }

    def training_step(self, batch: dict[Literal["random", "streaming"], ObjectDetectionBatch], batch_idx: int):
        batch = self.merge_training_batch(batch["random"], batch["streaming"])

        assert batch.has_labels.any(), "Batch must contain at least one labelled sample."

        with CudaTimer("train.forward"):
            outputs: ObjectDetectorOutputs = self.model(batch)

        yolox_labels = [batch.yolox_labels[i] for i in outputs.labelled_indices]
        yolox_labels = pad_sequence(yolox_labels, batch_first=True, padding_value=0)
        losses = self.loss(outputs.yolox_outputs_labelled, yolox_labels)

        batch_size = len(batch.sequence_ids)
        self.log(*self.aggregator("train/loss", losses["loss"]), prog_bar=True)
        self.log(*self.aggregator("train/loss_iou", losses["iou"]), batch_size=batch_size)
        self.log(*self.aggregator("train/loss_obj", losses["obj"]), batch_size=batch_size)
        self.log(*self.aggregator("train/loss_cls", losses["cls"]), batch_size=batch_size)
        self.log(*self.aggregator("train/num_fg", losses["num_fg"]), batch_size=batch_size)

        return losses["loss"]

    def on_validation_epoch_start(self) -> None:
        self.on_evalution_epoch_begin(Split.val)

    def validation_step(self, batch: ObjectDetectionBatch, batch_idx: int):
        self.evaluation_step(batch)

    def on_validation_epoch_end(self) -> None:
        self.evaluation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.on_evalution_epoch_begin(self.test_split)

    def test_step(self, batch: ObjectDetectionBatch, batch_idx: int):
        self.evaluation_step(batch)

    def on_test_epoch_end(self) -> None:
        self.evaluation_epoch_end()

    def on_evalution_epoch_begin(self, split: Split):
        self.evaluation_split = split

    def evaluation_step(self, batch: ObjectDetectionBatch):
        split = self.evaluation_split.value
        with CudaTimer(f"{split}.forward"):
            outputs: ObjectDetectorOutputs = self.model(batch)

        if outputs.yolox_outputs_labelled is not None:
            yolox_labels = [batch.yolox_labels[i] for i in outputs.labelled_indices]
            yolox_labels = pad_sequence(yolox_labels, batch_first=True, padding_value=0)
            losses = self.loss(outputs.yolox_outputs_labelled, yolox_labels)

            batch_size = len(batch.sequence_ids)
            self.log(f"{split}/loss", losses["loss"], prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log(f"{split}/loss_iou", losses["iou"], batch_size=batch_size, sync_dist=True)
            self.log(f"{split}/loss_obj", losses["obj"], batch_size=batch_size, sync_dist=True)
            self.log(f"{split}/loss_cls", losses["cls"], batch_size=batch_size, sync_dist=True)
            self.log(f"{split}/num_fg", losses["num_fg"], batch_size=batch_size, sync_dist=True)

        # list of tensor, shape [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
        predictions = postprocess(
            outputs.yolox_outputs_all["predictions"],
            self.num_classes,
            conf_thre=self.config.nms.conf,
            nms_thre=self.config.nms.iou,
        )

        predictions = self.batch_format_predictions(predictions, batch.prediction_times)
        iterator = zip(batch.sequence_ids, predictions, batch.prophesee_labels, strict=True)
        for sequence_id, prediction, labels in iterator:
            if prediction.length != 0:
                self.evaluator.add_predictions(sequence_id, prediction)
            if labels is not None:
                self.evaluator.add_labels(sequence_id, labels)

    def evaluation_epoch_end(self) -> None:
        split = self.evaluation_split.value
        metrics = self.evaluator.evaluate()
        self.log_dict(
            {
                f"{split}/mAP@COCO": metrics["AP"],
                f"{split}/mAP@.50": metrics["AP_50"],
                f"{split}/mAP@.75": metrics["AP_75"],
                f"{split}/mAP@S": metrics["AP_S"],
                f"{split}/mAP@M": metrics["AP_M"],
                f"{split}/mAP@L": metrics["AP_L"],
            },
            sync_dist=True,
        )

    def batch_format_predictions(
        self, predictions: list[torch.Tensor], timestamps: torch.Tensor
    ) -> list[ObjectDetectionPrediction]:
        return [
            self.format_predictions(prediction, timestamp)
            for prediction, timestamp in zip(predictions, timestamps, strict=True)
        ]

    def format_predictions(
        self, predictions: torch.Tensor | None, timestamp: torch.Tensor
    ) -> ObjectDetectionPrediction:
        if predictions is None or predictions.shape[0] == 0:
            return ObjectDetectionPrediction(
                t=torch.tensor([], dtype=torch.int64),
                x=torch.tensor([], dtype=torch.float32),
                y=torch.tensor([], dtype=torch.float32),
                w=torch.tensor([], dtype=torch.float32),
                h=torch.tensor([], dtype=torch.float32),
                class_id=torch.tensor([], dtype=torch.long),
                class_confidence=torch.tensor([], dtype=torch.float32),
            )

        boxes = predictions[:, :4]
        obj_confidence = predictions[:, 4]
        class_confidence = predictions[:, 5]
        class_id = predictions[:, 6].to(torch.long)

        scores = obj_confidence * class_confidence
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = boxes.unbind(dim=-1)
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        boxes = torch.stack((top_left_x, top_left_y, width, height), dim=-1)
        boxes, crop_mask = self.crop_boxes_outside_of_image(boxes)

        return ObjectDetectionPrediction(
            t=timestamp.expand(boxes.shape[0]),
            x=boxes[:, 0],
            y=boxes[:, 1],
            w=boxes[:, 2],
            h=boxes[:, 3],
            class_id=class_id[crop_mask],
            class_confidence=scores[crop_mask],
        )

    def crop_boxes_outside_of_image(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = boxes[:, 0]
        right = boxes[:, 0] + boxes[:, 2]
        top = boxes[:, 1]
        bottom = boxes[:, 1] + boxes[:, 3]

        left = torch.clamp(left, 0, self.evaluation.width - 1)
        right = torch.clamp(right, 0, self.evaluation.width - 1)
        top = torch.clamp(top, 0, self.evaluation.height - 1)
        bottom = torch.clamp(bottom, 0, self.evaluation.height - 1)

        width = right - left
        height = bottom - top

        assert torch.all(width >= 0)
        assert torch.all(height >= 0)

        cropped_boxes = boxes.clone()
        cropped_boxes[:, 0] = left
        cropped_boxes[:, 1] = top
        cropped_boxes[:, 2] = width
        cropped_boxes[:, 3] = height

        # remove boxes with zero area
        # this may happen if the box is completely outside of the image
        mask = (width > 0) & (height > 0)

        return cropped_boxes, mask

    def merge_training_batch(
        self, random_batch: ObjectDetectionBatch, streaming_batch: ObjectDetectionBatch
    ) -> ObjectDetectionBatch:
        streaming_batch_indices = streaming_batch.batch_indices + self.batch_size_random
        batch_indices = torch.cat([random_batch.batch_indices, streaming_batch_indices], dim=0)
        inputs = self.merge_inputs(random_batch.inputs, streaming_batch.inputs)
        has_labels = torch.cat([random_batch.has_labels, streaming_batch.has_labels], dim=0)
        prediction_times = torch.cat([random_batch.prediction_times, streaming_batch.prediction_times], dim=0)
        prophesee_labels = random_batch.prophesee_labels + streaming_batch.prophesee_labels
        reset = torch.cat([random_batch.reset, streaming_batch.reset], dim=0)
        sequence_ids = random_batch.sequence_ids + streaming_batch.sequence_ids
        sequence_lengths = torch.cat([random_batch.sequence_lengths, streaming_batch.sequence_lengths], dim=0)
        yolox_labels = random_batch.yolox_labels + streaming_batch.yolox_labels

        return ObjectDetectionBatch(
            batch_indices=batch_indices,
            inputs=inputs,
            has_labels=has_labels,
            prediction_times=prediction_times,
            prophesee_labels=prophesee_labels,
            reset=reset,
            sequence_ids=sequence_ids,
            sequence_lengths=sequence_lengths,
            yolox_labels=yolox_labels,
            worker_id=streaming_batch.worker_id,  # worker_id is only used in streaming mode
        )

    def merge_clouds(
        self, random_batch: torch_geometric.data.Batch, streaming_batch: torch_geometric.data.Batch
    ) -> torch_geometric.data.Batch:
        streaming_batch_ids = streaming_batch.batch + random_batch.batch_size
        batch = torch.cat([random_batch.batch, streaming_batch_ids], dim=0)
        pos = torch.cat([random_batch.pos, streaming_batch.pos], dim=0)
        x = torch.cat([random_batch.x, streaming_batch.x], dim=0)
        return torch_geometric.data.Batch(
            batch=batch,
            pos=pos,
            x=x,
        )

    def merge_graphs(
        self, random_batch: torch_geometric.data.Batch, streaming_batch: torch_geometric.data.Batch
    ) -> torch_geometric.data.Batch:
        streaming_batch_ids = streaming_batch.batch + random_batch.batch_size
        batch = torch.cat([random_batch.batch, streaming_batch_ids], dim=0)
        streaming_edge_index = streaming_batch.edge_index + random_batch.x.size(0)
        edge_index = torch.cat([random_batch.edge_index, streaming_edge_index], dim=1)
        pos = torch.cat([random_batch.pos, streaming_batch.pos], dim=0)
        x = torch.cat([random_batch.x, streaming_batch.x], dim=0)
        return torch_geometric.data.Batch(
            batch=batch,
            edge_index=edge_index,
            pos=pos,
            x=x,
        )

    def merge_tokens(self, random_batch: TokensBatch, streaming_batch: TokensBatch) -> TokensBatch:
        max_random_tokens = random_batch.tokens.size(1)
        max_streaming_tokens = streaming_batch.tokens.size(1)
        max_tokens = max(max_random_tokens, max_streaming_tokens)
        random_padded = self.pad_tokens(random_batch, max_tokens)
        streaming_padded = self.pad_tokens(streaming_batch, max_tokens)
        return TokensBatch(
            batch_size=random_batch.batch_size + streaming_batch.batch_size,
            prediction_time=torch.cat([random_padded.prediction_time, streaming_padded.prediction_time], dim=0),
            pos_x=torch.cat([random_padded.pos_x, streaming_padded.pos_x], dim=0),
            pos_y=torch.cat([random_padded.pos_y, streaming_padded.pos_y], dim=0),
            pos_t=torch.cat([random_padded.pos_t, streaming_padded.pos_t], dim=0),
            tokens=torch.cat([random_padded.tokens, streaming_padded.tokens], dim=0),
            padding_mask=torch.cat([random_padded.padding_mask, streaming_padded.padding_mask], dim=0),
        )

    def pad_tokens(self, batch: TokensBatch, padding_length: int) -> TokensBatch:
        num_tokens = batch.tokens.size(1)
        if num_tokens == padding_length:
            return batch

        num_dimensions = len(batch.tokens.shape)
        num_pad = padding_length - num_tokens
        tokens_pad = [0, 0] * num_dimensions
        tokens_pad[-3] = num_pad
        tokens_pad = tuple(tokens_pad)
        tokens = F.pad(batch.tokens, tokens_pad)
        pad = (0, num_pad)
        return TokensBatch(
            batch_size=batch.batch_size,
            prediction_time=batch.prediction_time,
            pos_x=F.pad(batch.pos_x, pad),
            pos_y=F.pad(batch.pos_y, pad),
            pos_t=F.pad(batch.pos_t, pad),
            tokens=tokens,
            padding_mask=F.pad(batch.padding_mask, pad, value=True),
        )
