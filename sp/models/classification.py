from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from torch.nn import functional as F

from sp.aggregators import MeanVoter
from sp.aggregators import MetricAggregator
from sp.configs import Config
from sp.configs import Model
from sp.configs import Split
from sp.data_types import ClassificationBatch
from sp.nn import GNNClassifier
from sp.nn import PCNClassifier
from sp.nn import TransformerClassifier
from sp.timers import CudaTimer


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        num_classes: int,
        test_split: Split = Split.test,
    ):
        super().__init__()

        self.optimizer = config.optimizer
        self.test_split = test_split

        self.aggregator = MetricAggregator()
        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy("multiclass", num_classes=num_classes, average="micro"),
                "precision": torchmetrics.Precision("multiclass", num_classes=num_classes, average="micro"),
                "recall": torchmetrics.Recall("multiclass", num_classes=num_classes, average="micro"),
                "f1": torchmetrics.F1Score("multiclass", num_classes=num_classes, average="micro"),
            },
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix=f"{test_split.value}/")

        self.model = self.make_network(config, num_classes)

        self.val_votes = MeanVoter()
        self.test_votes = MeanVoter()

    def training_step(self, batch: ClassificationBatch | None, batch_idx: int) -> Any:
        if batch is None:
            return None

        with CudaTimer("train.forward"):
            logits = self.model(batch.inputs)
        loss = self.loss(logits, batch.labels)

        self.log(*self.aggregator("train/loss", loss), prog_bar=True, batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch: ClassificationBatch | None, batch_idx: int) -> Any:
        if batch is None:
            return None

        with CudaTimer("val.forward"):
            logits = self.model(batch.inputs)
        loss = self.loss(logits, batch.labels)

        self.log("val/loss", loss, batch_size=batch.batch_size)

        preds = logits.softmax(dim=-1)
        sparse_labels = torch.argmax(batch.labels, dim=-1)
        self.val_votes.update(batch.ids, preds, sparse_labels)

        return loss

    def on_validation_epoch_end(self):
        preds, labels = self.val_votes.compute()
        metrics = self.val_metrics(preds, labels)
        self.log_dict(metrics)
        self.val_metrics.reset()
        self.val_votes.reset()

    def test_step(self, batch: ClassificationBatch | None, batch_idx: int) -> Any:
        if batch is None:
            return None

        prefix = self.test_split.value
        with CudaTimer(f"{prefix}.forward"):
            logits = self.model(batch.inputs)
        loss = self.loss(logits, batch.labels)

        self.log(f"{prefix}/loss", loss, batch_size=batch.batch_size)

        preds = logits.softmax(dim=-1)
        sparse_labels = torch.argmax(batch.labels, dim=-1)
        self.test_votes.update(batch.ids, preds, sparse_labels)

        return loss

    def on_test_epoch_end(self):
        preds, labels = self.test_votes.compute()
        metrics = self.test_metrics(preds, labels)
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_votes.reset()

    def configure_optimizers(self) -> Any:
        if self.optimizer is None:
            raise ValueError("Missing an optimizer configuration.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer.lr)

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

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def make_network(self, config: Config, num_classes: int) -> GNNClassifier | TransformerClassifier:
        match config.model:
            case Model.gnn:
                return GNNClassifier(config=config, num_classes=num_classes)
            case Model.pcn:
                return PCNClassifier(config=config, num_classes=num_classes)
            case Model.transformer:
                return TransformerClassifier(config=config, num_classes=num_classes)
            case _:
                raise NotImplementedError(f"Model {config.model} is not implemented for classification.")
