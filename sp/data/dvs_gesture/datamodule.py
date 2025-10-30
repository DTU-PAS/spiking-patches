import json
from pathlib import Path

import lightning.pytorch as pl
import torch

from sp.collators import ClassificationEventGraphCollator
from sp.collators import ClassificationEventPointCollator
from sp.collators import ClassificationTokenCollator
from sp.collators import ClassificationTokenGraphCollator
from sp.collators import ClassificationTokenPointCollator
from sp.configs import Config
from sp.configs import Model
from sp.configs import Split
from sp.configs import TokenizerType
from sp.data.dvs_gesture.dataset import Dataset
from sp.data.dvs_gesture.dataset import Sample
from sp.dvs_gesture import DATASET_NAME
from sp.dvs_gesture import VALIDATION_USERS
from sp.paths import get_dataset_dir


class DVSGestureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Config,
        augment: bool = True,
        pin_memory: bool = False,
        test_split: Split = Split.test,
    ):
        super().__init__()

        self.augment = augment
        self.batch_size = config.batch_size
        self.config = config
        self.num_workers = config.num_workers
        self.pin_memory = pin_memory
        self.test_split = test_split
        self.validate = config.validate

        self.model = config.model
        self.tokenizer = config.tokenizer

        self.dataset_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed")

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            samples = self.load_samples(self.dataset_dir / "train.json")
            train, val = self.train_val_split(samples)
            self.train_dataset = Dataset(self.config, train, augment=self.augment)
            self.val_dataset = None if val is None else Dataset(self.config, val)
        elif stage == "test":
            if self.test_split == Split.train:
                samples = self.load_samples(self.dataset_dir / "train.json")
                samples, _ = self.train_val_split(samples)
            elif self.test_split == Split.val:
                samples = self.load_samples(self.dataset_dir / "train.json")
                _, samples = self.train_val_split(samples)
            elif self.test_split == Split.test:
                samples = self.load_samples(self.dataset_dir / "test.json")
            self.test_dataset = Dataset(self.config, samples)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.get_collator(),
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.get_collator(),
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.get_collator(),
            pin_memory=self.pin_memory,
        )

    def load_samples(self, path: Path) -> list[Sample]:
        json_samples = json.loads(path.read_text())
        samples = []
        for json_sample in json_samples:
            sample = Sample(
                label=json_sample["label"],
                id=json_sample["action_id"],
                path=self.dataset_dir / json_sample["filename"],
                user=json_sample["user"],
            )
            samples.append(sample)
        return samples

    def train_val_split(self, samples: list[Sample]):
        if not self.validate:
            return samples, None

        train = [sample for sample in samples if sample.user not in VALIDATION_USERS]
        val = [sample for sample in samples if sample.user in VALIDATION_USERS]
        return train, val

    def get_collator(self):
        if self.model == Model.gnn:
            if self.tokenizer == TokenizerType.none:
                return ClassificationEventGraphCollator(self.config)
            else:
                return ClassificationTokenGraphCollator(self.config)
        elif self.model == Model.pcn:
            if self.tokenizer == TokenizerType.none:
                return ClassificationEventPointCollator(self.config)
            else:
                return ClassificationTokenPointCollator(self.config)
        elif self.model == Model.transformer:
            return ClassificationTokenCollator(self.config)
        else:
            raise NotImplementedError
