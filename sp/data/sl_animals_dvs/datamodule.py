import json

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
from sp.data.sl_animals_dvs.dataset import Dataset
from sp.data.sl_animals_dvs.dataset import Sample
from sp.paths import get_dataset_dir
from sp.sl_animals_dvs import DATASET_NAME
from sp.sl_animals_dvs import TEST_USERS
from sp.sl_animals_dvs import VAL_USERS


class SLAnimalsDVSDataModule(pl.LightningDataModule):
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
        self.model = config.model
        self.num_workers = config.num_workers
        self.pin_memory = pin_memory
        self.test_split = test_split
        self.tokenizer = config.tokenizer
        self.validate = config.validate
        self.dataset_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed")

    def setup(self, stage: str):
        with open(self.dataset_dir / "index.json", "r") as file:
            samples = json.load(file)

        samples = [
            Sample(
                id=sample["gesture_id"],
                label=sample["label"],
                path=self.dataset_dir / sample["path"],
                user=sample["user"],
            )
            for sample in samples
        ]

        if stage == "fit":
            if self.validate:
                train = [sample for sample in samples if sample.user not in VAL_USERS and sample.user not in TEST_USERS]
                val = [sample for sample in samples if sample.user in VAL_USERS]
                self.train_dataset = Dataset(self.config, train, augment=self.augment)
                self.val_dataset = Dataset(self.config, val)
            else:
                train = [sample for sample in samples if sample.user not in TEST_USERS]
                self.train_dataset = Dataset(self.config, train, augment=self.augment)
                self.val_dataset = None
        elif stage == "test":
            match self.test_split:
                case Split.train:
                    test = [
                        sample for sample in samples if sample.user not in VAL_USERS and sample.user not in TEST_USERS
                    ]
                case Split.val:
                    test = [sample for sample in samples if sample.user in VAL_USERS]
                case Split.test:
                    test = [sample for sample in samples if sample.user in TEST_USERS]
            self.test_dataset = Dataset(self.config, test)
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
