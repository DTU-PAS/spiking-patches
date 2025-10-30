import json
import math

import lightning.pytorch as pl
import numpy as np
import torch

from sp.collators import ObjectDetectionCollator
from sp.configs import Config
from sp.configs import Split
from sp.data.object_detection.dataset_random import DatasetRandom
from sp.data.object_detection.dataset_random import Sample
from sp.data.object_detection.dataset_streaming import DatasetStreaming
from sp.io import Sequence
from sp.io import load_sequences
from sp.paths import get_dataset_dir


class ObjectDetectionDataModule(pl.LightningDataModule):
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
        self.batch_size_random = config.object_detection.batch_size_random
        self.batch_size_streaming = config.object_detection.batch_size_streaming
        self.config = config
        self.num_workers = config.num_workers
        self.num_workers_random = config.object_detection.num_workers_random
        self.num_workers_streaming = config.object_detection.num_workers_streaming
        self.pin_memory = pin_memory
        self.predict_every_us = config.predict_every_ms * 1000
        self.prepare_data_per_node = True
        self.preprocessed_dir = get_dataset_dir(f"{config.dataset.value}-preprocessed")
        self.test_split = test_split

        preprocess_config_path = self.preprocessed_dir / "config.json"
        if not preprocess_config_path.exists():
            raise FileNotFoundError(f"Preprocessing config not found at '{preprocess_config_path}'")

        preprocess_config = json.loads(preprocess_config_path.read_text())
        if preprocess_config["chunk_duration_ms"] != config.predict_every_ms:
            raise ValueError(
                f"Chunk duration in preprocessed config ({preprocess_config['chunk_duration_ms']} ms) "
                f"does not match --predict-every-ms ({config.predict_every_ms} ms)"
            )

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.train_random_dataset = self.load_random_dataset(Split.train, augment=self.augment)
                self.train_streaming_dataset = self.load_streaming_dataset(
                    Split.train, augment=self.augment, batch_size=self.batch_size_streaming, training=True
                )
                self.val_dataset = self.load_streaming_dataset(Split.val, batch_size=self.batch_size)
            case "validate":
                self.val_dataset = self.load_streaming_dataset(Split.val, batch_size=self.batch_size)
            case "test":
                self.test_dataset = self.load_streaming_dataset(self.test_split, batch_size=self.batch_size)
            case _:
                raise ValueError(f"Unsupported stage: {stage}")

    def load_random_dataset(self, split: Split, augment: bool = False):
        sequences = load_sequences(self.preprocessed_dir / split.value)
        chunk_index_to_labels = {
            sequence_index: self.get_chunk_index_to_labels(split, sequence)
            for sequence_index, sequence in enumerate(sequences)
        }
        samples = self.load_random_access_samples(sequences, chunk_index_to_labels)
        return DatasetRandom(
            augment=augment,
            config=self.config,
            chunk_index_to_labels=chunk_index_to_labels,
            samples=samples,
            sequences=sequences,
        )

    def load_random_access_samples(
        self, sequences: list[Sequence], chunk_index_to_labels: dict[int, dict[int, np.ndarray]]
    ) -> list[Sample]:
        samples = []
        for sequence_index in range(len(sequences)):
            sequence_chunk_index_to_labels = chunk_index_to_labels[sequence_index]
            for chunk_index in sorted(sequence_chunk_index_to_labels.keys()):
                samples.append(Sample(sequence=sequence_index, chunk=chunk_index))
        return samples

    def load_streaming_dataset(self, split: Split, batch_size: int, augment: bool = False, training: bool = False):
        sequences = load_sequences(self.preprocessed_dir / split.value)
        chunk_index_to_labels = {
            sequence_index: self.get_chunk_index_to_labels(split, sequence)
            for sequence_index, sequence in enumerate(sequences)
        }
        return DatasetStreaming(
            augment=augment,
            batch_size=batch_size,
            config=self.config,
            chunk_index_to_labels=chunk_index_to_labels,
            sequences=sequences,
            training=training,
        )

    def get_chunk_index_to_labels(self, split: Split, sequence: Sequence) -> dict[int, np.ndarray]:
        """Create a mapping from chunk index to labels for the given sequence."""
        path = self.preprocessed_dir / split.value / sequence.name / "bbox.npy"
        labels = np.load(path)

        _, unique_indices = np.unique(labels["t"], return_index=True)  # assumes sorted labels by time
        labels = np.split(labels, unique_indices)

        # some sequences may not have any labels due to preprocessing
        if len(labels) == 0:
            return {}

        labels = labels[1:]  # skip the first empty label group

        label_index = 0
        chunk_index_to_labels = {}
        for chunk_index, chunk in enumerate(sequence.chunks):
            # we assume that chunks durations are aligned with label times
            if label_index < len(labels) and chunk.end == labels[label_index][0]["t"].item():
                chunk_index_to_labels[chunk_index] = labels[label_index]
                label_index += 1
        return chunk_index_to_labels

    def train_dataloader(self):
        random_loader = torch.utils.data.DataLoader(
            batch_size=self.batch_size_random,
            collate_fn=ObjectDetectionCollator(self.config),
            dataset=self.train_random_dataset,
            num_workers=self.num_workers_random,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

        streaming_loader = torch.utils.data.DataLoader(
            batch_size=self.batch_size_streaming,
            collate_fn=ObjectDetectionCollator(self.config),
            dataset=self.train_streaming_dataset,
            num_workers=self.num_workers_streaming,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.streaming_worker_init_fn,
        )

        return {
            "random": random_loader,
            "streaming": streaming_loader,
        }

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=self.batch_size,
            collate_fn=ObjectDetectionCollator(self.config),
            dataset=self.val_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.streaming_worker_init_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=self.batch_size,
            collate_fn=ObjectDetectionCollator(self.config),
            dataset=self.test_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.streaming_worker_init_fn,
        )

    def streaming_worker_init_fn(self, worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        per_worker = math.ceil(len(dataset.sequences) / worker_info.num_workers)
        start = worker_info.id * per_worker
        end = min(start + per_worker, len(dataset.sequences))
        dataset.start = start
        dataset.end = end
        dataset.worker_id = worker_id
