import dataclasses

import numpy as np
import torch

from sp import augmentations
from sp.configs import Config
from sp.configs import TokenizerType
from sp.data.object_detection.dataset_base import DatasetBase
from sp.data_types import ObjectDetectionTokensData
from sp.events import Events
from sp.io import Sequence
from sp.io import load_chunks
from sp.timers import Timer
from sp.tokenizer import BatchTokenizer


@dataclasses.dataclass
class Sample:
    sequence: int
    chunk: int


class DatasetRandom(torch.utils.data.Dataset, DatasetBase):
    def __init__(
        self,
        config: Config,
        chunk_index_to_labels: dict[int, dict[int, np.ndarray]],
        samples: list[Sample],
        sequences: list[Sequence],
        augment: bool = False,
    ):
        super().__init__(
            augment=augment,
            config=config,
            chunk_index_to_labels=chunk_index_to_labels,
            sequences=sequences,
            streaming=False,
        )

        self.samples = samples
        self.tokenizer = None if config.tokenizer == TokenizerType.none else BatchTokenizer(config)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        end_chunk_index = sample.chunk + 1
        start_chunk_index = max(0, end_chunk_index - self.sequence_length)
        events = self.load_events(index)
        split_times = self.get_split_times(sample.sequence, start_chunk_index, end_chunk_index)
        labels = self.load_labels(sample.sequence, start_chunk_index, end_chunk_index)
        inputs, augmented_labels = self.process_inputs_and_labels(events, labels)
        inputs = self.split_inputs(split_times, inputs)
        inputs = self.process_split_inputs(inputs, split_times)
        labels = self.split_labels(split_times, augmented_labels)
        prophesee_labels = self.convert_to_prophesee_labels(labels)
        yolox_labels = self.convert_to_yolox_labels(labels)
        reset = [True] + ([False] * (len(inputs) - 1))
        assert len(inputs) == len(prophesee_labels) == len(yolox_labels) == len(split_times)
        return ObjectDetectionTokensData(
            batch_index=None,
            inputs=inputs,
            prophesee_labels=prophesee_labels,
            prediction_time=split_times,
            reset=reset,
            sequence_id=[sample.sequence] * len(inputs),
            yolox_labels=yolox_labels,
            worker_id=-1,  # Worker ID is not used in random access datasets
        )

    @Timer("augment")
    def augment(self, sample: augmentations.Sample) -> augmentations.Sample:
        return self.augmentation(sample)

    @Timer("tokenize")
    def tokenize(self, events: Events):
        prediction_time = [0] if events.x.size == 0 else [events.t[-1].item()]
        return self.tokenizer([events], prediction_time=prediction_time)[0]

    @Timer("load_events")
    def load_events(self, index: int):
        sample = self.samples[index]
        return load_chunks(
            sequences=self.sequences,
            sequence_index=sample.sequence,
            end_chunk_index=sample.chunk,
            max_chunks=self.sequence_length,
        )
