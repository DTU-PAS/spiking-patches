import random

import numpy as np
import torch

from sp import augmentations
from sp.configs import Config
from sp.configs import TokenizerType
from sp.data.object_detection.dataset_base import DatasetBase
from sp.data_types import ObjectDetectionTokensData
from sp.data_types import Tokens
from sp.events import Events
from sp.io import Sequence
from sp.io import load_chunks
from sp.timers import Timer
from sp.tokenizer import StreamingTokenizer


class DatasetStreaming(torch.utils.data.IterableDataset):
    def __init__(
        self,
        batch_size: int,
        config: Config,
        chunk_index_to_labels: dict[int, dict[int, np.ndarray]],
        sequences: list[Sequence],
        augment: bool = False,
        training: bool = False,
    ):
        super().__init__()

        self.augment = augment
        self.batch_size = batch_size
        self.config = config
        self.chunk_index_to_labels = chunk_index_to_labels
        self.sequences = sequences
        self.training = training

        # These will be re-initialized in the worker_init_fn of the datamodule
        self.start = 0
        self.end = len(sequences)
        self.worker_id = 0

    def __iter__(self):
        return iter(self.run(self.start, self.end))

    def run(self, sequence_start_index: int, sequence_end_index: int):
        # This ensures that sequences are shuffled every epoch
        # but the order of items within each sequence is preserved
        sequence_indices = list(range(sequence_start_index, sequence_end_index))
        if self.training:
            random.shuffle(sequence_indices)
        sequence_indices = iter(sequence_indices)

        workers: list[Worker] = [
            Worker(
                augment=self.augment,
                batch_index=batch_index,
                config=self.config,
                chunk_index_to_labels=self.chunk_index_to_labels,
                sequences=self.sequences,
                training=self.training,
                worker_id=self.worker_id,
            )
            for batch_index in range(self.batch_size)
        ]

        while len(workers) > 0:
            finished_workers = []
            for worker in workers:
                finished = False
                try:
                    while worker.done():  # loop to find sequence with labels when training
                        sequence_index = next(sequence_indices)
                        worker.start(sequence_index)
                except StopIteration:
                    finished = True
                if not finished:
                    data = worker.step()
                    yield data
                finished_workers.append(finished)
            workers = [worker for worker, finished in zip(workers, finished_workers, strict=True) if not finished]


class Worker(DatasetBase):
    def __init__(
        self,
        augment: bool,
        batch_index: int,
        config: Config,
        chunk_index_to_labels: dict[int, dict[int, np.ndarray]],
        sequences: list[Sequence],
        training: bool,
        worker_id: int,
    ):
        super().__init__(
            augment=augment,
            config=config,
            chunk_index_to_labels=chunk_index_to_labels,
            sequences=sequences,
            streaming=True,
        )

        self.batch_index = batch_index
        self.tokenizer = None if config.tokenizer == TokenizerType.none else StreamingTokenizer(config)
        self.training = training
        self.worker_id = worker_id

        # Initialize these to ensure that self.done() is true initially
        self.chunk_range_index = 0
        self.chunk_ranges = []

    def done(self) -> bool:
        return self.chunk_range_index >= len(self.chunk_ranges)

    def start(self, sequence_index: int):
        self.sequence_index = sequence_index
        self.chunk_range_index = 0
        sequence = self.sequences[sequence_index]
        self.chunk_ranges = self.get_labelled_ranges(sequence_index) if self.training else [(0, len(sequence.chunks))]
        self.start_chunk_index = 0 if len(self.chunk_ranges) == 0 else self.chunk_ranges[0][0]

        if self.tokenizer is not None:
            self.tokenizer.reset()

        if self.should_augment:
            self.current_sequence_augmentation = self.augmentation.sample()

        self.is_first_sample = True

    def step(self) -> ObjectDetectionTokensData:
        chunk_end = self.chunk_ranges[self.chunk_range_index][1]

        start_chunk_index = self.start_chunk_index
        end_chunk_index = min(start_chunk_index + self.sequence_length, chunk_end)

        events = load_chunks(
            sequences=self.sequences,
            sequence_index=self.sequence_index,
            end_chunk_index=end_chunk_index - 1,
            max_chunks=end_chunk_index - start_chunk_index,
        )
        split_times = self.get_split_times(self.sequence_index, start_chunk_index, end_chunk_index)
        labels = self.load_labels(self.sequence_index, start_chunk_index, end_chunk_index)
        inputs, augmented_labels = self.process_inputs_and_labels(events, labels)
        inputs = self.split_inputs(split_times, inputs)
        inputs = self.process_split_inputs(inputs, split_times)
        labels = self.split_labels(split_times, augmented_labels)
        prophesee_labels = self.convert_to_prophesee_labels(labels)
        if self.is_first_sample:
            reset = [True] + ([False] * (len(inputs) - 1))
            self.is_first_sample = False
        else:
            reset = [False] * len(inputs)
        yolox_labels = self.convert_to_yolox_labels(labels)
        assert len(inputs) == len(prophesee_labels) == len(yolox_labels) == len(split_times)

        if end_chunk_index == chunk_end:
            self.chunk_range_index += 1
            self.start_chunk_index = 0 if self.done() else self.chunk_ranges[self.chunk_range_index][0]
            # There may be a large gap between chunks, so we reset the tokenizer
            # and ensure that the RNN state will be reset for the next chunk range.
            if self.tokenizer is not None:
                self.tokenizer.reset()
            self.is_first_sample = True
        else:
            self.start_chunk_index = end_chunk_index

        return ObjectDetectionTokensData(
            batch_index=[self.batch_index] * len(inputs),
            inputs=inputs,
            prophesee_labels=prophesee_labels,
            prediction_time=split_times,
            reset=reset,
            sequence_id=[self.sequence_index] * len(inputs),
            yolox_labels=yolox_labels,
            worker_id=self.worker_id,
        )

    @Timer("augment")
    def augment(self, sample: augmentations.Sample) -> augmentations.Sample:
        return self.current_sequence_augmentation(sample)

    @Timer("tokenize")
    def tokenize(self, events: Events) -> Tokens:
        prediction_time = 0 if events.x.size == 0 else events.t[-1].item()
        return self.tokenizer(events, prediction_time=prediction_time)

    def get_labelled_ranges(self, sequence_index: int) -> list[tuple[int, int]]:
        """
        Find ranges that guarantee at least one label in each range.
        This is needed for training, but not for evaluation.

        Source: https://github.com/uzh-rpg/RVT/blob/master/data/genx_utils/sequence_for_streaming.py
        """
        indices = sorted(self.chunk_index_to_labels[sequence_index].keys())
        indices = np.array(indices)
        if len(indices) == 0:
            # No labels in this sequence, return empty ranges.
            # This corresponds to skipping the sequence during training.
            return []

        meta_indices_stop = np.flatnonzero(np.diff(indices) > self.sequence_length)

        meta_indices_start = np.concatenate((np.atleast_1d(0), meta_indices_stop + 1))
        meta_indices_stop = np.concatenate((meta_indices_stop, np.atleast_1d(len(indices) - 1)))

        out = []
        for meta_idx_start, meta_idx_stop in zip(meta_indices_start, meta_indices_stop, strict=True):
            idx_start = max(indices[meta_idx_start] - self.sequence_length + 1, 0)
            idx_stop = indices[meta_idx_stop] + 1
            out.append((idx_start, idx_stop))
        return out
