import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sp.configs import Dataset
from sp.configs import Split
from sp.etram import DATASET_NAME as ETRAM_DATASET_NAME
from sp.etram import HEIGHT as ETRAM_HEIGHT
from sp.etram import MIN_BOX_DIAG as ETRAM_MIN_BOX_DIAG
from sp.etram import MIN_BOX_SIDE as ETRAM_MIN_BOX_SIDE
from sp.etram import WIDTH as ETRAM_WIDTH
from sp.gen1 import DATASET_NAME as GEN1_DATASET_NAME
from sp.gen1 import HEIGHT as GEN1_HEIGHT
from sp.gen1 import MIN_BOX_DIAG as GEN1_MIN_BOX_DIAG
from sp.gen1 import MIN_BOX_SIDE as GEN1_MIN_BOX_SIDE
from sp.gen1 import WIDTH as GEN1_WIDTH
from sp.io import save_events
from sp.one_mpx import DATASET_NAME as ONE_MPX_DATASET_NAME
from sp.one_mpx import HEIGHT as ONE_MPX_HEIGHT
from sp.one_mpx import MIN_BOX_DIAG as ONE_MPX_MIN_BOX_DIAG
from sp.one_mpx import MIN_BOX_SIDE as ONE_MPX_MIN_BOX_SIDE
from sp.one_mpx import WIDTH as ONE_MPX_WIDTH
from sp.paths import get_dataset_dir
from sp.preprocessing.events import preprocess_events
from sp.preprocessing.loaders import ETraMLoader
from sp.preprocessing.loaders import PropheseeLoader

LABELS_DTYPE = np.dtype(
    [
        ("sequence", np.uint32),
        ("chunk", np.uint32),
        ("event", np.uint32),
        ("t", np.uint64),
        ("x", np.float32),
        ("y", np.float32),
        ("w", np.float32),
        ("h", np.float32),
        ("class_id", np.uint8),
    ]
)


class ObjectDetectionPreprocessor:
    """Preprocess one of the object detection datasets: 1mpx, eTraM, and GEN1."""

    def __init__(
        self,
        chunk_duration_ms: int,
        dataset: Dataset,
        limit: int | None,
        test: bool,
        train: bool,
        val: bool,
    ):
        super().__init__()

        if dataset not in (Dataset.etram, Dataset.gen1, Dataset.one_mpx):
            raise ValueError(f"Unsupported dataset for object detection preprocessing: {dataset.value}")

        self.chunk_duration_us = chunk_duration_ms * 1000
        self.dataset = dataset
        self.limit = limit
        self.train = train
        self.val = val
        self.test = test
        self.source_dir = self.get_source_dir(dataset)
        self.output_dir = self.get_output_dir(dataset)
        self.height, self.width, self.min_box_side, self.min_box_diag = self.get_dimensions(dataset)
        self.timestamp_key = "ts" if dataset == Dataset.gen1 else "t"
        self.loader_cls = ETraMLoader if dataset == Dataset.etram else PropheseeLoader
        self.file_suffix = "_td.h5" if dataset == Dataset.etram else "_td.dat"

    def get_source_dir(self, dataset: Dataset) -> Path:
        match dataset:
            case Dataset.etram:
                return get_dataset_dir(ETRAM_DATASET_NAME) / "Static" / "HDF5"
            case Dataset.gen1:
                return get_dataset_dir(GEN1_DATASET_NAME)
            case Dataset.one_mpx:
                return get_dataset_dir(ONE_MPX_DATASET_NAME)

    def get_output_dir(self, dataset: Dataset) -> Path:
        match dataset:
            case Dataset.etram:
                name = ETRAM_DATASET_NAME
            case Dataset.gen1:
                name = GEN1_DATASET_NAME
            case Dataset.one_mpx:
                name = ONE_MPX_DATASET_NAME
        output_name = f"{name}-preprocessed"
        return get_dataset_dir(output_name, raise_missing=False)

    def get_dimensions(self, dataset: Dataset) -> tuple[int, int, int, int]:
        match dataset:
            case Dataset.etram:
                return ETRAM_HEIGHT, ETRAM_WIDTH, ETRAM_MIN_BOX_SIDE, ETRAM_MIN_BOX_DIAG
            case Dataset.gen1:
                return GEN1_HEIGHT, GEN1_WIDTH, GEN1_MIN_BOX_SIDE, GEN1_MIN_BOX_DIAG
            case Dataset.one_mpx:
                return ONE_MPX_HEIGHT, ONE_MPX_WIDTH, ONE_MPX_MIN_BOX_SIDE, ONE_MPX_MIN_BOX_DIAG

    def get_split_paths(self, split: Split) -> list[Path]:
        match self.dataset:
            case Dataset.etram:
                return list(self.source_dir.glob(f"{split.value}*_bbox.npy"))
            case _:
                split_dir = self.source_dir / split.value
                return list(split_dir.glob("*_bbox.npy"))

    def preprocess(self):
        # save config
        config = {
            "dataset": self.dataset.value,
            "chunk_duration_ms": self.chunk_duration_us // 1000,
            "limit": self.limit,
        }
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        if self.train:
            self.preprocess_split(Split.train)
        if self.val:
            self.preprocess_split(Split.val)
        if self.test:
            self.preprocess_split(Split.test)

    def preprocess_split(self, split: Split):
        labels_paths = self.get_split_paths(split)
        if self.limit is not None:
            labels_paths = labels_paths[: self.limit]
        output_dir = self.output_dir / split.value
        output_dir.mkdir(exist_ok=True, parents=True)
        index = []
        sequence_id = 0
        for labels_path in tqdm(labels_paths, desc=f"Preprocessing {split.value}"):
            sequence_dir = output_dir / labels_path.stem.removesuffix("_bbox")
            sequence_dir.mkdir(exist_ok=True, parents=True)
            index.append(self.preprocess_sequence(split, sequence_dir, sequence_id, labels_path))
            sequence_id += 1
        index_path = output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f)

    def preprocess_sequence(self, split: Split, output_dir: Path, sequence_id: int, labels_path: Path) -> dict:
        sequence_name = labels_path.stem.removesuffix("_bbox")
        events_filename = sequence_name + self.file_suffix
        events_path = labels_path.parent / events_filename
        chunks = []
        loader = self.loader_cls(events_path)
        loader.seek_time(self.chunk_duration_us)
        events = loader.load_past()
        events = preprocess_events(events, self.height, self.width, 0)
        chunk_index = 0
        if len(events) > 0:
            start_time = 0
            end_time = self.chunk_duration_us - 1
            chunks.append(
                {
                    "index": chunk_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "count": len(events),
                    "t": events.t,
                }
            )
            output_path = output_dir / f"{chunk_index}.h5"
            save_events(output_path, events)
            chunk_index += 1
        time_offset = self.chunk_duration_us
        while not loader.done():
            events = loader.load_delta_t(self.chunk_duration_us)
            events = preprocess_events(events, self.height, self.width, time_offset)
            if len(events) > 0:
                start_time = time_offset
                end_time = time_offset + self.chunk_duration_us - 1
                chunks.append(
                    {
                        "index": chunk_index,
                        "start_time": start_time,
                        "end_time": end_time,
                        "count": len(events),
                        "t": events.t,
                    }
                )
                output_path = output_dir / f"{chunk_index}.h5"
                save_events(output_path, events)
                chunk_index += 1
            time_offset += self.chunk_duration_us
        loader.close()
        labels = np.load(labels_path)
        labels = self.preprocess_labels(labels, split, sequence_id, chunks)
        labels_path = output_dir / "bbox.npy"
        np.save(labels_path, labels)
        for chunk in chunks:
            # We only needed to store timestamps for preprocessing labels, so we can remove them now
            del chunk["t"]
        return {"name": sequence_name, "chunks": chunks}

    def preprocess_labels(self, labels: np.ndarray, split: Split, sequence: int, chunks: list[dict]) -> np.ndarray:
        if self.dataset == Dataset.one_mpx:
            labels = self.remove_ignored_classes(labels)
        labels = self.crop_boxes_outside_of_image(labels)
        labels = self.remove_small_boxes(labels)
        if split == Split.train:
            labels = self.remove_faulty_huge_bbox(labels)
        labels = self.add_chunk_location(labels, sequence, chunks)
        return labels

    def remove_ignored_classes(self, labels: np.ndarray) -> np.ndarray:
        """
        Source: https://github.com/uzh-rpg/RVT/blob/master/scripts/genx/preprocess_dataset.py#L263C1-L272C1

        Original 1mpx labels: pedestrian, two wheeler, car, truck, bus, traffic sign, traffic light
        1mpx labels to keep: pedestrian, two wheeler, car
        1mpx labels to remove: truck, bus, traffic sign, traffic light

        class_id in {0, 1, 2, 3, 4, 5, 6} in the order mentioned above
        """
        keep = labels["class_id"] <= 2
        labels = labels[keep]
        return labels

    def crop_boxes_outside_of_image(self, labels: np.ndarray) -> np.ndarray:
        left = labels["x"]
        right = labels["x"] + labels["w"]
        top = labels["y"]
        bottom = labels["y"] + labels["h"]

        left = np.clip(left, 0, self.width - 1)
        right = np.clip(right, 0, self.width - 1)
        top = np.clip(top, 0, self.height - 1)
        bottom = np.clip(bottom, 0, self.height - 1)

        width = right - left
        height = bottom - top

        assert np.all(width >= 0)
        assert np.all(height >= 0)

        cropped_labels = labels.copy()
        cropped_labels["x"] = left
        cropped_labels["y"] = top
        cropped_labels["w"] = width
        cropped_labels["h"] = height

        # remove boxes with zero area
        # this may happen if the box is completely outside of the image
        mask = (width > 0) & (height > 0)
        cropped_labels = cropped_labels[mask]

        return cropped_labels

    def remove_small_boxes(self, labels: np.ndarray) -> np.ndarray:
        width = labels["w"]
        height = labels["h"]
        diagonal_mask = width**2 + height**2 >= self.min_box_diag**2
        width_mask = width >= self.min_box_side
        height_mask = height >= self.min_box_side
        mask = diagonal_mask & width_mask & height_mask
        return labels[mask]

    def remove_faulty_huge_bbox(self, labels: np.ndarray) -> np.ndarray:
        """There are some labels which span the frame horizontally without actually covering an object.
        Source: https://github.com/uzh-rpg/RVT/blob/master/scripts/genx/preprocess_dataset.py#L222
        """
        width = labels["w"]
        max_width = (9 * self.width) // 10
        mask = width <= max_width
        labels = labels[mask]
        return labels

    def add_chunk_location(self, labels: list[np.ndarray], sequence: int, chunks: list[dict]) -> np.ndarray:
        if len(labels) == 0:
            return np.empty(0, dtype=LABELS_DTYPE)
        labels = labels[np.argsort(labels[self.timestamp_key])]
        _, indices = np.unique(labels[self.timestamp_key], return_index=True)
        groups = np.split(labels, indices)[1:]
        labels = []
        chunk_index = 0
        for group in groups:
            timestamp = group[0][self.timestamp_key]
            while chunk_index < len(chunks) and timestamp > chunks[chunk_index]["end_time"]:
                chunk_index += 1
            chunk_index = min(chunk_index, len(chunks) - 1)
            chunk_labels = np.empty(len(group), dtype=LABELS_DTYPE)
            chunk_labels["sequence"] = sequence
            chunk_labels["chunk"] = chunk_index
            chunk_labels["event"] = np.searchsorted(chunks[chunk_index]["t"], timestamp, side="right")
            chunk_labels["t"] = group[self.timestamp_key]
            chunk_labels["x"] = group["x"]
            chunk_labels["y"] = group["y"]
            chunk_labels["w"] = group["w"]
            chunk_labels["h"] = group["h"]
            chunk_labels["class_id"] = group["class_id"]
            labels.append(chunk_labels)
        labels = np.concatenate(labels)
        return labels
