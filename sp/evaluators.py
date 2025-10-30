from collections import defaultdict

import numpy as np

from sp.configs import Dataset
from sp.configs import ObjectDetectionEvaluatorConfig
from sp.data_types import PROPHESEE_BBOX_DTYPE
from sp.data_types import ObjectDetectionPrediction
from sp.prophesee.evaluation import evaluate_list


class ObjectDetectionEvaluator:
    def __init__(self, config: ObjectDetectionEvaluatorConfig):
        self.camera = self.get_camera(config.dataset)
        self.height = config.height
        self.width = config.width
        self.skip_time_us = config.skip_time_us
        self.time_tol = config.time_tol

        # sequence id -> list of predictions for that sequence
        self.predictions: dict[int, dict[int, ObjectDetectionPrediction]] = defaultdict(dict)

        # (sequence id, timestamp) -> labels for that sequence and timestamp
        self.labels: dict[int, dict[int, np.ndarray]] = defaultdict(dict)

    def add_predictions(self, sequence_id: int, prediction: ObjectDetectionPrediction):
        timestamp = prediction.t[0].item()
        self.predictions[sequence_id][timestamp] = prediction

    def add_labels(self, sequence_id: int, labels: np.ndarray):
        timestamp = labels[0]["t"].item()
        self.labels[sequence_id][timestamp] = labels

    def evaluate(self) -> dict[str, float]:
        """Evaluate the current buffer of predictions/labels and reset the buffer."""

        sequence_ids = sorted(set(self.predictions.keys()) | set(self.labels.keys()))

        all_predictions = []
        all_labels = []

        for sequence_id in sequence_ids:
            sequence_predictions_dict = self.predictions[sequence_id]
            sequence_predictions = []
            for timestamp in sorted(sequence_predictions_dict.keys()):
                sequence_predictions.append(sequence_predictions_dict[timestamp].numpy())
            sequence_predictions = self.concat_or_empty(sequence_predictions)
            all_predictions.append(sequence_predictions)

            sequence_labels_dict = self.labels[sequence_id]
            sequence_labels = []
            for timestamp in sorted(sequence_labels_dict.keys()):
                sequence_labels.append(sequence_labels_dict[timestamp])
            sequence_labels = self.concat_or_empty(sequence_labels)
            all_labels.append(sequence_labels)

        self.predictions.clear()
        self.labels.clear()

        return evaluate_list(
            all_predictions,
            all_labels,
            camera=self.camera,
            height=self.height,
            skip_ts=self.skip_time_us,
            time_tol=self.time_tol,
            width=self.width,
        )

    def get_camera(self, dataset: Dataset) -> str:
        match dataset:
            case Dataset.etram:
                return "gen4"
            case Dataset.gen1:
                return "gen1"
            case Dataset.one_mpx:
                return "gen4"
            case _:
                raise ValueError(f"Unsupported dataset: {dataset}")

    def concat_or_empty(self, values: list[np.ndarray]) -> np.ndarray:
        if len(values) == 0:
            return np.empty((0,), dtype=PROPHESEE_BBOX_DTYPE)
        return np.concatenate(values, axis=0)
