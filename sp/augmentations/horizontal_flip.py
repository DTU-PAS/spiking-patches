import dataclasses
from typing import Callable

import numpy as np

from sp.augmentations.augmentation import Augmentation
from sp.augmentations.augmentation import ObjectDetectionLabel
from sp.augmentations.augmentation import Sample
from sp.events import Events


class HorizontalFlip(Augmentation):
    def __init__(self, width: int, classification_label_mapper: Callable | None = None):
        self.width = width
        self.classification_label_mapper = classification_label_mapper

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        events = self.flip_events(sample.events)

        label = sample.label
        if label is not None:
            classification = label.classification
            if classification is not None and self.classification_label_mapper is not None:
                classification = self.classification_label_mapper(classification)

            object_detection = label.object_detection
            if object_detection is not None:
                object_detection = self.flip_object_detection_label(object_detection)

            label = dataclasses.replace(label, classification=classification, object_detection=object_detection)

        return Sample(events=events, label=label)

    def sample_parameters(self):
        return None

    def flip_events(self, events: Events) -> Events:
        x = (self.width - 1) - events.x
        return dataclasses.replace(events, x=x)

    def flip_object_detection_label(self, object_detection: list[ObjectDetectionLabel]) -> list[ObjectDetectionLabel]:
        flipped = []
        for label in object_detection:
            x = self.width - (label.x + label.width)
            flipped.append(dataclasses.replace(label, x=x))
        return flipped
