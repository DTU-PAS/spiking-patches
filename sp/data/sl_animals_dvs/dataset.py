import dataclasses
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F

from sp import augmentations
from sp.configs import Config
from sp.data_types import ClassificationEventsData
from sp.io import load_events
from sp.sl_animals_dvs import HEIGHT
from sp.sl_animals_dvs import NUM_CLASSES
from sp.sl_animals_dvs import WIDTH
from sp.timers import Timer


@dataclasses.dataclass
class Sample:
    id: str
    path: Path
    label: int
    user: str


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config: Config, samples: list[Sample], augment: bool = False):
        super().__init__()
        self.augment = augment
        self.samples = samples
        self.max_events = config.max_events

        if augment:
            self.augmentation = augmentations.Compose(
                [
                    augmentations.CutMix(height=HEIGHT, width=WIDTH),
                    augmentations.Chance(
                        augmentations.HorizontalFlip(width=WIDTH),
                    ),
                    augmentations.OneOf(
                        [
                            augmentations.Identity(),
                            augmentations.DropByArea(height=HEIGHT, width=WIDTH, max_ratio=0.5),
                            augmentations.DropByTime(),
                            augmentations.DropEvent(),
                            augmentations.HorizontalShear(width=WIDTH),
                            augmentations.Rolling(height=HEIGHT, width=WIDTH),
                            augmentations.Rotation(height=HEIGHT, width=WIDTH),
                        ]
                    ),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample, sample_id = self.load_sample(index)

        if self.augment:
            mix = self.load_mix_sample(index)
            with Timer("augmentation"):
                sample = self.augmentation(sample, mix=mix)

        if self.max_events is not None and len(sample.events) > self.max_events:
            sample.events = sample.events[: self.max_events]

        return ClassificationEventsData(events=sample.events, label=sample.label.classification, id=sample_id)

    @Timer("load_sample")
    def load_sample(self, index: int) -> tuple[augmentations.Sample, str]:
        sample = self.samples[index]
        events = load_events(sample.path)
        label = torch.tensor([sample.label - 1])
        label = F.one_hot(label, num_classes=NUM_CLASSES).squeeze(0)
        label = label.to(torch.float32)
        label = augmentations.Label(classification=label)
        return augmentations.Sample(events=events, label=label), sample.id

    def load_mix_sample(self, sample_index: int) -> augmentations.Sample:
        # Pick a different sample to use for mixing
        mix_index = randint(0, len(self.samples) - 2)
        mix_index = mix_index + 1 if mix_index >= sample_index else mix_index
        mix, _ = self.load_sample(mix_index)
        return mix
