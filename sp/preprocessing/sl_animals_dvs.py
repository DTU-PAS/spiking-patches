import dataclasses
import json
import math
from hashlib import md5
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.contrib.concurrent import process_map

from sp.aedat import AedatReader
from sp.io import save_events
from sp.paths import get_dataset_dir
from sp.preprocessing.events import preprocess_events
from sp.sl_animals_dvs import DATASET_NAME
from sp.sl_animals_dvs import HEIGHT
from sp.sl_animals_dvs import WIDTH


class SLAnimalsDVSPreprocessor:
    """Preprocess SL-Animals-DVS."""

    def __init__(
        self,
        limit: int | None,
        max_duration_ms: int,
        max_workers: int | None,
    ):
        super().__init__()
        self.limit = limit
        self.max_duration_us = max_duration_ms * 1_000
        self.max_workers = max_workers
        self.source_dir = get_dataset_dir(DATASET_NAME)
        self.output_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed", raise_missing=False)
        self.aedat_dir = self.source_dir / "allusers_aedat"
        self.tags_dir = self.source_dir / "tags_updated_19_08_2020"

    def preprocess(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        recording_paths = list(self.aedat_dir.glob("*.aedat"))
        if self.limit is not None:
            recording_paths = recording_paths[: self.limit]
        kwargs = {"chunksize": 1, "desc": "Preprocessing", "unit": "recording"}
        if self.max_workers is not None:
            kwargs["max_workers"] = self.max_workers
        recordings = process_map(self.preprocess_recording, recording_paths, **kwargs)
        samples = [sample for recording in recordings for sample in recording]
        output_path = self.output_dir / "index.json"
        output_path.write_text(json.dumps(samples))

    def preprocess_recording(self, recording_path: Path):
        with AedatReader(recording_path) as aedat:
            events = aedat.read()

        tags_path = self.tags_dir / f"{recording_path.stem}.csv"
        tags = pd.read_csv(tags_path)
        user = recording_path.stem

        samples = []
        for tag_index in range(len(tags)):
            tag = tags.iloc[tag_index]
            label = tag["class"].item()
            start = tag["startTime_ev"]
            end = tag["endTime_ev"]

            gesture_events = events[start:end]
            t = gesture_events.t - gesture_events.t[0]
            gesture_events = dataclasses.replace(gesture_events, t=t)
            gesture_events = preprocess_events(
                events=gesture_events,
                height=HEIGHT,
                min_time=0,
                width=WIDTH,
            )

            params_hash = self.params_to_hash([("user", user), ("start", start), ("end", end)])

            gesture_id = f"{recording_path.stem}_{label}"

            duration_us = gesture_events.t[-1] - gesture_events.t[0]
            num_slices = math.ceil(duration_us / self.max_duration_us)

            for slice_index in range(num_slices):
                start = slice_index * self.max_duration_us
                end = start + self.max_duration_us
                mask = (gesture_events.t >= start) & (gesture_events.t < end)
                slice_events = gesture_events.mask(mask)
                slice_events = dataclasses.replace(slice_events, t=slice_events.t - start)

                if len(slice_events) == 0:
                    continue

                sample_path = self.output_dir / f"{params_hash}_{slice_index}.h5"
                save_events(sample_path, slice_events, compress=False)

                relative_sample_path = str(sample_path.relative_to(self.output_dir))
                samples.append({"gesture_id": gesture_id, "label": label, "path": relative_sample_path, "user": user})

        return samples

    def params_to_hash(self, params: list[tuple[str, Any]]):
        params = [f"{name}={value}" for name, value in params if value is not None]
        params = "-".join(params).encode()
        params_hash = md5(params).hexdigest()
        return params_hash[:16]
