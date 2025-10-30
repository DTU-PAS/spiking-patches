import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np

from sp.events import Events


@dataclasses.dataclass
class Chunk:
    id: int
    count: int
    start: int
    end: int


@dataclasses.dataclass
class Sequence:
    source_dir: Path
    name: str
    chunks: list[Chunk]


def load_sequences(source_dir: Path) -> list[Sequence]:
    index_path = source_dir / "index.json"
    sequences = json.loads(index_path.read_text())
    return [
        Sequence(
            source_dir=source_dir,
            name=sequence["name"],
            chunks=[
                Chunk(id=chunk["index"], count=chunk["count"], start=chunk["start_time"], end=chunk["end_time"])
                for chunk in sequence["chunks"]
            ],
        )
        for sequence in sequences
    ]


def load_chunks(
    sequences: list[Sequence],
    sequence_index: int,
    end_chunk_index: int,
    max_chunks: int,
) -> Events:
    sequence = sequences[sequence_index]
    chunks = sequence.chunks
    start_chunk_index = max(0, end_chunk_index - max_chunks + 1)
    chunks = chunks[start_chunk_index : end_chunk_index + 1]

    events = []
    for chunk in chunks:
        path = sequence.source_dir / sequence.name / f"{chunk.id}.h5"
        events.append(load_events(path))

    return Events(
        x=np.concatenate([chunk_events.x for chunk_events in events]),
        y=np.concatenate([chunk_events.y for chunk_events in events]),
        t=np.concatenate([chunk_events.t for chunk_events in events]),
        p=np.concatenate([chunk_events.p for chunk_events in events]),
    )


def load_events(input_path: Path) -> Events:
    with h5py.File(input_path, "r") as file:
        x = file["x"][()]
        y = file["y"][()]
        t = file["t"][()]
        p = file["p"][()]

    t = t.astype(np.uint64)

    return Events(x=x, y=y, t=t, p=p)


def save_events(output_path: Path, events: Events, compress: bool = True):
    kwargs = {"compression": "gzip"} if compress else {}
    with h5py.File(output_path, "w") as out_file:
        out_file.create_dataset(
            "x",
            data=events.x,
            dtype=np.uint16,
            **kwargs,
        )
        out_file.create_dataset(
            "y",
            data=events.y,
            dtype=np.uint16,
            **kwargs,
        )
        out_file.create_dataset(
            "t",
            data=events.t,
            dtype=np.uint32,
            **kwargs,
        )
        out_file.create_dataset(
            "p",
            data=events.p,
            dtype=bool,
            **kwargs,
        )
