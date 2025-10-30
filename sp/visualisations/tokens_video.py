from math import ceil
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from tempfile import mktemp
from typing import BinaryIO

import ffmpeg
import numpy as np

from sp.data_types import Tokens


class TokensVideo:
    def __init__(
        self,
        tokens: Tokens,
        height: int,
        width: int,
        fps: int = 30,
        accumulate: bool = False,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ):
        self.tokens = tokens
        self.height = height
        self.width = width
        self.fps = fps
        self.accumulate = accumulate
        self.boxes = boxes

    def __enter__(self):
        self.frames_dir = Path(mkdtemp())
        self.video_path = Path(mktemp(suffix=".mp4"))

        self.create_video()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rmtree(self.frames_dir)
        self.video_path.unlink()

    def create_video(self):
        x = np.concat(self.tokens.events_x)
        y = np.concat(self.tokens.events_y)
        t = np.concat(self.tokens.events_t)
        p = np.concat(self.tokens.events_p)

        x = x.astype(np.int64)
        y = y.astype(np.int64)
        p = p.astype(np.int64)

        start_time = t.min()
        end_time = t.max()

        t = t - start_time
        frame_duration_us = 1_000_000 / self.fps
        num_frames = ceil((end_time - start_time) / frame_duration_us)
        t_bucket = (t // frame_duration_us).astype(np.int64)

        area = self.height * self.width
        volume = area * num_frames
        bucket = (p * volume) + (t_bucket * area) + (y * self.width) + x
        num_buckets = 2 * area * num_frames
        frames = np.bincount(bucket, minlength=num_buckets)
        frames = frames.reshape(2, num_frames, self.height, self.width)

        negative = frames[0]
        positive = frames[1]
        images = np.full((num_frames, self.height, self.width, 3), 156, dtype=np.uint8)
        images[negative > positive] = np.array([0, 0, 0], dtype=np.uint8)
        images[(positive >= negative) & (positive > 0)] = np.array([44, 96, 251], dtype=np.uint8)

        if self.boxes is not None:
            border_colour = (255, 0, 0)
            for box in self.boxes:
                x, y, width, height = box
                # draw top border
                images[:, y, x : x + width, :] = border_colour
                # draw bottom border
                images[:, y + height, x : x + width, :] = border_colour
                # draw left border
                images[:, y : y + height, x, :] = border_colour
                # draw right border
                images[:, y : y + height, x + width, :] = border_colour

        video = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            r=self.fps,
            s=f"{self.width}x{self.height}",
            pix_fmt="rgb24",
        )
        video = video.output(str(self.video_path))
        video = video.overwrite_output()
        video = video.run_async(pipe_stdin=True, quiet=True)

        video_bytes = images.tobytes()
        video.stdin.write(video_bytes)
        video.stdin.close()
        video.wait()

    def read(self) -> bytes:
        """Read the video."""
        with open(self.video_path, "rb") as video:
            return video.read()

    def save(self, buffer_or_path: str | Path | BinaryIO):
        """Save the video to a buffer or a file."""
        if isinstance(buffer_or_path, (str, Path)):
            buffer_or_path = open(buffer_or_path, "wb")
        with open(self.video_path, "rb") as video:
            buffer_or_path.write(video.read())
        buffer_or_path.close()
