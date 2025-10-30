import numpy as np

from sp.events import Events


def events_to_volume(
    events: Events, buckets: int, height: int, width: int, start_time: int | None = None, duration: int | None = None
):
    """Converts a sequence of events to a 2D-volume.

    x and y coordinates are wrapped around the frame.
    """

    shape = (2, buckets, height, width)
    if len(events) == 0:
        return np.zeros(shape, dtype=np.int64)

    x = events.x.astype(np.int64, casting="safe")
    y = events.y.astype(np.int64, casting="safe")
    p = events.p.astype(np.int64, casting="safe")

    x = x % width
    y = y % height

    if start_time is None:
        start_time = events.t[0]

    if duration is None:
        duration = events.t[-1] - start_time

    t_relative = (events.t - start_time) / duration
    t_bin = np.floor(t_relative * buckets)
    t_bin = t_bin.astype(np.int64)
    t_bin = np.minimum(t_bin, buckets - 1)

    area_size = height * width
    volume_size = buckets * area_size

    positions = (p * volume_size) + (t_bin * area_size) + (y * width) + x

    length = np.prod(shape)

    assert positions.min() >= 0
    assert positions.max() <= length

    volume = np.bincount(positions, minlength=length)
    volume = volume.reshape(shape)

    return volume


def events_to_logspace_volume(
    events: Events,
    buckets: int,
    height: int,
    width: int,
    time_base_us: int = 1000,
    power_base: int = 2,
):
    """Converts a sequence of events to a 2D-volume. The distribution of the time bins is logarithmic.

    x and y coordinates are wrapped around the frame.
    """

    shape = (2, buckets, height, width)
    if len(events) == 0:
        return np.zeros(shape, dtype=np.int64)

    x = events.x.astype(np.int64, casting="safe")
    y = events.y.astype(np.int64, casting="safe")
    p = events.p.astype(np.int64, casting="safe")

    x = x % width
    y = y % height

    t = events.t[-1] - events.t

    bucket_ends = time_base_us * np.power(power_base, np.arange(buckets))
    t_bucket = np.digitize(t, bucket_ends)
    t_bucket = np.minimum(t_bucket, buckets - 1)

    area_size = height * width
    volume_size = buckets * area_size

    positions = (p * volume_size) + (t_bucket * area_size) + (y * width) + x

    length = np.prod(shape)

    assert positions.min() >= 0
    assert positions.max() <= length

    volume = np.bincount(positions, minlength=length)
    volume = volume.reshape(shape)

    return volume


def batched_events_to_logspace_volume(
    batch: list[np.ndarray],
    x: list[np.ndarray],
    y: list[np.ndarray],
    t: list[np.ndarray],
    p: list[np.ndarray],
    buckets: int,
    height: int,
    width: int,
    time_base_us: int = 1000,
    power_base: int = 2,
) -> np.ndarray:
    batch_size = len(batch)
    shape = (batch_size, 2, buckets, height, width)
    if batch_size == 0:
        return np.zeros(shape, dtype=np.int64)

    batch = np.concatenate(batch)

    x = np.concatenate(x) % width
    y = np.concatenate(y) % height

    t = [values if len(values) == 0 else values[-1] - values for values in t]
    t = np.concatenate(t)

    p = np.concatenate(p)

    bucket_ends = time_base_us * np.power(power_base, np.arange(buckets))
    t_bucket = np.digitize(t, bucket_ends)
    t_bucket = np.minimum(t_bucket, buckets - 1)

    batch = batch.astype(np.uint32)
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    t_bucket = t_bucket.astype(np.uint32)
    p = p.astype(np.uint32)

    area_size = height * width
    volume_size = buckets * area_size
    batch_volume_size = 2 * volume_size

    positions = (batch * batch_volume_size) + (p * volume_size) + (t_bucket * area_size) + (y * width) + x

    positions = positions.astype(np.int64)
    length = np.prod(shape, dtype=np.int64)

    assert positions.size == 0 or positions.min() >= 0
    assert positions.size == 0 or positions.max() <= length

    volume = np.bincount(positions, minlength=length)
    volume = volume.reshape(shape)

    return volume
