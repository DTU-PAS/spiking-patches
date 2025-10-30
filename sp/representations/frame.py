import numpy as np

from sp.events import Events


def events_to_frame(events: Events, height: int, width: int):
    """Converts a sequence of events to a 2D-frame.

    x and y coordinates are wrapped around the frame.
    """

    x = events.x.astype(np.int32, casting="safe")
    y = events.y.astype(np.int32, casting="safe")
    p = events.p.astype(np.int32, casting="safe")

    x = x % width
    y = y % height

    area = height * width

    positions = (p * area) + (y * width) + x

    shape = (2, height, width)
    length = np.prod(shape)

    assert positions.min() >= 0
    assert positions.max() <= length

    frame = np.bincount(positions, minlength=length)
    frame = frame.reshape(shape)

    return frame
