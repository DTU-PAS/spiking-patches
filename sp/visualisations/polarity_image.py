import numpy as np

from sp.events import Events


def polarity_image(events: Events, height: int, width: int) -> np.ndarray:
    x = events.x.astype(np.int32, casting="safe")
    y = events.y.astype(np.int32, casting="safe")
    p = events.p.astype(np.int32, casting="safe")

    image = (p * height * width) + (y * width) + x
    image = np.bincount(image, minlength=2 * height * width)
    image = image.reshape(2, height, width)

    coloured_image = np.full((height, width, 3), 156, dtype=np.uint8)
    coloured_image[image[0] > image[1]] = (0, 0, 0)
    coloured_image[image[0] < image[1]] = (44, 96, 251)

    return coloured_image
