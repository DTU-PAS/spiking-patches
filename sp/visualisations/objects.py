from pathlib import Path
from typing import BinaryIO

import matplotlib.pyplot as plt

from sp.events import Events
from sp.visualisations.polarity_image import polarity_image


def plot_objects(
    boxes: list[tuple[int, int, int, int, str]],
    events: Events,
    height: int,
    width: int,
    title: str | None = None,
    ax=None,
) -> None:
    image = polarity_image(events, height, width)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.imshow(image)
    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    for x, y, w, h, colour in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1, edgecolor=colour, facecolor="none"))


def compare_predictions_labels(
    output: str | Path | BinaryIO,
    label_boxes: list[tuple[int, int, int, int, str]],
    prediction_boxes: list[tuple[int, int, int, int, str]],
    events: Events,
    height: int,
    width: int,
    dpi: int = 75,
    figsize: tuple[int, int] = (10, 5),
):
    _, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    plot_objects(
        ax=axes[0],
        boxes=prediction_boxes,
        events=events,
        height=height,
        title="Predictions",
        width=width,
    )

    plot_objects(
        ax=axes[1],
        boxes=label_boxes,
        events=events,
        height=height,
        title="Labels",
        width=width,
    )

    plt.tight_layout()
    plt.savefig(output)
    plt.show()
