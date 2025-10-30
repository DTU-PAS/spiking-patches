DATASET_NAME = "1mpx"

WIDTH = 1280
HEIGHT = 720

MIN_BOX_DIAG = 60
MIN_BOX_SIDE = 20

SKIP_TIME_US = 500_000

CLASS_NAMES = [
    "pedestrian",
    "two wheeler",
    "car",
    # We follow the Prophesee training/evaluation protocol and omit the following classes:
    # "truck",
    # "bus",
    # "traffic sign",
    # "traffic light",
]

LABEL_TO_CLASS = {index: label for index, label in enumerate(CLASS_NAMES)}

NUM_CLASSES = len(CLASS_NAMES)
