import lightning.pytorch as pl

from sp.configs import Config
from sp.configs import Dataset
from sp.configs import Split


def load_dimensions(dataset: Dataset) -> tuple[int, int]:
    match dataset:
        case Dataset.dvsgesture:
            from sp.dvs_gesture import HEIGHT
            from sp.dvs_gesture import WIDTH
        case Dataset.etram:
            from sp.etram import HEIGHT
            from sp.etram import WIDTH
        case Dataset.gen1:
            from sp.gen1 import HEIGHT
            from sp.gen1 import WIDTH
        case Dataset.one_mpx:
            from sp.one_mpx import HEIGHT
            from sp.one_mpx import WIDTH
        case Dataset.slanimalsdvs:
            from sp.sl_animals_dvs import HEIGHT
            from sp.sl_animals_dvs import WIDTH
        case _:
            raise ValueError(f"Dataset '{dataset.value}' is not supported yet.")

    return HEIGHT, WIDTH


def load_datamodule(config: Config, test_split: Split = Split.test) -> pl.LightningDataModule:
    match config.dataset:
        case Dataset.dvsgesture:
            from sp.data import DVSGestureDataModule

            return DVSGestureDataModule(config=config, test_split=test_split)
        case Dataset.etram:
            from sp.data import ObjectDetectionDataModule

            return ObjectDetectionDataModule(config=config, test_split=test_split)
        case Dataset.gen1:
            from sp.data import ObjectDetectionDataModule

            return ObjectDetectionDataModule(config=config, test_split=test_split)
        case Dataset.one_mpx:
            from sp.data import ObjectDetectionDataModule

            return ObjectDetectionDataModule(config=config, test_split=test_split)
        case Dataset.slanimalsdvs:
            from sp.data import SLAnimalsDVSDataModule

            return SLAnimalsDVSDataModule(config=config, test_split=test_split)
        case _:
            raise ValueError(f"Dataset '{config.dataset.value}' is not supported yet.")


def load_model(config: Config, test_split: Split = Split.test) -> pl.LightningModule:
    if config.dataset in {Dataset.dvsgesture, Dataset.slanimalsdvs}:
        from sp.models.classification import ClassificationModel

        if config.dataset == Dataset.dvsgesture:
            from sp.dvs_gesture import NUM_CLASSES
        else:
            from sp.sl_animals_dvs import NUM_CLASSES

        return ClassificationModel(
            config=config,
            num_classes=NUM_CLASSES,
            test_split=test_split,
        )
    elif config.dataset in {Dataset.etram, Dataset.gen1, Dataset.one_mpx}:
        from sp.configs import ObjectDetectionEvaluatorConfig
        from sp.models.object_detection import ObjectDetectionModel

        if config.dataset == Dataset.etram:
            from sp.etram import CLASS_NAMES
            from sp.etram import HEIGHT
            from sp.etram import MIN_BOX_DIAG
            from sp.etram import MIN_BOX_SIDE
            from sp.etram import NUM_CLASSES
            from sp.etram import SKIP_TIME_US
            from sp.etram import WIDTH
        elif config.dataset == Dataset.gen1:
            from sp.gen1 import CLASS_NAMES
            from sp.gen1 import HEIGHT
            from sp.gen1 import MIN_BOX_DIAG
            from sp.gen1 import MIN_BOX_SIDE
            from sp.gen1 import NUM_CLASSES
            from sp.gen1 import SKIP_TIME_US
            from sp.gen1 import WIDTH
        else:
            from sp.one_mpx import CLASS_NAMES
            from sp.one_mpx import HEIGHT
            from sp.one_mpx import MIN_BOX_DIAG
            from sp.one_mpx import MIN_BOX_SIDE
            from sp.one_mpx import NUM_CLASSES
            from sp.one_mpx import SKIP_TIME_US
            from sp.one_mpx import WIDTH

        evaluation = ObjectDetectionEvaluatorConfig(
            class_names=CLASS_NAMES,
            dataset=config.dataset,
            height=HEIGHT,
            min_box_diag=MIN_BOX_DIAG,
            min_box_side=MIN_BOX_SIDE,
            num_classes=NUM_CLASSES,
            skip_time_us=SKIP_TIME_US,
            time_tol=(config.predict_every_ms * 1000) // 2,
            width=WIDTH,
        )

        return ObjectDetectionModel(
            config=config,
            evaluation=evaluation,
            test_split=test_split,
        )
    else:
        raise ValueError(f"Dataset '{config.dataset.value}' is not supported yet.")
