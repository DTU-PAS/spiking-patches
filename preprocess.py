import tyro

from sp.configs import Dataset
from sp.preprocessing.dvs_gesture import DVSGesturePreprocessor
from sp.preprocessing.object_detection import ObjectDetectionPreprocessor
from sp.preprocessing.sl_animals_dvs import SLAnimalsDVSPreprocessor


def main(
    dataset: Dataset,
    /,
    limit: int | None = None,
    chunk_duration_ms: int = 1000,
    max_workers: int | None = None,
    train: bool = True,
    val: bool = True,
    test: bool = True,
):
    if dataset == Dataset.dvsgesture:
        preprocessor = DVSGesturePreprocessor(
            limit=limit, max_duration_ms=chunk_duration_ms, max_workers=max_workers, test=test, train=train
        )
    elif dataset == Dataset.slanimalsdvs:
        preprocessor = SLAnimalsDVSPreprocessor(limit=limit, max_duration_ms=chunk_duration_ms, max_workers=max_workers)
    elif dataset in (Dataset.etram, Dataset.gen1, Dataset.one_mpx):
        preprocessor = ObjectDetectionPreprocessor(
            chunk_duration_ms=chunk_duration_ms,
            dataset=dataset,
            limit=limit,
            test=test,
            train=train,
            val=val,
        )
    else:
        raise ValueError(f"Missing preprocessor for dataset: {dataset.value}")

    preprocessor.preprocess()


tyro.cli(main)
