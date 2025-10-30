import dataclasses

import lightning.pytorch as pl
import torch
import tyro
import wandb
from lightning.pytorch.loggers import WandbLogger

from sp.configs import Checkpoint
from sp.configs import Config
from sp.configs import Split
from sp.constants import WANDB_PROJECT
from sp.loaders import load_datamodule
from sp.loaders import load_model
from sp.paths import get_experiment_dir


def main(
    name: str,
    /,
    batch_size: int = 20,
    checkpoint: Checkpoint = Checkpoint.best,
    num_workers: int = 4,
    split: Split = Split.val,
) -> None:
    torch.set_float32_matmul_precision("high")

    config, run_id, weights_path = load_checkpoint(name, checkpoint)
    config = dataclasses.replace(config, batch_size=batch_size, num_workers=num_workers)
    datamodule = load_datamodule(config, test_split=split)
    lightning_model = load_model(config, test_split=split)

    logger = WandbLogger(
        id=run_id,
        prefix="eval",
        project=WANDB_PROJECT,
        settings=wandb.Settings(_disable_stats=True),
    )

    trainer = pl.Trainer(
        callbacks=[],
        logger=logger,
        precision=config.train.precision,
    )

    trainer.test(lightning_model, datamodule=datamodule, ckpt_path=weights_path)


def load_checkpoint(name: str, checkpoint: Checkpoint) -> tuple[Config, str, str]:
    experiment_dir = get_experiment_dir(name)

    if not experiment_dir.exists():
        raise ValueError(f"Could not find experiment directory {experiment_dir}.")

    run_id = (experiment_dir / "run_id.txt").read_text()

    # get model parameters from wandb run with the given name
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")

    config = Config.from_dict(run.config)

    checkpoints_dir = experiment_dir / "checkpoints"
    best_path = checkpoints_dir / f"{Checkpoint.best.value}.ckpt"
    last_path = checkpoints_dir / f"{Checkpoint.last.value}.ckpt"
    if not best_path.exists() and not last_path.exists():
        raise ValueError(f"Could not find any checkpoints in {checkpoints_dir}.")
    if checkpoint == Checkpoint.best:
        weights_path = best_path if best_path.exists() else last_path
    else:
        weights_path = last_path if last_path.exists() else best_path

    return config, run_id, weights_path


tyro.cli(main)
