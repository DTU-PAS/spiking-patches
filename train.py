from datetime import timedelta
from pathlib import Path

import lightning.pytorch as pl
import torch
import tyro
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from sp.configs import Checkpoint
from sp.configs import Config
from sp.configs import ContinuousTokenizerConfig
from sp.configs import Dataset
from sp.configs import DiscreteTokenizerConfig
from sp.configs import GNNConfig
from sp.configs import Initialization
from sp.configs import Model
from sp.configs import OptimizerConfig
from sp.configs import PCNConfig
from sp.configs import Size
from sp.configs import TokenizerType
from sp.configs import TrainConfig
from sp.configs import VoxelTokenizerConfig
from sp.constants import WANDB_PROJECT
from sp.loaders import load_datamodule
from sp.loaders import load_model
from sp.paths import get_experiment_dir


def train(config: Config):
    torch.set_float32_matmul_precision("high")

    if config.model == Model.transformer and config.tokenizer == TokenizerType.none:
        raise ValueError("Transformers require tokenization, but '--tokenizer none' was selected.")

    if (
        config.model == Model.transformer
        and config.transformer.init == Initialization.mae
        and config.transformer.size != Size.base
    ):
        raise ValueError("MAE initialisation is only supported for the base model.")

    if config.debug:
        print("Running in debug mode. Model weights will not be saved.")

    datamodule = load_datamodule(config)

    lightning_model = load_model(config)

    run_id, checkpoint_path = load_checkpoint(config.name)

    if config.debug:
        callbacks = []
        logger = False
    else:
        logger = WandbLogger(
            config=config.to_dict(),
            group=config.group,
            id=run_id,
            log_model=False,
            name=config.name,
            project=WANDB_PROJECT,
            settings=wandb.Settings(_disable_stats=True),
            tags=TAGS[config.dataset],
        )

        name = config.name
        if name is None:
            name = logger.experiment.name

        experiment_dir = get_experiment_dir(name, raise_missing=False)
        setup_experiment_dir(experiment_dir, logger.experiment.id)

        checkpoint_kwargs = {}
        if config.ckpt_n_hour is not None:
            checkpoint_kwargs["train_time_interval"] = timedelta(hours=config.ckpt_n_hour)

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=experiment_dir / "checkpoints",
                enable_version_counter=False,
                filename=Checkpoint.best.value,
                mode=MONITOR_MODE[config.dataset],
                monitor=MONITOR_METRIC[config.dataset],
                save_last=True,
                save_top_k=1,
                save_weights_only=False,
                **checkpoint_kwargs,
            ),
        ]

    if not config.validate:
        config.train.limit_val_batches = 0

    trainer = pl.Trainer(
        accumulate_grad_batches=config.train.acc_gradients,
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        enable_checkpointing=not config.debug,
        enable_progress_bar=config.debug,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.train.grad_clip_value,
        limit_train_batches=config.train.limit_train_batches,
        limit_val_batches=config.train.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.train.log_every_n_step,
        max_steps=config.steps,
        precision=config.train.precision,
    )

    trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=checkpoint_path)


def load_checkpoint(name: str | None) -> tuple[str | None, Path | None]:
    run_id = None
    checkpoint_path = None

    if name is None:
        return None, None

    experiment_dir = get_experiment_dir(name, raise_missing=False)
    if experiment_dir.exists():
        run_id_path = experiment_dir / "run_id.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text()

        potential_checkpoint_path = experiment_dir / "checkpoints" / "last.ckpt"
        if potential_checkpoint_path.exists():
            checkpoint_path = potential_checkpoint_path

    return run_id, checkpoint_path


@pl.utilities.rank_zero_only
def setup_experiment_dir(experiment_dir: Path, logger_run_id: str) -> pl.utilities.rank_zero_only:
    experiment_dir.mkdir(exist_ok=True, parents=True)

    run_id_path = experiment_dir / "run_id.txt"
    if not run_id_path.exists():
        run_id_path.write_text(logger_run_id)


MONITOR_METRIC = {
    Dataset.dvsgesture: "val/accuracy",
    Dataset.etram: "val/mAP@.50",
    Dataset.gen1: "val/mAP@.50",
    Dataset.one_mpx: "val/mAP@.50",
    Dataset.slanimalsdvs: "val/accuracy",
}

MONITOR_MODE = {
    Dataset.dvsgesture: "max",
    Dataset.etram: "max",
    Dataset.gen1: "max",
    Dataset.one_mpx: "max",
    Dataset.slanimalsdvs: "max",
}

TAGS = {
    Dataset.dvsgesture: ["DVSGesture"],
    Dataset.etram: ["eTraM"],
    Dataset.gen1: ["GEN1"],
    Dataset.one_mpx: ["1mpx"],
    Dataset.slanimalsdvs: ["SL-Animals-DVS"],
}


DEFAULT_CONFIGS = {
    "DG-GNN-E": (
        "DvsGesture with a GNN on raw events",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=36,
            max_events=10_000,
            gnn=GNNConfig(fps_ratio=0.1, node_radius=3, pool_radius=30),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            steps=20_000,
            time_scale=10_000,
            train=TrainConfig(acc_gradients=2, precision="32"),
            tokenizer=TokenizerType.none,
        ),
    ),
    "DG-GNN-SP": (
        "DvsGesture with a GNN on Spiking Patches",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=72,
            continuous=ContinuousTokenizerConfig(threshold=25),
            discrete=DiscreteTokenizerConfig(duration_ms=50, threshold=25),
            gnn=GNNConfig(fps_ratio=0.1, node_radius=3, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            steps=20_000,
            time_scale=25_000,
            train=TrainConfig(precision="32"),
        ),
    ),
    "DG-GNN-V": (
        "DvsGesture with a GNN on voxels",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=72,
            gnn=GNNConfig(fps_ratio=0.1, node_radius=3, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            steps=20_000,
            time_scale=25_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(precision="32"),
            voxel=VoxelTokenizerConfig(duration_ms=50, threshold=1),
        ),
    ),
    "DG-PCN-E": (
        "DvsGesture with a PCN on raw events",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=72,
            max_events=10_000,
            model=Model.pcn,
            optimizer=OptimizerConfig(lr=0.001),
            pcn=PCNConfig(fps1=0.5, fps2=0.25, radius1=3, radius2=6),
            steps=20_000,
            time_scale=10_000,
            tokenizer=TokenizerType.none,
        ),
    ),
    "DG-PCN-SP": (
        "DvsGesture with a PCN on Spiking Patches",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=36,
            continuous=ContinuousTokenizerConfig(threshold=20),
            discrete=DiscreteTokenizerConfig(duration_ms=50, threshold=20),
            model=Model.pcn,
            num_workers=10,
            patch_size=4,
            pcn=PCNConfig(fps1=0.5, fps2=0.75, radius1=1, radius2=6),
            optimizer=OptimizerConfig(lr=0.001),
            steps=20_000,
            time_scale=10_000,
            train=TrainConfig(acc_gradients=2),
        ),
    ),
    "DG-PCN-V": (
        "DvsGesture with a PCN on voxels",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=36,
            model=Model.pcn,
            num_workers=10,
            patch_size=4,
            pcn=PCNConfig(fps1=0.5, fps2=0.75, radius1=1, radius2=6),
            optimizer=OptimizerConfig(lr=0.001),
            steps=20_000,
            time_scale=10_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(acc_gradients=2),
            voxel=VoxelTokenizerConfig(duration_ms=50, threshold=1),
        ),
    ),
    "DG-T-F": (
        "DVSGesture with a Transformer on frames",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=72,
            model=Model.transformer,
            steps=20_000,
            tokenizer=TokenizerType.voxel,
            voxel=VoxelTokenizerConfig(duration_ms=1000, threshold=0),
        ),
    ),
    "DG-T-SP": (
        "DVSGesture with a Transformer on Spiking Patches",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=72,
            model=Model.transformer,
            steps=20_000,
        ),
    ),
    "DG-T-V": (
        "DVSGesture with a Transformer on voxels",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=72,
            model=Model.transformer,
            steps=20_000,
            tokenizer=TokenizerType.voxel,
            voxel=VoxelTokenizerConfig(duration_ms=100, threshold=1),
        ),
    ),
    "G1-GNN-E": (
        "GEN1 with a GNN on raw events",
        Config(
            dataset=Dataset.gen1,
            batch_size=16,
            max_events=2_000,
            gnn=GNNConfig(fps_ratio=0.9, grid_size=8, node_radius=3, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            reverse_time=True,
            steps=400_000,
            time_scale=5_000,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100, precision="32"),
            tokenizer=TokenizerType.none,
        ),
    ),
    "G1-GNN-SP": (
        "GEN1 with a GNN on Spiking Patches",
        Config(
            dataset=Dataset.gen1,
            batch_size=16,
            continuous=ContinuousTokenizerConfig(threshold=50),
            discrete=DiscreteTokenizerConfig(duration_ms=50, threshold=50),
            gnn=GNNConfig(fps_ratio=0.9, node_radius=2, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            reverse_time=True,
            steps=400_000,
            time_scale=10_000,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100, precision="32"),
        ),
    ),
    "G1-GNN-V": (
        "GEN1 with a GNN on voxels",
        Config(
            dataset=Dataset.gen1,
            batch_size=16,
            gnn=GNNConfig(fps_ratio=0.9, node_radius=2, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            reverse_time=True,
            steps=400_000,
            time_scale=10_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100, precision="32"),
            voxel=VoxelTokenizerConfig(duration_ms=25, threshold=1),
        ),
    ),
    "G1-PCN-E": (
        "GEN1 with a PCN on raw events",
        Config(
            dataset=Dataset.gen1,
            batch_size=16,
            max_events=10_000,
            model=Model.pcn,
            optimizer=OptimizerConfig(lr=0.001),
            pcn=PCNConfig(fps1=0.9, fps2=0.9, grid_size=8, radius1=3, radius2=6),
            reverse_time=True,
            steps=400_000,
            time_scale=10_000,
            tokenizer=TokenizerType.none,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100),
        ),
    ),
    "G1-PCN-SP": (
        "GEN1 with a PCN on Spiking Patches",
        Config(
            dataset=Dataset.gen1,
            batch_size=16,
            continuous=ContinuousTokenizerConfig(threshold=50),
            discrete=DiscreteTokenizerConfig(duration_ms=50, threshold=50),
            model=Model.pcn,
            num_workers=6,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            pcn=PCNConfig(fps1=0.9, fps2=0.9, radius1=1, radius2=6),
            reverse_time=True,
            steps=400_000,
            time_scale=10_000,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100),
        ),
    ),
    "G1-PCN-V": (
        "GEN1 with a PCN on voxels",
        Config(
            dataset=Dataset.gen1,
            batch_size=16,
            model=Model.pcn,
            num_workers=6,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            pcn=PCNConfig(fps1=0.9, fps2=0.9, radius1=1, radius2=6),
            reverse_time=True,
            steps=400_000,
            time_scale=10_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100),
            voxel=VoxelTokenizerConfig(duration_ms=25, threshold=1),
        ),
    ),
    "G1-T-F": (
        "GEN1 with a Transformer on frames",
        Config(
            dataset=Dataset.gen1,
            reverse_time=True,
            steps=400_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100),
            voxel=VoxelTokenizerConfig(duration_ms=50, threshold=0),
        ),
    ),
    "G1-T-SP": (
        "GEN1 with a Transformer on Spiking Patches",
        Config(
            dataset=Dataset.gen1,
            continuous=ContinuousTokenizerConfig(abs_ms=25, threshold=250),
            reverse_time=True,
            steps=400_000,
            tokenizer=TokenizerType.continuous,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100),
        ),
    ),
    "G1-T-V": (
        "GEN1 with a Transformer on voxels",
        Config(
            dataset=Dataset.gen1,
            reverse_time=True,
            steps=400_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(check_val_every_n_epoch=3, log_every_n_step=100),
            voxel=VoxelTokenizerConfig(duration_ms=50, threshold=1),
        ),
    ),
    "SL-GNN-E": (
        "SL-Animals-DVS with a GNN on raw events",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=36,
            max_events=10_000,
            gnn=GNNConfig(fps_ratio=0.1, node_radius=3, pool_radius=30),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            time_scale=10_000,
            train=TrainConfig(acc_gradients=2, precision="32"),
            tokenizer=TokenizerType.none,
        ),
    ),
    "SL-GNN-SP": (
        "SL-Animals-DVS with a GNN on Spiking Patches",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=72,
            continuous=ContinuousTokenizerConfig(threshold=25),
            discrete=DiscreteTokenizerConfig(duration_ms=50, threshold=25),
            gnn=GNNConfig(fps_ratio=0.1, node_radius=3, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            time_scale=25_000,
            train=TrainConfig(precision="32"),
        ),
    ),
    "SL-GNN-V": (
        "SL-Animals-DVS with a GNN on voxels",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=72,
            gnn=GNNConfig(fps_ratio=0.1, node_radius=3, pool_radius=9),
            model=Model.gnn,
            optimizer=OptimizerConfig(lr=0.001),
            patch_size=8,
            time_scale=25_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(precision="32"),
            voxel=VoxelTokenizerConfig(duration_ms=50, threshold=1),
        ),
    ),
    "SL-PCN-E": (
        "SL-Animals-DVS with a PCN on raw events",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=72,
            max_events=10_000,
            model=Model.pcn,
            optimizer=OptimizerConfig(lr=0.001),
            pcn=PCNConfig(fps1=0.5, fps2=0.25, radius1=3, radius2=6),
            time_scale=10_000,
            tokenizer=TokenizerType.none,
        ),
    ),
    "SL-PCN-SP": (
        "SL-Animals-DVS with a PCN on Spiking Patches",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=36,
            continuous=ContinuousTokenizerConfig(threshold=20),
            discrete=DiscreteTokenizerConfig(duration_ms=50, threshold=20),
            model=Model.pcn,
            num_workers=10,
            patch_size=4,
            pcn=PCNConfig(fps1=0.5, fps2=0.75, radius1=1, radius2=6),
            optimizer=OptimizerConfig(lr=0.001),
            time_scale=10_000,
            train=TrainConfig(acc_gradients=2),
        ),
    ),
    "SL-PCN-V": (
        "SL-Animals-DVS with a PCN on voxels",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=36,
            model=Model.pcn,
            num_workers=10,
            patch_size=4,
            pcn=PCNConfig(fps1=0.5, fps2=0.75, radius1=1, radius2=6),
            optimizer=OptimizerConfig(lr=0.001),
            time_scale=10_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(acc_gradients=2),
            voxel=VoxelTokenizerConfig(duration_ms=50, threshold=1),
        ),
    ),
    "SL-T-F": (
        "SL-Animals-DVS with a Transformer on frames",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=72,
            model=Model.transformer,
            steps=15_000,
            tokenizer=TokenizerType.voxel,
            voxel=VoxelTokenizerConfig(duration_ms=1000, threshold=0),
        ),
    ),
    "SL-T-SP": (
        "SL-Animals-DVS with a Transformer on Spiking Patches",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=36,
            model=Model.transformer,
            steps=5_000,
            train=TrainConfig(acc_gradients=2),
        ),
    ),
    "SL-T-V": (
        "SL-Animals-DVS with a Transformer on voxels",
        Config(
            dataset=Dataset.slanimalsdvs,
            batch_size=36,
            model=Model.transformer,
            steps=5_000,
            tokenizer=TokenizerType.voxel,
            train=TrainConfig(acc_gradients=2),
            voxel=VoxelTokenizerConfig(duration_ms=100, threshold=1),
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(DEFAULT_CONFIGS)
    train(config)
