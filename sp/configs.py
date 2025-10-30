import math
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Literal

from dataclasses_json import dataclass_json


class Dataset(StrEnum):
    dvsgesture = "dvsgesture"
    etram = "etram"
    gen1 = "gen1"
    one_mpx = "1mpx"
    slanimalsdvs = "slanimalsdvs"


class Checkpoint(StrEnum):
    best = "best"
    last = "last"


class Initialization(StrEnum):
    random = "random"
    mae = "mae"


class Model(StrEnum):
    gnn = "gnn"
    pcn = "pcn"
    transformer = "transformer"


class Size(StrEnum):
    nano = "nano"
    tiny = "tiny"
    small = "small"
    base = "base"


class Split(StrEnum):
    train = "train"
    val = "val"
    test = "test"


class TokenizerType(StrEnum):
    continuous = "continuous"
    discrete = "discrete"
    none = "none"
    voxel = "voxel"


@dataclass_json
@dataclass
class ContinuousTokenizerConfig:
    abs_ms: int = 0
    decay: float = 0.0
    rel_ms: int = 0
    rel_scale: float = 0.4
    threshold: float = 256


@dataclass_json
@dataclass
class DiscreteTokenizerConfig:
    abs_ms: int = 0
    duration_ms: int = 100
    threshold: int = 256


@dataclass_json
@dataclass
class VoxelTokenizerConfig:
    duration_ms: int = 100
    threshold: int = 32


@dataclass_json
@dataclass
class GNNConfig:
    bias: bool = False
    dim1: int = 16
    dim2: int = 32
    dim3: int = 128
    fps_ratio: float = 0.1
    grid_size: int = 1  # Size of the grid for final pooling. Used for object detection.
    kernel_size: int = 4
    loop: bool = True
    max_num_neighbors: int | None = 32
    node_radius: float = 3.0
    pool_radius: float = 9.0
    root_weight: bool = False


@dataclass_json
@dataclass
class ObjectDetectionConfig:
    batch_size_random: int = None  # will be set to floor(batch_size / 2)
    batch_size_streaming: int = None  # will be set to ceil(batch_size / 2)
    num_workers_random: int = None  # will be set to floor(num_workers / 2)
    num_workers_streaming: int = None  # will be set to ceil(num_workers / 2)


@dataclass_json
@dataclass
class ObjectDetectionEvaluatorConfig:
    class_names: list[str]
    dataset: Dataset
    height: int
    min_box_diag: int
    min_box_side: int
    num_classes: int
    skip_time_us: int
    time_tol: int
    width: int


@dataclass_json
@dataclass
class NMSConfig:
    conf: float = 0.1
    iou: float = 0.45


@dataclass_json
@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    div_factor: float = 25.0
    final_div_factor: float = 4.0
    pct_start: float = 0.01

    # Number of steps to train the model. Will be set to the same value as `steps` from the parent config.
    steps: int = 30_000


@dataclass_json
@dataclass
class PCNConfig:
    dim1: int = 16
    dim2: int = 32
    dim3: int = 128
    fps1: float = 0.5
    fps2: float = 0.25
    grid_size: int = 1  # Size of the grid for final pooling. Used for object detection.
    max_num_neighbors: int = 64
    radius1: float = 3.0
    radius2: float = 6.0


@dataclass_json
@dataclass
class TrainConfig:
    acc_gradients: int = 1
    check_val_every_n_epoch: int = 1
    grad_clip_value: float = 1.0
    log_every_n_step: int = 10
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    steps: int = 30_000
    precision: Literal["32"] | Literal["16-mixed"] = "16-mixed"


TRANSFORMER_SIZES = {
    Size.nano: {
        "intermediate_size": 256,
        "hidden_size": 128,
        "num_heads": 4,
        "num_layers": 4,
        "size": Size.nano,
    },
    Size.tiny: {
        "intermediate_size": 512,
        "hidden_size": 256,
        "num_heads": 4,
        "num_layers": 7,
        "size": Size.tiny,
    },
    Size.small: {
        "intermediate_size": 2048,
        "hidden_size": 512,
        "num_heads": 8,
        "num_layers": 10,
        "size": Size.small,
    },
    Size.base: {
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
        "size": Size.base,
    },
}


@dataclass_json
@dataclass
class TransformerConfig:
    grid_size: int = 1  # Size of the grid for final pooling. Used for object detection.
    hidden_size: int = (None,)  # Will be set based on size.
    init: Initialization = Initialization.mae
    intermediate_size: int = (None,)  # Will be set based on size.
    num_heads: int = (None,)  # Will be set based on size.
    num_layers: int = (None,)  # Will be set based on size.
    size: Size = Size.base

    def __post_init__(self):
        if self.size in TRANSFORMER_SIZES:
            config = TRANSFORMER_SIZES[self.size]
            self.intermediate_size = config["intermediate_size"]
            self.hidden_size = config["hidden_size"]
            self.num_heads = config["num_heads"]
            self.num_layers = config["num_layers"]
        else:
            raise ValueError(f"Unknown size {self.size}")


@dataclass_json
@dataclass
class Config:
    dataset: Dataset
    batch_size: int = 16
    buckets: int = 10
    continuous: ContinuousTokenizerConfig = field(default_factory=ContinuousTokenizerConfig)
    discrete: DiscreteTokenizerConfig = field(default_factory=DiscreteTokenizerConfig)
    ckpt_n_hour: int | None = None
    debug: bool = False
    dropout: float = 0.1
    gnn: GNNConfig = field(default_factory=GNNConfig)
    group: str | None = None
    max_events: int | None = None  # Subsample events for GNN and PCN models on raw events.
    model: Model = Model.transformer
    name: str | None = None
    nms: NMSConfig = field(default_factory=NMSConfig)
    num_workers: int = 4
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    patch_size: int = 16
    pcn: PCNConfig = field(default_factory=PCNConfig)
    predict_every_ms: int = 50  # How often to predict in milliseconds. Used in object detection.
    reverse_time: bool = False
    sequence_length: int = 10  # Defaults to 10 - corresponds to 500 ms when --predict-every-ms is 50.
    steps: int = 30_000
    time_scale: float = 50_000.0
    tokenizer: TokenizerType = TokenizerType.continuous
    train: TrainConfig = field(default_factory=TrainConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    validate: bool = True
    voxel: VoxelTokenizerConfig = field(default_factory=VoxelTokenizerConfig)

    def __post_init__(self):
        self.object_detection.batch_size_random = math.floor(self.batch_size / 2)
        self.object_detection.batch_size_streaming = math.ceil(self.batch_size / 2)
        self.object_detection.num_workers_random = math.floor(self.num_workers / 2)
        self.object_detection.num_workers_streaming = math.ceil(self.num_workers / 2)
        self.optimizer.steps = self.steps
