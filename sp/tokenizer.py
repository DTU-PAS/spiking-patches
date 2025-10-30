import spiking_patches
from sp.configs import Config
from sp.configs import TokenizerType
from sp.data_types import Tokens
from sp.events import Events
from sp.loaders import load_dimensions


class BatchTokenizer:
    """This is a Python wrapper around the Rust tokenizers."""

    def __init__(self, config: Config):
        self.config = config
        self.height, self.width = load_dimensions(config.dataset)

    def __call__(self, batch: list[Events], prediction_time: list[int] | None = None) -> list[Tokens]:
        """Tokenizes sequences of events into tokens."""

        if prediction_time is None:
            prediction_time = [events.t[-1].item() for events in batch]

        batch = [(events.x, events.y, events.t, events.p) for events in batch]

        tokenizer = self._get_tokenizer()
        batch = tokenizer.tokenize_batch(batch)

        outputs = []
        for (x, y, t, events_x, events_y, events_t, events_p), pred_time in zip(batch, prediction_time, strict=True):
            tokens = Tokens(
                prediction_time=pred_time,
                pos_x=x,
                pos_y=y,
                pos_t=t,
                events_x=events_x,
                events_y=events_y,
                events_t=events_t,
                events_p=events_p,
            )
            outputs.append(tokens)

        return outputs

    def _get_tokenizer(self):
        match self.config.tokenizer:
            case TokenizerType.continuous:
                continuous = self.config.continuous
                return spiking_patches.ContinuousBatchTokenizer(
                    absolute_refractory_period=continuous.abs_ms * 1000,
                    decay=continuous.decay,
                    height=self.height,
                    patch_size=self.config.patch_size,
                    relative_refractory_period=continuous.rel_ms * 1000,
                    relative_refractory_scale=continuous.rel_scale,
                    spike_threshold=continuous.threshold,
                    width=self.width,
                )
            case TokenizerType.discrete:
                discrete = self.config.discrete
                return spiking_patches.DiscreteBatchTokenizer(
                    decay=discrete.duration_ms * 1000,
                    delay=discrete.abs_ms * 1000,
                    height=self.height,
                    patch_size=self.config.patch_size,
                    spike_threshold=discrete.threshold,
                    width=self.width,
                )
            case TokenizerType.voxel:
                voxel = self.config.voxel
                return spiking_patches.VoxelTokenizer(
                    duration_us=voxel.duration_ms * 1000,
                    height=self.height,
                    patch_size=self.config.patch_size,
                    threshold=voxel.threshold,
                    width=self.width,
                )
            case TokenizerType.none:
                raise ValueError("User requested no tokenization, but 'BatchTokenizer(...)' was called anyway.")


class StreamingTokenizer:
    """This is a Python wrapper around the Rust tokenizers."""

    def __init__(self, config: Config):
        self.config = config
        self.height, self.width = load_dimensions(config.dataset)
        self.tokenizer = self._get_tokenizer()

    def __call__(self, events: Events, prediction_time: int | None = None) -> Tokens:
        """Tokenizes a stream of events into tokens.

        The output tokens may include events that were part of the previous call(s).
        Use `reset()` to clear the tokenizer state before processing a new stream of events.
        """

        if prediction_time is None:
            prediction_time = events.t[-1].item()

        x, y, t, events_x, events_y, events_t, events_p = self.tokenizer.stream(
            events.x,
            events.y,
            events.t,
            events.p,
        )

        return Tokens(
            prediction_time=prediction_time,
            pos_x=x,
            pos_y=y,
            pos_t=t,
            events_x=events_x,
            events_y=events_y,
            events_t=events_t,
            events_p=events_p,
        )

    def reset(self) -> None:
        """Resets the tokenizer state."""
        self.tokenizer.reset()

    def _get_tokenizer(self):
        match self.config.tokenizer:
            case TokenizerType.continuous:
                continuous = self.config.continuous
                return spiking_patches.ContinuousStreamingTokenizer(
                    absolute_refractory_period=continuous.abs_ms * 1000,
                    decay=continuous.decay,
                    height=self.height,
                    patch_size=self.config.patch_size,
                    relative_refractory_period=continuous.rel_ms * 1000,
                    relative_refractory_scale=continuous.rel_scale,
                    spike_threshold=continuous.threshold,
                    width=self.width,
                )
            case TokenizerType.discrete:
                discrete = self.config.discrete
                return spiking_patches.DiscreteStreamingTokenizer(
                    decay=discrete.duration_ms * 1000,
                    delay=discrete.abs_ms * 1000,
                    height=self.height,
                    patch_size=self.config.patch_size,
                    spike_threshold=discrete.threshold,
                    width=self.width,
                )
            case TokenizerType.voxel:
                voxel = self.config.voxel
                return spiking_patches.VoxelTokenizer(
                    duration_us=voxel.duration_ms * 1000,
                    height=self.height,
                    patch_size=self.config.patch_size,
                    threshold=voxel.threshold,
                    width=self.width,
                )
            case TokenizerType.none:
                raise ValueError("User requested no tokenization, but 'StreamingTokenizer(...)' was called anyway.")
