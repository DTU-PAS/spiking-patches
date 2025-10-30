import torch
from torch import nn

from sp.configs import Config


class PositionEmbedding(nn.Module):
    def __init__(self, config: Config, theta: float = 10000.0):
        super().__init__()

        hidden_size = config.transformer.hidden_size
        divisor = hidden_size // 3
        remainder = hidden_size % 3

        self.x_size = divisor
        self.y_size = divisor
        self.t_size = divisor + remainder

        self.time_scale = config.time_scale
        self.theta = theta

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        x = self.encode(x, self.x_size)
        y = self.encode(y, self.y_size)

        t = t / self.time_scale
        t = self.encode(t, self.t_size)

        encoding = torch.cat([x, y, t], dim=-1)

        return encoding

    def encode(self, positions: torch.Tensor, size: int):
        positions = positions.unsqueeze(-1)  # (batch_size, sequence_length, 1)

        dim_series = torch.arange(size, dtype=torch.int64, device=positions.device).float()
        dim_series = torch.div(dim_series, 2, rounding_mode="floor")

        frequencies = self.theta ** (2 * dim_series / size)
        angles = positions / frequencies

        sine = torch.sin(angles[:, :, 0::2])
        cosine = torch.cos(angles[:, :, 1::2])

        encoding = torch.cat([sine, cosine], dim=-1)

        return encoding


class TokenEmbedding(nn.Module):
    def __init__(self, config: Config, output_size: int):
        super().__init__()

        self.patch_size = config.patch_size
        self.channels = 2 * config.buckets  # multiply by 2 for positive and negative events
        self.output_size = output_size

        self.projection = nn.Conv2d(
            self.channels,
            output_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, tokens: torch.Tensor):
        patches = tokens.to(torch.float32)

        # Join batch/sequence dimensions and polarity/bucket dimensions
        ndim = patches.ndim
        if ndim == 5:
            num_nodes = patches.size(0)
            shape = (num_nodes, self.channels, self.patch_size, self.patch_size)
            patches = patches.view(*shape)
        elif ndim == 6:
            batch_size = patches.size(0)
            sequence_length = patches.size(1)
            shape = (-1, self.channels, self.patch_size, self.patch_size)
            patches = patches.view(*shape)
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")

        # Logarithmically scale the input to account for poorly distributed values
        patches = torch.log(patches + 1)

        embeddings = self.projection(patches)

        if ndim == 5:
            embeddings = embeddings.flatten(1)
        elif ndim == 6:
            embeddings = embeddings.view(batch_size, sequence_length, self.output_size)

        return embeddings

    def init_masked_autoencoder(self, pretrained_weights: dict[str, torch.Tensor]):
        if self.patch_size != 16:
            print(
                "Masked autoencoder initialisation is only supported for patch size 16. "
                + "Using random initialisation for patch embeddings."
            )
            return

        state_dict = {}

        mae_embeddings_prefix = "embeddings.patch_embeddings"

        mae_bias_key = f"{mae_embeddings_prefix}.projection.bias"
        state_dict["projection.bias"] = pretrained_weights[mae_bias_key]

        mae_weight_key = f"{mae_embeddings_prefix}.projection.weight"
        weight = self.projection.weight.data
        mae_weight = pretrained_weights[mae_weight_key].mean(axis=1)
        for channel_index in range(self.channels):
            weight[:, channel_index] = mae_weight
        state_dict["projection.weight"] = weight

        self.load_state_dict(state_dict, strict=True)


class PolarityEmbedding(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()

        self.embedding = nn.Embedding(2, output_size)

    def forward(self, polarities: torch.Tensor):
        polarities = polarities.int()
        polarities = self.embedding(polarities)
        polarities = polarities.squeeze(1)
        return polarities
