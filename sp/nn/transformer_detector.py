import math

import torch
from torch import nn
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTraining

from sp.configs import Config
from sp.configs import Initialization
from sp.configs import TransformerConfig
from sp.data_types import TokensBatch
from sp.loaders import load_dimensions
from sp.nn.embeddings import PositionEmbedding
from sp.nn.embeddings import TokenEmbedding

_MAE_BASE_NAME = "facebook/vit-mae-base"


class TransformerDetector(nn.Module):
    def __init__(
        self,
        config: Config,
        bias: bool = True,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
    ):
        super().__init__()

        self.bias = bias
        self.initializer_range = initializer_range
        self.patch_size = config.patch_size

        self.height, self.width = load_dimensions(config.dataset)

        cls = torch.empty(1, 1, config.transformer.hidden_size)
        nn.init.normal_(cls, mean=0.0, std=initializer_range)
        self.cls = nn.Parameter(cls)

        self.patch_embeddings = TokenEmbedding(config, config.transformer.hidden_size)

        self.positional = PositionEmbedding(config)

        self.encoder = Encoder(config.transformer, config.dropout, bias=bias, layer_norm_eps=layer_norm_eps)
        self.encoder_norm = nn.LayerNorm(config.transformer.hidden_size, bias=bias, eps=layer_norm_eps)

        self.hidden_size = config.transformer.hidden_size
        self.grid_size = config.transformer.grid_size
        num_input_rows = math.ceil(self.height / config.patch_size)
        num_input_cols = math.ceil(self.width / config.patch_size)
        self.num_rows = math.ceil(num_input_rows / self.grid_size)
        self.num_cols = math.ceil(num_input_cols / self.grid_size)
        self.num_cells = self.num_rows * self.num_cols

        self.initialise_parameters(config.transformer.init)

    def forward(self, batch: TokensBatch, batch_size: int) -> torch.Tensor:
        patch_embeddings = self.patch_embeddings(batch.tokens)
        patch_embeddings = patch_embeddings + self.positional(batch.pos_x, batch.pos_y, batch.pos_t)

        # Add the CLS token
        cls_token = self.cls.expand(batch_size, -1, -1)
        patch_embeddings = torch.cat([cls_token, patch_embeddings], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=batch.padding_mask.device),
                batch.padding_mask,
            ],
            dim=1,
        )

        encoder_outputs = self.encoder(patch_embeddings, padding_mask)
        encoder_outputs = self.encoder_norm(encoder_outputs)

        # Max-pool tokens into a 2D grid
        mask = batch.padding_mask.logical_not()
        flattened_mask = mask.view(-1)
        flattened_tokens = encoder_outputs[:, 1:].reshape(-1, self.hidden_size)[flattened_mask]
        pos_x = (batch.pos_x / self.grid_size).view(-1).to(torch.long)[flattened_mask]
        pos_y = (batch.pos_y / self.grid_size).view(-1).to(torch.long)[flattened_mask]
        lengths = mask.sum(dim=1)
        batch_ids = torch.cat(
            [torch.full((length,), i, dtype=torch.long, device=pos_x.device) for i, length in enumerate(lengths)],
            dim=0,
        )
        outputs = torch.zeros(
            batch_size * self.num_cells, self.hidden_size, device=flattened_tokens.device, dtype=flattened_tokens.dtype
        )
        index = batch_ids * self.num_cells + pos_y * self.num_cols + pos_x
        outputs.index_reduce_(0, index, flattened_tokens, reduce="amax")
        outputs = outputs.reshape(batch_size, self.num_rows, self.num_cols, self.hidden_size)
        return outputs

    def initialise_parameters(self, initialisation: Initialization):
        if initialisation == Initialization.random:
            self.initialise_random()
        elif initialisation == Initialization.mae:
            self.initialise_masked_autoencoder()
        else:
            raise ValueError(f"Unknown initialisation: {initialisation}")

    def initialise_random(self):
        self.apply(self.init_module_weights)

    def init_module_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def initialise_masked_autoencoder(self):
        mae = ViTMAEForPreTraining.from_pretrained(_MAE_BASE_NAME)
        pretrained_weights = mae.state_dict()

        self.patch_embeddings.init_masked_autoencoder(
            {
                key.removeprefix("vit."): value
                for key, value in pretrained_weights.items()
                if key.startswith("vit.embeddings.")
            }
        )

        self.encoder.init_masked_autoencoder(
            {
                key.removeprefix("vit."): value
                for key, value in pretrained_weights.items()
                if key.startswith("vit.encoder.")
            }
        )

        state_dict = {
            "cls": pretrained_weights["vit.embeddings.cls_token"],
            "encoder_norm.weight": pretrained_weights["vit.layernorm.weight"],
        }
        if self.bias:
            state_dict["encoder_norm.bias"] = pretrained_weights["vit.layernorm.bias"]

        self.load_state_dict(state_dict, strict=False)


class Encoder(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        dropout: float,
        layer_norm_eps: float = 1e-12,
        bias: bool = True,
    ):
        super().__init__()

        self.bias = bias
        self.num_layers = config.num_layers

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    config=config,
                    dropout=dropout,
                    bias=bias,
                    layer_norm_eps=layer_norm_eps,
                )
                for i in range(config.num_layers)
            ]
        )

    def forward(self, patches: torch.Tensor, padding_mask: torch.Tensor):
        for layer in self.layers:
            patches = layer(patches, padding_mask)
        return patches

    def init_masked_autoencoder(self, pretrained_weights: dict[str, torch.Tensor]):
        state_dict = {}

        model_mae_keys = [
            ("linear1", "intermediate.dense"),
            ("linear2", "output.dense"),
            ("norm_self_attention", "layernorm_before"),
            ("norm_feed_forward", "layernorm_after"),
            ("self_attn.out_proj", "attention.output.dense"),
        ]

        mae_attention_keys = [
            "attention.attention.query",
            "attention.attention.key",
            "attention.attention.value",
        ]

        suffixes = ["weight"]
        if self.bias:
            suffixes.append("bias")

        for layer_index in range(self.num_layers):
            mae_prefix = f"encoder.layer.{layer_index}"
            model_prefix = f"layers.{layer_index}"

            for model_key, mae_key in model_mae_keys:
                for suffix in suffixes:
                    source = f"{mae_prefix}.{mae_key}.{suffix}"
                    target = f"{model_prefix}.{model_key}.{suffix}"
                    state_dict[target] = pretrained_weights[source]

            for suffix in suffixes:
                in_proj = [pretrained_weights[f"{mae_prefix}.{key}.{suffix}"] for key in mae_attention_keys]
                in_proj = torch.concat(in_proj, dim=0)
                target = f"{model_prefix}.self_attn.in_proj_{suffix}"
                state_dict[target] = in_proj

        self.load_state_dict(state_dict, strict=True)


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, dropout: float, bias: bool, layer_norm_eps: float):
        super().__init__()

        self.activation = nn.GELU()

        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_feedforward1 = nn.Dropout(dropout)
        self.dropout_feedforward2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

        self.norm_self_attention = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps, bias=bias)
        self.norm_feed_forward = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps, bias=bias)

        self.self_attn = nn.MultiheadAttention(
            bias=bias,
            batch_first=True,
            dropout=dropout,
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
        )

    def forward(self, patches: torch.Tensor, padding_mask: torch.Tensor):
        patches = self.self_attention(patches, padding_mask)
        patches = self.feed_forward(patches)
        return patches

    def self_attention(self, patches: torch.Tensor, padding_mask: torch.Tensor):
        skip_patches = patches

        patches = self.norm_self_attention(patches)

        patches, _ = self.self_attn(
            query=patches,
            key=patches,
            value=patches,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        patches = self.dropout_self_attention(patches)
        patches = skip_patches + patches

        return patches

    def feed_forward(self, hidden_state: torch.Tensor):
        skip_connection = hidden_state

        hidden_state = self.norm_feed_forward(hidden_state)

        hidden_state = self.linear1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout_feedforward1(hidden_state)

        hidden_state = self.linear2(hidden_state)
        hidden_state = self.dropout_feedforward2(hidden_state)

        hidden_state = skip_connection + hidden_state

        return hidden_state
