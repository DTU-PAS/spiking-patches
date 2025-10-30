import torch
from torch import nn
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel

from sp.configs import Config
from sp.configs import Initialization
from sp.configs import TransformerConfig
from sp.data_types import TokensBatch
from sp.nn.embeddings import PositionEmbedding
from sp.nn.embeddings import TokenEmbedding

_MAE_BASE_NAME = "facebook/vit-mae-base"


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        config: Config,
        num_classes: int,
        bias: bool = True,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()

        self.bias = bias
        self.initializer_range = initializer_range
        self.num_layers = config.transformer.num_layers

        cls = torch.empty(1, 1, config.transformer.hidden_size)
        nn.init.normal_(cls, mean=0.0, std=initializer_range)
        self.cls = nn.Parameter(cls)

        self.embeddings = TokenEmbedding(config, config.transformer.hidden_size)
        self.positional = PositionEmbedding(config)
        self.encoder = Encoder(config.transformer, config.dropout, bias=bias, layer_norm_eps=layer_norm_eps)
        self.norm = nn.LayerNorm(config.transformer.hidden_size, bias=bias, eps=layer_norm_eps)
        self.head = ClassificationHead(config.transformer.hidden_size, num_classes)

        self.initialise_parameters(config.transformer.init)

    def forward(self, batch: TokensBatch):
        embeddings = self.embeddings(batch.tokens) + self.positional(batch.pos_x, batch.pos_y, batch.pos_t)

        # Add the CLS token
        batch_size = batch.batch_size
        cls_token = self.cls.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=batch.padding_mask.device),
                batch.padding_mask,
            ],
            dim=1,
        )

        encoded_state = self.encoder(embeddings, padding_mask)

        cls_state = self.norm(encoded_state[:, 0])

        logits = self.head(cls_state)

        return logits

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def initialise_masked_autoencoder(self):
        mae = ViTMAEModel.from_pretrained(_MAE_BASE_NAME)
        pretrained_weights = mae.state_dict()

        self.embeddings.init_masked_autoencoder(pretrained_weights)
        self.encoder.init_masked_autoencoder(pretrained_weights)

        state_dict = {
            "cls": pretrained_weights["embeddings.cls_token"],
            "norm.weight": pretrained_weights["layernorm.weight"],
        }
        if self.bias:
            state_dict["norm.bias"] = pretrained_weights["layernorm.bias"]
        self.load_state_dict(state_dict, strict=False)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()

        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


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
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor):
        """
        Args:
            embeddings (torch.Tensor): The embeddings to encode. Shape (batch_size, sequence_length, hidden_size).
            padding_mask (torch.Tensor): The padding mask for the transformer. Shape (batch_size, sequence_length).
        """
        hidden_state = embeddings

        for layer in self.layers:
            hidden_state = layer(hidden_state, padding_mask)

        return hidden_state

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

    def forward(self, hidden_state: torch.Tensor, padding_mask: torch.Tensor):
        hidden_state = self.self_attention(hidden_state, padding_mask)
        hidden_state = self.feed_forward(hidden_state)
        return hidden_state

    def self_attention(self, hidden_state: torch.Tensor, padding_mask: torch.Tensor):
        skip_connection = hidden_state

        hidden_state = self.norm_self_attention(hidden_state)

        hidden_state, _ = self.self_attn(
            query=hidden_state,
            key=hidden_state,
            value=hidden_state,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        hidden_state = self.dropout_self_attention(hidden_state)
        hidden_state = skip_connection + hidden_state

        return hidden_state

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
