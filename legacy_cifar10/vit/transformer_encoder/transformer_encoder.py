import math

import torch
from torch import nn

from vit.config import ViTConfig


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention from Equations 5-8 of the ViT paper."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output_projection = nn.Linear(
            config.embedding_dim, config.embedding_dim
        )
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Expected tokens with shape [B, S, embedding_dim] where "
                f"embedding_dim={self.embedding_dim}, "
                f"got {tuple(tokens.shape)}."
            )

        batch_size, sequence_length, _ = tokens.shape

        # [B, S, D] -> [B, S, 3, heads, head_dim]
        qkv = self.qkv(tokens).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        # Split into three [B, heads, S, head_dim] tensors.
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Scaled dot-product attention: softmax(QK^T / sqrt(head_dim))V.
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended_values = attention_weights @ value

        # Join heads: [B, heads, S, head_dim] -> [B, S, D].
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, sequence_length, self.embedding_dim)
        )

        output = self.output_projection(attended_values)
        output = self.output_dropout(output)
        assert output.shape == tokens.shape
        return output


class MLPBlock(nn.Module):
    """The two-layer GELU MLP used inside each Encoder block."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1 = nn.Linear(config.embedding_dim, config.mlp_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.mlp_dim, config.embedding_dim)
        self.dropout2 = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = self.fc1(tokens)
        output = self.activation(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.dropout2(output)
        assert output.shape == tokens.shape
        return output


class EncoderBlock(nn.Module):
    """One pre-LayerNorm attention/MLP block from Equations 2 and 3."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.embedding_dim, eps=config.layer_norm_eps
        )
        self.attention = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(
            config.embedding_dim, eps=config.layer_norm_eps
        )
        self.mlp = MLPBlock(config)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Equation 2: z' = MSA(LN(z)) + z
        tokens = tokens + self.attention(self.norm1(tokens))

        # Equation 3: z = MLP(LN(z')) + z'
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class TransformerEncoder(nn.Module):
    """Stack ViT Encoder blocks and apply the paper's final LayerNorm."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.input_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(
            config.embedding_dim, eps=config.layer_norm_eps
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Expected tokens with shape [B, S, embedding_dim] where "
                f"embedding_dim={self.embedding_dim}, "
                f"got {tuple(tokens.shape)}."
            )

        tokens = self.input_dropout(tokens)
        for layer in self.layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)
        return tokens
