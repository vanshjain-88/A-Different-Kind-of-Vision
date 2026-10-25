import torch
from torch import nn

from vit.config import ViTConfig


class ClassificationHead(nn.Module):
    """The paper's one-hidden-layer tanh MLP classification head."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.pre_logits = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

        nn.init.xavier_uniform_(self.pre_logits.weight)
        nn.init.zeros_(self.pre_logits.bias)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        if (
            encoded_tokens.ndim != 3
            or encoded_tokens.shape[-1] != self.embedding_dim
        ):
            raise ValueError(
                "Expected encoded tokens with shape [B, S, embedding_dim] "
                f"where embedding_dim={self.embedding_dim}, "
                f"got {tuple(encoded_tokens.shape)}."
            )

        cls_representation = encoded_tokens[:, 0]
        representation = self.pre_logits(cls_representation)
        representation = self.activation(representation)
        logits = self.classifier(representation)
        return logits
