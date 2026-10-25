import torch
from torch import nn

from vit.classification_head import ClassificationHead
from vit.config import ViTConfig
from vit.patch_embedding import PatchEmbedding
from vit.transformer_encoder import TransformerEncoder


class ViTInputPipeline(nn.Module):
    """Convert images into tokens at the Encoder input boundary."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(images)


class VisionTransformer(nn.Module):
    """The paper's ViT from image patches through classification logits."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.head = ClassificationHead(config)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return all encoded tokens before the classification head."""
        tokens = self.patch_embedding(images)
        encoded_tokens = self.encoder(tokens)
        return encoded_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded_tokens = self.forward_features(images)
        logits = self.head(encoded_tokens)
        return logits
