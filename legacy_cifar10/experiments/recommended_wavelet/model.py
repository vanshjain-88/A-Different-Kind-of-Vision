import torch
from torch import nn

from experiments.recommended_wavelet.tokenizer import RecommendedWaveletTokenizer
from vit.classification_head import ClassificationHead
from vit.config import ViTConfig
from vit.transformer_encoder import TransformerEncoder


class RecommendedWaveletViT(nn.Module):
    """ViT using sequence-axis SPT pixel and normalized wavelet tokens."""

    def __init__(self, config: ViTConfig, wavelet: str = "haar") -> None:
        super().__init__()
        self.tokenizer = RecommendedWaveletTokenizer(config, wavelet)
        self.encoder = TransformerEncoder(config)
        self.head = ClassificationHead(config)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.tokenizer(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))
