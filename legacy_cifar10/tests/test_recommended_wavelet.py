import unittest

import torch

from experiments.recommended_wavelet import (
    RecommendedWaveletTokenizer,
    RecommendedWaveletViT,
)
from vit import ViTConfig


class RecommendedWaveletTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ViTConfig(
            tokenizer="patch",
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_dim=64,
            dropout=0.0,
        )
        self.images = torch.randn(2, 3, 32, 32)

    def test_tokenizer_uses_two_tokens_per_spatial_location(self) -> None:
        tokenizer = RecommendedWaveletTokenizer(self.config, wavelet="haar")

        self.assertEqual(
            tokenizer(self.images).shape,
            (2, 2 * self.config.num_patches + 1, 32),
        )
        self.assertEqual(tokenizer.pixel_tokens(self.images).shape, (2, 64, 32))
        self.assertEqual(
            tokenizer.wavelet_tokens(self.images).shape, (2, 64, 32)
        )

    def test_haar_and_biorthogonal_versions_return_logits(self) -> None:
        for wavelet in ("haar", "biorthogonal"):
            with self.subTest(wavelet=wavelet):
                model = RecommendedWaveletViT(self.config, wavelet=wavelet)
                self.assertEqual(model(self.images).shape, (2, 10))

    def test_band_weights_receive_gradients(self) -> None:
        tokenizer = RecommendedWaveletTokenizer(self.config, wavelet="haar")
        tokenizer(self.images).square().mean().backward()

        self.assertIsNotNone(tokenizer.band_logits.grad)
        self.assertTrue(torch.isfinite(tokenizer.band_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
