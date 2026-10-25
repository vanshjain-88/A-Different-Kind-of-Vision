import unittest

import torch
from torch import nn

from vit import ViTConfig, VisionTransformer
from vit.classification_head import ClassificationHead
from vit.transformer_encoder import (
    EncoderBlock,
    MLPBlock,
    MultiHeadSelfAttention,
    TransformerEncoder,
)


def small_config(**overrides: object) -> ViTConfig:
    values = {
        "image_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "embedding_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_dim": 64,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "num_classes": 10,
    }
    values.update(overrides)
    return ViTConfig(**values)


class TransformerEncoderTests(unittest.TestCase):
    def test_attention_preserves_token_shape(self) -> None:
        attention = MultiHeadSelfAttention(small_config())
        tokens = torch.randn(2, 65, 32, requires_grad=True)

        output = attention(tokens)
        output.sum().backward()

        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(attention.qkv.weight.shape, (96, 32))
        self.assertEqual(attention.output_projection.weight.shape, (32, 32))
        self.assertIsNotNone(tokens.grad)

    def test_mlp_uses_paper_dimensions_and_gelu(self) -> None:
        mlp = MLPBlock(small_config())

        self.assertEqual(mlp.fc1.weight.shape, (64, 32))
        self.assertIsInstance(mlp.activation, nn.GELU)
        self.assertEqual(mlp.fc2.weight.shape, (32, 64))
        self.assertEqual(mlp(torch.randn(2, 65, 32)).shape, (2, 65, 32))

    def test_zeroed_block_is_identity_through_residuals(self) -> None:
        block = EncoderBlock(small_config())
        for parameter in block.attention.parameters():
            nn.init.zeros_(parameter)
        for parameter in block.mlp.parameters():
            nn.init.zeros_(parameter)
        tokens = torch.randn(2, 65, 32)

        output = block(tokens)

        self.assertTrue(torch.equal(output, tokens))

    def test_encoder_has_requested_number_of_pre_norm_blocks(self) -> None:
        encoder = TransformerEncoder(small_config(num_layers=3))

        self.assertEqual(len(encoder.layers), 3)
        for layer in encoder.layers:
            self.assertIsInstance(layer.norm1, nn.LayerNorm)
            self.assertIsInstance(layer.attention, MultiHeadSelfAttention)
            self.assertIsInstance(layer.norm2, nn.LayerNorm)
            self.assertIsInstance(layer.mlp, MLPBlock)

    def test_encoder_applies_final_layer_norm(self) -> None:
        encoder = TransformerEncoder(small_config())
        tokens = torch.randn(2, 65, 32)

        output = encoder(tokens)

        token_means = output.mean(dim=-1)
        token_variances = output.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(token_means, torch.zeros_like(token_means), atol=1e-5))
        self.assertTrue(
            torch.allclose(
                token_variances, torch.ones_like(token_variances), atol=1e-4
            )
        )

    def test_classification_head_uses_cls_and_tanh_mlp(self) -> None:
        head = ClassificationHead(small_config())
        encoded_tokens = torch.randn(2, 65, 32)

        logits = head(encoded_tokens)

        self.assertIsInstance(head.activation, nn.Tanh)
        self.assertEqual(head.pre_logits.weight.shape, (32, 32))
        self.assertEqual(head.classifier.weight.shape, (10, 32))
        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))

    def test_complete_model_returns_class_logits(self) -> None:
        model = VisionTransformer(small_config())
        images = torch.randn(2, 3, 32, 32)

        encoded_tokens = model.forward_features(images)
        logits = model(images)

        self.assertEqual(encoded_tokens.shape, (2, 65, 32))
        self.assertEqual(logits.shape, (2, 10))

    def test_embedding_dimension_must_split_across_heads(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "embedding_dim must be divisible by num_heads"
        ):
            small_config(embedding_dim=30, num_heads=8)


if __name__ == "__main__":
    unittest.main()
