import unittest

import torch
from torch import nn

from vit import ViTConfig, ViTInputPipeline
from vit.patch_embedding import PatchEmbedding
from vit.wavelets import BiorthogonalDWT, CoifletDWT


class PatchEmbeddingTests(unittest.TestCase):
    def test_cifar_stage_shapes(self) -> None:
        config = ViTConfig(
            image_size=32,
            patch_size=4,
            in_channels=3,
            embedding_dim=128,
            num_classes=10,
        )
        module = PatchEmbedding(config)
        images = torch.randn(4, 3, 32, 32)

        patches = module.patchify(images)
        self.assertEqual(patches.shape, (4, 64, 48))

        patch_embeddings = module.project_patches(patches)
        self.assertEqual(patch_embeddings.shape, (4, 64, 128))

        tokens_with_cls = module.prepend_cls(patch_embeddings)
        self.assertEqual(tokens_with_cls.shape, (4, 65, 128))

        positioned_tokens = module.add_position(tokens_with_cls)
        self.assertEqual(positioned_tokens.shape, (4, 65, 128))
        self.assertEqual(module(images).shape, (4, 65, 128))

        wavelets = module.wavelet_transform(images)
        self.assertEqual(wavelets.shape, (4, 12, 16, 16))
        wavelet_patches = module.patchify_wavelets(wavelets)
        self.assertEqual(wavelet_patches.shape, (4, 64, 48))
        wavelet_embeddings = module.project_wavelet_patches(wavelet_patches)
        self.assertEqual(wavelet_embeddings.shape, (4, 64, 128))

    def test_multiple_patch_sizes(self) -> None:
        for patch_size, expected_patches in ((2, 256), (8, 16)):
            with self.subTest(patch_size=patch_size):
                config = ViTConfig(
                    image_size=32,
                    patch_size=patch_size,
                    in_channels=3,
                    embedding_dim=128,
                    num_classes=10,
                )
                model = ViTInputPipeline(config)
                images = torch.randn(4, 3, 32, 32)

                tokens = model(images)

                self.assertEqual(
                    tokens.shape, (4, expected_patches + 1, 128)
                )

    def test_patchify_preserves_patch_values_and_order(self) -> None:
        config = ViTConfig(
            image_size=4,
            patch_size=2,
            in_channels=1,
            embedding_dim=8,
            num_classes=2,
        )
        module = PatchEmbedding(config)
        image = torch.arange(16).reshape(1, 1, 4, 4)

        patches = module.patchify(image)

        expected = torch.tensor(
            [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]
        )
        self.assertTrue(torch.equal(patches, expected))

    def test_wavelet_variants_produce_four_half_resolution_subbands(self) -> None:
        images = torch.randn(2, 3, 32, 32, requires_grad=True)

        for transform in (BiorthogonalDWT(), CoifletDWT()):
            with self.subTest(wavelet=transform.name):
                subbands = transform(images)
                self.assertEqual(subbands.shape, (2, 12, 16, 16))
                self.assertTrue(torch.isfinite(subbands).all())

    def test_wavelet_transforms_preserve_autograd(self) -> None:
        for transform in (BiorthogonalDWT(), CoifletDWT()):
            with self.subTest(wavelet=transform.name):
                images = torch.randn(1, 3, 8, 8, requires_grad=True)
                transform(images).square().mean().backward()
                self.assertIsNotNone(images.grad)
                self.assertTrue(torch.isfinite(images.grad).all())

    def test_wavelet_and_rgb_branches_have_aligned_tokens(self) -> None:
        config = ViTConfig(
            image_size=32,
            patch_size=8,
            in_channels=3,
            embedding_dim=16,
            num_classes=10,
        )
        module = PatchEmbedding(config)
        images = torch.randn(2, 3, 32, 32)

        rgb_embeddings = module.project_patches(module.patchify(images))
        wavelet_embeddings = module.project_wavelet_patches(
            module.patchify_wavelets(module.wavelet_transform(images))
        )

        self.assertEqual(rgb_embeddings.shape, (2, 16, 16))
        self.assertEqual(wavelet_embeddings.shape, (2, 16, 16))
        self.assertEqual(module.fuse_embeddings(rgb_embeddings, wavelet_embeddings).shape, (2, 16, 16))

    def test_tokenizer_variants_produce_matching_encoder_shapes(self) -> None:
        images = torch.randn(2, 3, 32, 32)

        for tokenizer in ("patch", "biorthogonal", "coiflet"):
            with self.subTest(tokenizer=tokenizer):
                config = ViTConfig(
                    tokenizer=tokenizer,
                    image_size=32,
                    patch_size=4,
                    embedding_dim=32,
                    num_heads=4,
                )
                module = PatchEmbedding(config)

                self.assertEqual(module(images).shape, (2, 65, 32))

        patch_module = PatchEmbedding(ViTConfig(tokenizer="patch"))
        self.assertFalse(hasattr(patch_module, "wavelet_projection"))

    def test_cls_and_position_parameters_are_learnable(self) -> None:
        module = PatchEmbedding(ViTConfig())

        self.assertIsInstance(module.cls_token, nn.Parameter)
        self.assertTrue(module.cls_token.requires_grad)
        self.assertEqual(module.cls_token.shape, (1, 1, 128))
        self.assertIsInstance(module.positional_embedding, nn.Parameter)
        self.assertTrue(module.positional_embedding.requires_grad)
        self.assertEqual(module.positional_embedding.shape, (1, 65, 128))

    def test_non_divisible_patch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "image_size must be divisible by patch_size"
        ):
            ViTConfig(image_size=32, patch_size=5)

    def test_wrong_input_shape_is_rejected(self) -> None:
        module = PatchEmbedding(ViTConfig())

        with self.assertRaisesRegex(ValueError, "Expected square images"):
            module(torch.randn(4, 3, 28, 28))


if __name__ == "__main__":
    unittest.main()
