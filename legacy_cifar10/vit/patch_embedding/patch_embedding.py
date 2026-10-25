import torch
from torch import nn

from vit.config import ViTConfig
from vit.wavelets import BiorthogonalDWT, CoifletDWT


class PatchEmbedding(nn.Module):
    """Convert images into ViT tokens, stopping before the Encoder."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(
            config.patch_dim, config.embedding_dim
        )
        if config.tokenizer in {"biorthogonal", "coiflet"}:
            transforms = {
                "biorthogonal": BiorthogonalDWT,
                "coiflet": CoifletDWT,
            }
            self.wavelet_transform = transforms[config.tokenizer]()
            self.wavelet_projection = nn.Linear(
                config.wavelet_patch_dim, config.embedding_dim
            )
            self.fusion = nn.Linear(
                2 * config.embedding_dim, config.embedding_dim
            )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embedding_dim))
        self.positional_embedding = nn.Parameter(
            torch.empty(1, config.num_patches + 1, config.embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer_name in (
            "projection",
            "wavelet_projection",
            "fusion",
        ):
            layer = getattr(self, layer_name, None)
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Split [B, C, H, W] images into [B, N, C * P * P] patches."""
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, C, H, W], "
                f"but received a {images.ndim}-D tensor."
            )

        batch_size, channels, height, width = images.shape
        expected_size = self.config.image_size
        if channels != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {channels}."
            )
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"Expected square images of size {expected_size}x{expected_size}, "
                f"got {height}x{width}."
            )

        patch_size = self.config.patch_size
        patches_per_side = self.config.patches_per_side

        # [B, C, H, W] -> [B, C, H/P, P, W/P, P]
        patches = images.reshape(
            batch_size,
            channels,
            patches_per_side,
            patch_size,
            patches_per_side,
            patch_size,
        )

        # [B, C, H/P, P, W/P, P] -> [B, H/P, W/P, C, P, P]
        patches = patches.permute(0, 2, 4, 1, 3, 5)

        # [B, H/P, W/P, C, P, P] -> [B, N, patch_dim]
        patches = patches.reshape(
            batch_size, self.config.num_patches, self.config.patch_dim
        )

        assert patches.shape == (
            batch_size,
            self.config.num_patches,
            self.config.patch_dim,
        )
        return patches

    def patchify_wavelets(self, wavelets: torch.Tensor) -> torch.Tensor:
        """Split [B, 4C, H/2, W/2] wavelets into aligned patch tokens."""
        if wavelets.ndim != 4:
            raise ValueError(
                "Expected wavelets with shape [B, 4C, H/2, W/2], "
                f"but received a {wavelets.ndim}-D tensor."
            )

        batch_size, channels, height, width = wavelets.shape
        expected_channels = 4 * self.config.in_channels
        expected_size = self.config.image_size // 2
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} wavelet channels, got {channels}."
            )
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"Expected wavelet maps of size {expected_size}x{expected_size}, "
                f"got {height}x{width}."
            )

        patch_size = self.config.wavelet_patch_size
        patches_per_side = self.config.patches_per_side
        patches = wavelets.reshape(
            batch_size,
            channels,
            patches_per_side,
            patch_size,
            patches_per_side,
            patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(
            batch_size, self.config.num_patches, self.config.wavelet_patch_dim
        )
        return patches

    def project_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Project flattened patches from patch_dim to embedding_dim."""
        if patches.ndim != 3 or patches.shape[-1] != self.config.patch_dim:
            raise ValueError(
                "Expected patches with shape [B, N, patch_dim] where "
                f"patch_dim={self.config.patch_dim}, got {tuple(patches.shape)}."
            )

        patch_embeddings = self.projection(patches)
        assert patch_embeddings.shape == (
            patches.shape[0],
            patches.shape[1],
            self.config.embedding_dim,
        )
        return patch_embeddings

    def project_wavelet_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Project flattened wavelet patches into the shared token dimension."""
        if (
            patches.ndim != 3
            or patches.shape[-1] != self.config.wavelet_patch_dim
        ):
            raise ValueError(
                "Expected wavelet patches with shape [B, N, wavelet_patch_dim] "
                f"where wavelet_patch_dim={self.config.wavelet_patch_dim}, "
                f"got {tuple(patches.shape)}."
            )

        wavelet_embeddings = self.wavelet_projection(patches)
        assert wavelet_embeddings.shape == (
            patches.shape[0],
            patches.shape[1],
            self.config.embedding_dim,
        )
        return wavelet_embeddings

    def fuse_embeddings(
        self,
        patch_embeddings: torch.Tensor,
        wavelet_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse aligned RGB and wavelet embeddings into one token sequence."""
        expected_shape = (
            self.config.num_patches,
            self.config.embedding_dim,
        )
        if (
            patch_embeddings.ndim != 3
            or tuple(patch_embeddings.shape[1:]) != expected_shape
            or wavelet_embeddings.shape != patch_embeddings.shape
        ):
            raise ValueError(
                "Expected aligned RGB and wavelet embeddings with shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], got "
                f"{tuple(patch_embeddings.shape)} and "
                f"{tuple(wavelet_embeddings.shape)}."
            )

        fused_embeddings = self.fusion(
            torch.cat((patch_embeddings, wavelet_embeddings), dim=-1)
        )
        return fused_embeddings

    def prepend_cls(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Prepend the learnable CLS token to [B, N, D] embeddings."""
        expected_shape = (self.config.num_patches, self.config.embedding_dim)
        if (
            patch_embeddings.ndim != 3
            or tuple(patch_embeddings.shape[1:]) != expected_shape
        ):
            raise ValueError(
                "Expected patch embeddings with shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(patch_embeddings.shape)}."
            )

        batch_size = patch_embeddings.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_embeddings), dim=1)
        assert tokens.shape == (
            batch_size,
            self.config.num_patches + 1,
            self.config.embedding_dim,
        )
        return tokens

    def add_position(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add the learnable [1, N + 1, D] positional embedding."""
        expected_shape = (
            self.config.num_patches + 1,
            self.config.embedding_dim,
        )
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected tokens with shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(tokens.shape)}."
            )

        tokens = tokens + self.positional_embedding
        assert tokens.shape == (
            tokens.shape[0],
            self.config.num_patches + 1,
            self.config.embedding_dim,
        )
        return tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.config.tokenizer == "patch":
            embeddings = self.project_patches(self.patchify(images))
        else:
            patch_embeddings = self.project_patches(self.patchify(images))
            wavelets = self.wavelet_transform(images)
            wavelet_patches = self.patchify_wavelets(wavelets)
            wavelet_embeddings = self.project_wavelet_patches(wavelet_patches)
            embeddings = self.fuse_embeddings(
                patch_embeddings, wavelet_embeddings
            )

        tokens = self.prepend_cls(embeddings)
        tokens = self.add_position(tokens)
        return tokens
