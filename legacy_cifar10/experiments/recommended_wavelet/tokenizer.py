import torch
from torch import nn
from torch.nn import functional as F

from experiments.recommended_wavelet.transforms import HaarDWT, rgb_to_ycbcr
from training.train import CIFAR10_MEAN, CIFAR10_STD
from vit.config import ViTConfig
from vit.wavelets import BiorthogonalDWT


class RecommendedWaveletTokenizer(nn.Module):
    """Build separate SPT-pixel and wavelet tokens for each spatial location."""

    def __init__(self, config: ViTConfig, wavelet: str = "haar") -> None:
        super().__init__()
        if config.in_channels != 3:
            raise ValueError("The YCbCr experiment requires three RGB channels.")
        transforms: dict[str, type[nn.Module]] = {
            "haar": HaarDWT,
            "biorthogonal": BiorthogonalDWT,
        }
        if wavelet not in transforms:
            raise ValueError(
                f"wavelet must be one of {sorted(transforms)}, got {wavelet!r}."
            )

        self.config = config
        self.wavelet_name = wavelet
        self.wavelet_transform = transforms[wavelet]()
        self.shift_size = max(1, config.patch_size // 2)
        shifted_patch_dim = 5 * config.patch_dim
        wavelet_patch_size = config.wavelet_patch_size
        # Luminance uses 2x2 wavelet patches. Downsampled Cb/Cr maps already
        # have one coefficient per final spatial token.
        ll_patch_dim = wavelet_patch_size**2 + 2
        detail_patch_dim = 3 * wavelet_patch_size**2 + 6

        self.pixel_projection = nn.Linear(
            shifted_patch_dim, config.embedding_dim
        )
        self.ll_projection = nn.Linear(ll_patch_dim, config.embedding_dim)
        self.detail_projection = nn.Linear(
            detail_patch_dim, config.embedding_dim
        )
        self.ll_norm = nn.LayerNorm(
            config.embedding_dim, eps=config.layer_norm_eps
        )
        self.detail_norm = nn.LayerNorm(
            config.embedding_dim, eps=config.layer_norm_eps
        )
        self.wavelet_norm = nn.LayerNorm(
            config.embedding_dim, eps=config.layer_norm_eps
        )
        self.band_logits = nn.Parameter(torch.zeros(2))
        self.type_embedding = nn.Parameter(
            torch.empty(1, 2, 1, config.embedding_dim)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embedding_dim))
        self.spatial_embedding = nn.Parameter(
            torch.empty(1, config.num_patches + 1, config.embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (
            self.pixel_projection,
            self.ll_projection,
            self.detail_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)

    def shifted_images(self, images: torch.Tensor) -> torch.Tensor:
        """Concatenate the image and four zero-padded diagonal shifts."""
        if images.ndim != 4:
            raise ValueError("Expected images with shape [B, C, H, W].")
        _, _, height, width = images.shape
        shift = self.shift_size
        shifted = (
            F.pad(images, (shift, 0, shift, 0))[:, :, :height, :width],
            F.pad(images, (0, shift, shift, 0))[:, :, :height, shift:],
            F.pad(images, (shift, 0, 0, shift))[:, :, shift:, :width],
            F.pad(images, (0, shift, 0, shift))[:, :, shift:, shift:],
        )
        return torch.cat((images, *shifted), dim=1)

    def patchify(self, maps: torch.Tensor, patch_size: int) -> torch.Tensor:
        batch_size, channels, height, width = maps.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Feature-map size must be divisible by patch size.")
        rows = height // patch_size
        columns = width // patch_size
        patches = maps.reshape(
            batch_size,
            channels,
            rows,
            patch_size,
            columns,
            patch_size,
        )
        return patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, rows * columns, channels * patch_size**2
        )

    def pixel_tokens(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(self.shifted_images(images), self.config.patch_size)
        return self.pixel_projection(patches)

    def wavelet_tokens(self, images: torch.Tensor) -> torch.Tensor:
        ycbcr = rgb_to_ycbcr(images, CIFAR10_MEAN, CIFAR10_STD)
        luminance = ycbcr[:, :1]
        chroma = F.avg_pool2d(ycbcr[:, 1:], kernel_size=2, stride=2)
        luminance_coefficients = self.wavelet_transform(luminance)
        chroma_coefficients = self.wavelet_transform(chroma)

        batch_size = images.shape[0]
        luminance_subbands = luminance_coefficients.reshape(
            batch_size,
            1,
            4,
            luminance_coefficients.shape[2],
            luminance_coefficients.shape[3],
        )
        chroma_subbands = chroma_coefficients.reshape(
            batch_size,
            2,
            4,
            chroma_coefficients.shape[2],
            chroma_coefficients.shape[3],
        )
        patch_size = self.config.wavelet_patch_size
        ll_patches = torch.cat(
            (
                self.patchify(luminance_subbands[:, :, 0], patch_size),
                self.patchify(chroma_subbands[:, :, 0], 1),
            ),
            dim=-1,
        )
        detail_patches = torch.cat(
            (
                self.patchify(
                    luminance_subbands[:, :, 1:].reshape(
                        batch_size,
                        3,
                        luminance_coefficients.shape[2],
                        luminance_coefficients.shape[3],
                    ),
                    patch_size,
                ),
                self.patchify(
                    chroma_subbands[:, :, 1:].reshape(
                        batch_size,
                        6,
                        chroma_coefficients.shape[2],
                        chroma_coefficients.shape[3],
                    ),
                    1,
                ),
            ),
            dim=-1,
        )
        ll_embeddings = self.ll_norm(
            self.ll_projection(ll_patches)
        )
        detail_embeddings = self.detail_norm(
            self.detail_projection(detail_patches)
        )
        band_weights = self.band_logits.softmax(dim=0)
        wavelet_embeddings = (
            band_weights[0] * ll_embeddings
            + band_weights[1] * detail_embeddings
        )
        return self.wavelet_norm(wavelet_embeddings)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_tokens = self.pixel_tokens(images)
        wavelet_tokens = self.wavelet_tokens(images)
        expected_shape = (
            images.shape[0],
            self.config.num_patches,
            self.config.embedding_dim,
        )
        if (
            pixel_tokens.shape != expected_shape
            or wavelet_tokens.shape != expected_shape
        ):
            raise RuntimeError(
                f"Expected aligned token streams with shape {expected_shape}, got "
                f"{tuple(pixel_tokens.shape)} and {tuple(wavelet_tokens.shape)}."
            )

        spatial_positions = self.spatial_embedding[:, 1:]
        pixel_tokens = (
            pixel_tokens + spatial_positions + self.type_embedding[:, 0]
        )
        wavelet_tokens = (
            wavelet_tokens + spatial_positions + self.type_embedding[:, 1]
        )
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        cls_token = cls_token + self.spatial_embedding[:, :1]
        return torch.cat((cls_token, pixel_tokens, wavelet_tokens), dim=1)
