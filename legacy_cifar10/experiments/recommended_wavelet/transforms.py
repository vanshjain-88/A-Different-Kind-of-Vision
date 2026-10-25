import torch
from torch import nn


class HaarDWT(nn.Module):
    """Apply a normalized one-level 2D Haar DWT per channel."""

    name = "haar"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, C, H, W], "
                f"but received a {images.ndim}-D tensor."
            )

        batch_size, channels, height, width = images.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "Haar DWT requires even image height and width, "
                f"got {height}x{width}."
            )

        pixels = images.reshape(
            batch_size,
            channels,
            height // 2,
            2,
            width // 2,
            2,
        )
        top_left = pixels[:, :, :, 0, :, 0]
        top_right = pixels[:, :, :, 0, :, 1]
        bottom_left = pixels[:, :, :, 1, :, 0]
        bottom_right = pixels[:, :, :, 1, :, 1]
        subbands = torch.stack(
            (
                (top_left + top_right + bottom_left + bottom_right) / 2,
                (top_left - top_right + bottom_left - bottom_right) / 2,
                (top_left + top_right - bottom_left - bottom_right) / 2,
                (top_left - top_right - bottom_left + bottom_right) / 2,
            ),
            dim=2,
        )
        return subbands.reshape(
            batch_size, 4 * channels, height // 2, width // 2
        )


def rgb_to_ycbcr(
    normalized_images: torch.Tensor,
    input_mean: tuple[float, float, float],
    input_std: tuple[float, float, float],
) -> torch.Tensor:
    """Reverse CIFAR normalization and convert RGB images to YCbCr."""
    mean = normalized_images.new_tensor(input_mean).reshape(1, 3, 1, 1)
    std = normalized_images.new_tensor(input_std).reshape(1, 3, 1, 1)
    images = normalized_images * std + mean
    red, green, blue = images.split(1, dim=1)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    chroma_blue = -0.168736 * red - 0.331264 * green + 0.5 * blue
    chroma_red = 0.5 * red - 0.418688 * green - 0.081312 * blue
    return torch.cat((luminance, chroma_blue, chroma_red), dim=1)
