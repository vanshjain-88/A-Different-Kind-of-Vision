import torch
from torch import nn
from torch.nn import functional as F


class FixedWaveletDWT(nn.Module):
    """Apply a fixed one-level separable 2D DWT to every image channel."""

    def __init__(
        self,
        low_pass: tuple[float, ...],
        high_pass: tuple[float, ...],
        name: str,
    ) -> None:
        super().__init__()
        if len(low_pass) != len(high_pass) or len(low_pass) % 2 != 0:
            raise ValueError("Wavelet filters must have the same even length.")

        low = torch.tensor(low_pass, dtype=torch.float32)
        high = torch.tensor(high_pass, dtype=torch.float32)
        # conv2d performs cross-correlation, so reverse the analysis kernels.
        kernels = torch.stack(
            (
                torch.outer(low, low),
                torch.outer(low, high),
                torch.outer(high, low),
                torch.outer(high, high),
            )
        ).flip(-1, -2)
        self.register_buffer("kernels", kernels.unsqueeze(1))
        self.name = name
        self.filter_length = len(low_pass)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, C, H, W], "
                f"but received a {images.ndim}-D tensor."
            )

        batch_size, channels, height, width = images.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"{self.name} DWT requires even image height and width, "
                f"got {height}x{width}."
            )

        padding = self.filter_length // 2 - 1
        padded = F.pad(
            images,
            (padding, padding, padding, padding),
            mode="circular",
        )
        channel_kernels = self.kernels.to(dtype=images.dtype).repeat(
            channels, 1, 1, 1
        )
        coefficients = F.conv2d(
            padded,
            channel_kernels,
            stride=2,
            groups=channels,
        )
        return coefficients.reshape(
            batch_size, 4 * channels, height // 2, width // 2
        )


class BiorthogonalDWT(FixedWaveletDWT):
    """One-level Cohen-Daubechies-Feauveau bior2.2 transform."""

    def __init__(self) -> None:
        super().__init__(
            low_pass=(
                0.0,
                -0.1767766952966369,
                0.3535533905932738,
                1.0606601717798212,
                0.3535533905932738,
                -0.1767766952966369,
            ),
            high_pass=(
                0.0,
                0.3535533905932738,
                -0.7071067811865476,
                0.3535533905932738,
                0.0,
                0.0,
            ),
            name="bior2.2",
        )


class CoifletDWT(FixedWaveletDWT):
    """One-level coif1 transform."""

    def __init__(self) -> None:
        super().__init__(
            low_pass=(
                -0.01565572813546454,
                -0.0727326195128539,
                0.38486484686420286,
                0.8525720202122554,
                0.3378976624578092,
                -0.0727326195128539,
            ),
            high_pass=(
                0.0727326195128539,
                0.3378976624578092,
                -0.8525720202122554,
                0.38486484686420286,
                0.0727326195128539,
                -0.01565572813546454,
            ),
            name="coif1",
        )
