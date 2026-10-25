from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for the complete Vision Transformer."""

    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    tokenizer: str = "biorthogonal"
    embedding_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    num_classes: int = 10

    def __post_init__(self) -> None:
        positive_fields = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "num_classes": self.num_classes,
        }
        for name, value in positive_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, "
                f"got image_size={self.image_size} and patch_size={self.patch_size}."
            )

        if self.patch_size % 2 != 0:
            raise ValueError(
                "patch_size must be even so the wavelet and RGB branches "
                f"can produce aligned tokens, got {self.patch_size}."
            )

        valid_tokenizers = {"patch", "biorthogonal", "coiflet"}
        if self.tokenizer not in valid_tokenizers:
            raise ValueError(
                f"tokenizer must be one of {sorted(valid_tokenizers)}, "
                f"got {self.tokenizer!r}."
            )

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads, "
                f"got embedding_dim={self.embedding_dim} and "
                f"num_heads={self.num_heads}."
            )

        probability_fields = {
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
        }
        for name, value in probability_fields.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= value < 1.0
            ):
                raise ValueError(
                    f"{name} must be in the range [0, 1), got {value!r}."
                )

        if (
            not isinstance(self.layer_norm_eps, (int, float))
            or isinstance(self.layer_norm_eps, bool)
            or self.layer_norm_eps <= 0
        ):
            raise ValueError(
                "layer_norm_eps must be positive, "
                f"got {self.layer_norm_eps!r}."
            )

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_per_side**2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size**2

    @property
    def wavelet_patch_size(self) -> int:
        return self.patch_size // 2

    @property
    def wavelet_patch_dim(self) -> int:
        return 4 * self.in_channels * self.wavelet_patch_size**2
