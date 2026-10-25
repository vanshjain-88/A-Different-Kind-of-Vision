"""Run the ViT input pipeline on real CIFAR-10 images.

This script also reverses patchification. If the reconstructed images exactly
match the inputs, no pixel was lost, duplicated, or placed in the wrong patch.
"""

import pickle
from pathlib import Path

import torch

from vit import ViTConfig, VisionTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "cifar-10-batches-py"


def load_cifar_batch(
    batch_path: Path, number_of_images: int
) -> tuple[torch.Tensor, list[int]]:
    """Load CIFAR's Python batch as float images in [B, C, H, W] format."""
    with batch_path.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")

    # CIFAR stores each image as 3,072 uint8 values:
    # 1,024 red values, then 1,024 green, then 1,024 blue.
    flat_images = batch[b"data"][:number_of_images]
    images = torch.from_numpy(flat_images).reshape(-1, 3, 32, 32)
    images = images.float() / 255.0
    labels = batch[b"labels"][:number_of_images]
    return images, labels


def load_label_names(metadata_path: Path) -> list[str]:
    """Load readable CIFAR-10 class names."""
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file, encoding="bytes")
    return [name.decode("utf-8") for name in metadata[b"label_names"]]


def unpatchify(patches: torch.Tensor, config: ViTConfig) -> torch.Tensor:
    """Reverse PatchEmbedding.patchify for a pixel-perfect correctness check."""
    batch_size = patches.shape[0]
    grid_size = config.patches_per_side
    patch_size = config.patch_size

    # [B, N, C*P*P] -> [B, H/P, W/P, C, P, P]
    patch_grid = patches.reshape(
        batch_size,
        grid_size,
        grid_size,
        config.in_channels,
        patch_size,
        patch_size,
    )

    # Undo patchify's permutation:
    # [B, H/P, W/P, C, P, P] -> [B, C, H/P, P, W/P, P]
    patch_grid = patch_grid.permute(0, 3, 1, 4, 2, 5)

    # Join the patch grid back into complete images.
    return patch_grid.reshape(
        batch_size,
        config.in_channels,
        config.image_size,
        config.image_size,
    )


def main() -> None:
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=128,
        num_layers=6,
        num_heads=8,
        mlp_dim=512,
        num_classes=10,
    )
    model = VisionTransformer(config)
    model.eval()

    images, labels = load_cifar_batch(DATASET_DIR / "test_batch", 4)
    label_names = load_label_names(DATASET_DIR / "batches.meta")

    patches = model.patch_embedding.patchify(images)
    reconstructed_images = unpatchify(patches, config)
    input_tokens = model.patch_embedding(images)
    encoded_tokens = model.forward_features(images)
    logits = model(images)

    exact_match = torch.equal(images, reconstructed_images)
    maximum_pixel_error = (images - reconstructed_images).abs().max().item()

    print(f"Classes: {[label_names[label] for label in labels]}")
    print(f"Input images:         {list(images.shape)}")
    print(f"Patchified:           {list(patches.shape)}")
    print(f"One flattened patch:  {list(patches[0, 0].shape)}")
    print(f"Encoder input tokens: {list(input_tokens.shape)}")
    print(f"Encoded tokens:       {list(encoded_tokens.shape)}")
    print(f"Class logits:         {list(logits.shape)}")
    print(f"Exact reconstruction: {exact_match}")
    print(f"Maximum pixel error:  {maximum_pixel_error}")

    if not exact_match:
        raise AssertionError("Patchification failed the reconstruction check.")


if __name__ == "__main__":
    main()
