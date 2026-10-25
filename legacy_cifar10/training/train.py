"""Train the Vision Transformer on CIFAR-10.

Run from the project root:
    python -m training.train

Use --help to see every model and training parameter.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from vit import ViTConfig, VisionTransformer


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the wavelet Vision Transformer on CIFAR-10."
    )

    training = parser.add_argument_group("training")
    training.add_argument("--data-dir", type=Path, default=Path("data"))
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--validation-size", type=int, default=5000)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--num-workers", type=int, default=2)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    training.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("training/checkpoints/best.pt"),
    )
    training.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("training/checkpoints/epoch_metrics.csv"),
        help="CSV path for per-epoch training and validation metrics.",
    )
    training.add_argument(
        "--test-csv",
        type=Path,
        default=Path("training/checkpoints/final_test_metrics.csv"),
        help="CSV path for final test metrics from the best checkpoint.",
    )
    training.add_argument(
        "--no-download",
        action="store_false",
        dest="download",
        help="Require CIFAR-10 to already exist in --data-dir.",
    )
    training.set_defaults(download=True)

    model = parser.add_argument_group("model")
    model.add_argument("--image-size", type=int, default=32)
    model.add_argument("--patch-size", type=int, default=4)
    model.add_argument("--in-channels", type=int, default=3)
    model.add_argument(
        "--tokenizer",
        choices=("patch", "biorthogonal", "coiflet"),
        default="biorthogonal",
        help="Input tokenizer used for controlled ablation experiments.",
    )
    model.add_argument("--embedding-dim", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=6)
    model.add_argument("--num-heads", type=int, default=8)
    model.add_argument("--mlp-dim", type=int, default=512)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--attention-dropout", type=float, default=0.0)
    model.add_argument("--layer-norm-eps", type=float, default=1e-6)

    return parser.parse_args()


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    download: bool,
    use_pin_memory: bool,
    validation_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )

    if not 0 < validation_size < len(full_train_dataset):
        raise ValueError(
            "validation_size must be between 1 and the number of training "
            f"examples minus 1, got {validation_size}."
        )

    train_size = len(full_train_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    return train_loader, validation_loader, test_loader


def train_one_epoch(
    model: VisionTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: VisionTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = ViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        tokenizer=args.tokenizer,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        layer_norm_eps=args.layer_norm_eps,
        num_classes=10,
    )
    model = VisionTransformer(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    train_loader, validation_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        use_pin_memory=device.type == "cuda",
        validation_size=args.validation_size,
        seed=args.seed,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Configuration: {config}")
    print(
        f"Dataset split: {len(train_loader.dataset):,} train | "
        f"{len(validation_loader.dataset):,} validation | "
        f"{len(test_loader.dataset):,} test"
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.test_csv.parent.mkdir(parents=True, exist_ok=True)

    best_validation_accuracy = -1.0
    best_epoch = 0
    with args.metrics_csv.open("w", newline="") as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            ),
        )
        metrics_writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, criterion, device
            )

            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )
            metrics_file.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train loss {train_loss:.4f} | "
                f"train accuracy {train_accuracy:.2%} | "
                f"validation loss {validation_loss:.4f} | "
                f"validation accuracy {validation_accuracy:.2%}"
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_config": asdict(config),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_accuracy": validation_accuracy,
                    },
                    args.checkpoint,
                )
                print(f"Saved best checkpoint to {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    with args.test_csv.open("w", newline="") as test_file:
        test_writer = csv.DictWriter(
            test_file,
            fieldnames=("checkpoint_epoch", "test_loss", "test_accuracy"),
        )
        test_writer.writeheader()
        test_writer.writerow(
            {
                "checkpoint_epoch": checkpoint["epoch"],
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

    print(f"Best validation accuracy: {best_validation_accuracy:.2%}")
    print(
        f"Final test loss: {test_loss:.4f} | "
        f"final test accuracy: {test_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
