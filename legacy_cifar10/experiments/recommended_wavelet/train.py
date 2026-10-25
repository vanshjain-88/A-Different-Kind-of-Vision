"""Train the isolated recommended-wavelet experiment on CIFAR-10."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from experiments.recommended_wavelet.model import RecommendedWaveletViT
from training.train import (
    choose_device,
    create_dataloaders,
    evaluate,
    train_one_epoch,
)
from vit import ViTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the isolated SPT and wavelet-token experiment."
    )
    parser.add_argument(
        "--wavelet",
        choices=("haar", "biorthogonal"),
        default="haar",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir or (
        Path("experiments/recommended_wavelet/results") / args.wavelet
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    metrics_path = output_dir / "epoch_metrics.csv"
    test_path = output_dir / "final_test_metrics.csv"

    config = ViTConfig(
        tokenizer="patch",
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
    )
    model = RecommendedWaveletViT(config, wavelet=args.wavelet).to(device)
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
    print(f"Wavelet: {args.wavelet}")
    print(f"Results: {output_dir}")
    print(
        f"Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    best_validation_accuracy = -1.0
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            ),
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, criterion, device
            )
            writer.writerow(
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
                torch.save(
                    {
                        "epoch": epoch,
                        "model_config": asdict(config),
                        "experiment": {
                            "name": "recommended_wavelet",
                            "wavelet": args.wavelet,
                        },
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_accuracy": validation_accuracy,
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )
    with test_path.open("w", newline="") as test_file:
        writer = csv.DictWriter(
            test_file,
            fieldnames=("checkpoint_epoch", "test_loss", "test_accuracy"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "checkpoint_epoch": checkpoint["epoch"],
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
    print(
        f"Best validation accuracy: {best_validation_accuracy:.2%} | "
        f"test accuracy: {test_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
