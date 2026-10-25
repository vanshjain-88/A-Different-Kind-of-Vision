import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from training.train import create_dataloaders, evaluate, train_one_epoch
from vit import ViTConfig, VisionTransformer


class FakeCIFAR10(Dataset):
    def __init__(self, train: bool) -> None:
        self.size = 50_000 if train else 10_000

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), index % 10


class TrainingTests(unittest.TestCase):
    @patch("training.train.datasets.CIFAR10")
    def test_cifar_is_split_without_using_test_for_selection(
        self, cifar_dataset: Mock
    ) -> None:
        cifar_dataset.side_effect = (
            lambda *args, train, **kwargs: FakeCIFAR10(train)
        )

        train_loader, validation_loader, test_loader = create_dataloaders(
            data_dir=Path("."),
            batch_size=128,
            num_workers=0,
            download=False,
            use_pin_memory=False,
            validation_size=5_000,
            seed=42,
        )

        self.assertEqual(len(train_loader.dataset), 45_000)
        self.assertEqual(len(validation_loader.dataset), 5_000)
        self.assertEqual(len(test_loader.dataset), 10_000)
        train_indices = set(train_loader.dataset.indices)
        validation_indices = set(validation_loader.dataset.indices)
        self.assertTrue(train_indices.isdisjoint(validation_indices))

    def test_training_and_evaluation_run_on_synthetic_batch(self) -> None:
        config = ViTConfig(
            image_size=32,
            patch_size=8,
            in_channels=3,
            embedding_dim=16,
            num_layers=1,
            num_heads=4,
            mlp_dim=32,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=10,
        )
        model = VisionTransformer(config)
        loader = DataLoader(
            TensorDataset(
                torch.randn(8, 3, 32, 32),
                torch.randint(0, 10, (8,)),
            ),
            batch_size=4,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        train_loss, train_accuracy = train_one_epoch(
            model, loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate(
            model, loader, criterion, device
        )

        self.assertGreater(train_loss, 0.0)
        self.assertGreater(test_loss, 0.0)
        self.assertGreaterEqual(train_accuracy, 0.0)
        self.assertLessEqual(train_accuracy, 1.0)
        self.assertGreaterEqual(test_accuracy, 0.0)
        self.assertLessEqual(test_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
