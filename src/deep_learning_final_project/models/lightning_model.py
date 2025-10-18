"""
PyTorch Lightning model example.

This module demonstrates how to create models using PyTorch Lightning,
following best practices for academic research projects.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, F1Score


class LightningCNN(pl.LightningModule):
    """
    Example CNN model using PyTorch Lightning.

    This class demonstrates PyTorch Lightning best practices:
    - Automatic optimization
    - Built-in logging
    - Easy multi-GPU support
    - Integration with TorchMetrics

    Args:
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout probability

    Example:
        >>> model = LightningCNN(num_classes=10)
        >>> trainer = pl.Trainer(max_epochs=10)
        >>> trainer.fit(model, train_loader, val_loader)
    """

    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        # Save hyperparameters (logged automatically to W&B/TensorBoard)
        self.save_hyperparameters()

        # Model architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Metrics using TorchMetrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global pooling and classifier
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step (called for each batch).

        Lightning handles optimization automatically.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Update metrics
        self.train_acc(logits, y)
        self.train_f1(logits, y)

        # Log metrics (automatically sent to W&B/TensorBoard)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step (called for each validation batch)."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Update metrics
        self.val_acc(logits, y)
        self.val_f1(logits, y)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step (called for each test batch)."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Update metrics
        self.test_acc(logits, y)

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self) -> dict:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and optionally lr_scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class LightningDataModule(pl.LightningDataModule):
    """
    Example DataModule for organizing data loading.

    DataModules encapsulate all data-related logic:
    - Downloading/preparing data
    - Loading datasets
    - Creating dataloaders

    Example:
        >>> dm = LightningDataModule(data_dir="./data", batch_size=32)
        >>> trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """
        Download data if needed (called only on 1 GPU/TPU).

        Example: Download MNIST, CIFAR, etc.
        """
        # TODO: Implement data download
        pass

    def setup(self, stage: str | None = None) -> None:
        """
        Load data and create train/val/test splits.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # TODO: Load your datasets here
        # Example:
        # self.train_dataset = YourDataset(train=True)
        # self.val_dataset = YourDataset(train=False)
        pass

    def train_dataloader(self):
        """Return training dataloader."""
        # TODO: Return DataLoader for training
        pass

    def val_dataloader(self):
        """Return validation dataloader."""
        # TODO: Return DataLoader for validation
        pass

    def test_dataloader(self):
        """Return test dataloader."""
        # TODO: Return DataLoader for testing
        pass
