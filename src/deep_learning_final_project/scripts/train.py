"""
Training script for deep learning models.

This script provides a template for training models with proper logging,
checkpointing, and metric tracking. Customize for your specific needs.

Usage:
    python -m deep_learning_final_project.scripts.train --config configs/experiments/baseline.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from deep_learning_final_project.data.dataset import BaseDataset
# from deep_learning_final_project.models.example_model import SimpleCNN
from deep_learning_final_project.utils.config import Config
from deep_learning_final_project.utils.metrics import MetricTracker
from deep_learning_final_project.utils.training import (
    AverageMeter,
    EarlyStopping,
    get_device,
    set_seed,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary of training metrics
    """
    model.train()
    loss_meter = AverageMeter()
    metric_tracker = MetricTracker()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        loss_meter.update(loss.item(), data.size(0))

        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch: {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f}"
            )

    return {"loss": loss_meter.avg}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Validate model on validation set.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            loss_meter.update(loss.item(), data.size(0))

    return {"loss": loss_meter.avg}


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a deep learning model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_file(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Set random seed for reproducibility
    set_seed(getattr(config, "seed", 42))

    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")

    # TODO: Initialize your model, dataset, and optimizer here
    # Example:
    # model = SimpleCNN(num_classes=config.num_classes).to(device)
    # train_dataset = BaseDataset(config.data_dir)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # criterion = nn.CrossEntropyLoss()

    logger.info("Training setup complete. Ready to train!")
    logger.info("TODO: Implement your training loop here")

    # Example training loop structure:
    # for epoch in range(config.num_epochs):
    #     train_metrics = train_one_epoch(
    #         model, train_loader, criterion, optimizer, device, epoch
    #     )
    #     val_metrics = validate(model, val_loader, criterion, device)
    #
    #     logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
    #                 f"Val Loss: {val_metrics['loss']:.4f}")
    #
    #     # Save checkpoint
    #     if epoch % config.save_freq == 0:
    #         checkpoint_path = Path(config.checkpoint_dir) / f"epoch_{epoch}.pth"
    #         torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
