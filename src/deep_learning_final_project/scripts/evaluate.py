"""
Evaluation script for trained models.

This script evaluates a trained model on test data and computes various metrics.

Usage:
    python -m deep_learning_final_project.scripts.evaluate --checkpoint checkpoints/best.pth
"""

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deep_learning_final_project.utils.training import get_device

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on test data.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            predictions = output.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # TODO: Compute your metrics here
    # Example:
    # from deep_learning_final_project.utils.metrics import compute_classification_metrics
    # metrics = compute_classification_metrics(
    #     np.array(all_predictions),
    #     np.array(all_targets)
    # )

    logger.info("Evaluation complete")
    return {}


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to test data directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    args = parser.parse_args()

    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")

    # TODO: Load your model and test dataset
    # Example:
    # model = SimpleCNN(num_classes=10).to(device)
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # test_dataset = BaseDataset(args.data_dir)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    logger.info("TODO: Implement evaluation logic")

    # metrics = evaluate_model(model, test_loader, device)
    # logger.info(f"Test Metrics: {metrics}")


if __name__ == "__main__":
    main()
