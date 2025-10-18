"""
PyTorch Lightning training script.

This script demonstrates how to train models using PyTorch Lightning
with Weights & Biases logging.

Usage:
    python train_lightning.py --config configs/experiments/wandb_example.yaml
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Uncomment when you have your model and datamodule ready
# from deep_learning_final_project.models.lightning_model import (
#     LightningCNN,
#     LightningDataModule,
# )
from deep_learning_final_project.utils.config import Config


def main() -> None:
    """Main training function using PyTorch Lightning."""
    parser = argparse.ArgumentParser(description="Train with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--offline", action="store_true", help="Run W&B in offline mode")
    args = parser.parse_args()

    # Load configuration
    config = Config.from_file(args.config)
    print(f"Loaded configuration from {args.config}")

    # Set random seed for reproducibility
    pl.seed_everything(getattr(config, "seed", 42), workers=True)

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project=config.wandb.project if hasattr(config, "wandb") else "deep-learning-project",
        name=config.experiment.name if hasattr(config, "experiment") else "experiment",
        save_dir="logs/",
        log_model=True,  # Save checkpoints to W&B
        offline=args.offline,
    )

    # Log hyperparameters to W&B
    wandb_logger.log_hyperparams(config.to_dict())

    # Setup callbacks
    callbacks = []

    # Early stopping
    if hasattr(config, "callbacks") and hasattr(config.callbacks, "early_stopping"):
        early_stop = EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            patience=config.callbacks.early_stopping.patience,
            mode=config.callbacks.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stop)

    # Model checkpointing
    checkpoint_dir = Path("checkpoints") / config.experiment.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_acc:.2f}",
        monitor="val_acc" if hasattr(config, "callbacks") else "val_loss",
        mode="max" if hasattr(config, "callbacks") else "min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs if hasattr(config, "training") else 100,
        accelerator="auto",  # Automatically select GPU/TPU/CPU
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        precision="16-mixed",  # Mixed precision training
        log_every_n_steps=10,
        deterministic=True,  # For reproducibility
    )

    # TODO: Initialize your model and datamodule
    print("\n⚠️  Please implement your model and datamodule:")
    print("  1. Uncomment the imports at the top of this file")
    print("  2. Create your model: model = LightningCNN(...)")
    print("  3. Create your datamodule: dm = LightningDataModule(...)")
    print("  4. Then run: trainer.fit(model, datamodule=dm)")
    print("\nExample:")
    print("  model = LightningCNN(")
    print("      num_classes=config.model.num_classes,")
    print("      learning_rate=config.training.learning_rate,")
    print("  )")
    print("  dm = LightningDataModule(")
    print("      data_dir=config.data.data_dir,")
    print("      batch_size=config.training.batch_size,")
    print("  )")
    print("  trainer.fit(model, datamodule=dm)")
    print("\n✨ Training will start automatically once implemented!")

    # Uncomment these lines when ready:
    # model = LightningCNN(
    #     num_classes=config.model.num_classes,
    #     learning_rate=config.training.learning_rate,
    # )
    # dm = LightningDataModule(
    #     data_dir=config.data.data_dir,
    #     batch_size=config.training.batch_size,
    # )
    # trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

    print("\n✅ Training configuration complete!")
    print(
        f"📊 View results at: https://wandb.ai/{wandb_logger.experiment.entity}/{wandb_logger.experiment.project}"
    )


if __name__ == "__main__":
    main()
