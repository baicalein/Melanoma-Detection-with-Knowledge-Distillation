# Weights & Biases (W&B) Guide

This guide explains how to use Weights & Biases for experiment tracking in your deep learning research project.

> [!NOTE]
> Weights & Biases is a powerful experiment tracking platform that integrates seamlessly with PyTorch Lightning.

## 🚀 Getting Started with W&B

### Installation and Setup

```bash
# Already installed via pyproject.toml
pip install wandb

# Login with your W&B account
wandb login

# Or set API key via environment variable
export WANDB_API_KEY=your_api_key_here
```

### First Experiment

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Initialize W&B logger
wandb_logger = WandbLogger(
    project="deep-learning-final-project",
    name="my-first-experiment",
    save_dir="logs/",
    log_model=True  # Save model checkpoints to W&B
)

# Use with Lightning Trainer
trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=100,
    accelerator="auto"
)

# Train model (metrics logged automatically)
trainer.fit(model, datamodule=dm)
```

## 📊 What W&B Tracks Automatically

When using PyTorch Lightning with W&B:

✅ **Training metrics** - All metrics logged via `self.log()`
✅ **System metrics** - GPU/CPU usage, memory, etc.
✅ **Hyperparameters** - From `self.save_hyperparameters()`
✅ **Model architecture** - Network graph visualization
✅ **Code version** - Git commit hash and diff
✅ **Environment** - Python version, package versions

## 🎯 Core Features for Research

### 1. Metric Logging

```python
class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        
        # Automatically logged to W&B
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        
        return loss
```

### 2. Image Logging

```python
import wandb

# In your validation_step or test_step
def validation_step(self, batch, batch_idx):
    if batch_idx == 0:  # Log only first batch
        images, labels = batch
        predictions = self(images)
        
        # Log images with predictions
        self.logger.log_image(
            key="val_predictions",
            images=[img.cpu() for img in images[:8]],
            caption=[f"Pred: {p}, True: {t}" 
                    for p, t in zip(predictions[:8], labels[:8])]
        )
```

### 3. Custom Tables

```python
import wandb

# Create a table for detailed results
columns = ["id", "prediction", "confidence", "true_label", "correct"]
data = []

for i, (pred, conf, true) in enumerate(zip(predictions, confidences, labels)):
    data.append([i, pred, conf, true, pred == true])

table = wandb.Table(columns=columns, data=data)
wandb.log({"predictions_table": table})
```

### 4. Artifact Tracking

```python
# Save dataset as artifact
with wandb.init() as run:
    artifact = wandb.Artifact("my-dataset", type="dataset")
    artifact.add_dir("data/processed")
    run.log_artifact(artifact)

# Save model checkpoint as artifact
artifact = wandb.Artifact("my-model", type="model")
artifact.add_file("checkpoints/best.ckpt")
run.log_artifact(artifact)
```

## 🔬 Academic Research Workflows

### Comparing Experiments

```python
# Group related experiments
wandb_logger = WandbLogger(
    project="deep-learning-final-project",
    group="ablation-study",  # Groups experiments together
    job_type="train",
    tags=["cnn", "baseline", "no-augmentation"]
)
```

### Sweeps for Hyperparameter Tuning

Create `sweep_config.yaml`:
```yaml
program: train.py
method: bayes  # bayes, grid, or random
metric:
  name: val_acc
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [16, 32, 64]
  dropout_rate:
    min: 0.1
    max: 0.5
```

Run sweep:
```bash
# Initialize sweep
wandb sweep sweep_config.yaml

# Run sweep agent (can run multiple in parallel)
wandb agent your-entity/project/sweep-id
```

### Reports for Papers

> [!TIP]
> Create W&B reports to share results with your advisor or include in papers.

1. Go to your project on wandb.ai
2. Select runs to compare
3. Click "Create Report"
4. Add plots, tables, and text
5. Export as PDF or share link

## 📈 Visualization Features

### Parallel Coordinates Plot

Compare multiple hyperparameters and their effect on metrics:

```python
# In W&B web interface:
# 1. Go to project workspace
# 2. Click "Add plot" → "Parallel Coordinates"
# 3. Select parameters to visualize
```

### Custom Charts

```python
# Log data for custom visualization
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=ground_truth,
        preds=predictions,
        class_names=class_names
    )
})

# ROC curve
wandb.log({
    "roc_curve": wandb.plot.roc_curve(
        y_true=ground_truth,
        y_probas=probabilities,
        labels=class_names
    )
})
```

## 🎓 Best Practices for Academic Research

### 1. Consistent Naming Convention

```python
# Use descriptive, consistent names
experiment_name = f"{model_type}_{dataset}_{timestamp}"
# Example: "resnet50_cifar10_2024-01-15_14-30"

wandb_logger = WandbLogger(
    project="my-research-project",
    name=experiment_name,
    tags=[model_type, dataset, "experiment"]
)
```

### 2. Save Configuration

```python
# Save all hyperparameters
config = {
    "model": "ResNet50",
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "Adam",
    "dataset": "CIFAR-10",
    "random_seed": 42
}

wandb.config.update(config)
```

### 3. Document Experiments

```python
# Add detailed notes
wandb_logger.experiment.notes = """
Experiment Notes:
- Testing effect of data augmentation
- Baseline model from Smith et al. (2023)
- Changed dropout from 0.3 to 0.5
"""

# Add tags for filtering
wandb_logger.experiment.tags.append("submitted-to-neurips")
```

### 4. Reproducibility

```python
# Log git commit hash
import subprocess

git_hash = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode("ascii").strip()

wandb.config.update({"git_commit": git_hash})

# Log random seed
wandb.config.update({"random_seed": 42})
```

## 🔒 Privacy and Collaboration

### Private Projects

```bash
# Create private project (default)
wandb_logger = WandbLogger(
    project="my-private-research",
    entity="your-username"
)
```

### Team Collaboration

```bash
# Share with team
wandb_logger = WandbLogger(
    project="shared-research",
    entity="your-team-name"  # Team entity
)
```

### Offline Mode

```bash
# Work offline (syncs later)
export WANDB_MODE=offline

# Or in code
wandb.init(mode="offline")

# Sync later when online
wandb sync wandb/offline-run-*
```

## 🐛 Troubleshooting

### Common Issues

**Q: W&B is slow or timing out**
```python
# Reduce logging frequency
self.log("train_loss", loss, on_step=False, on_epoch=True)

# Or disable syncing temporarily
wandb.init(mode="disabled")
```

**Q: Too many logged metrics**
```python
# Log only important metrics
if batch_idx % 100 == 0:  # Log every 100 batches
    self.log("detailed_metric", value)
```

**Q: Large checkpoint files**
```python
# Don't log all checkpoints
wandb_logger = WandbLogger(
    project="project",
    log_model=False  # Disable automatic checkpoint logging
)

# Or log only best checkpoint
# (use ModelCheckpoint callback with save_top_k=1)
```

## 📚 Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [PyTorch Lightning Integration](https://docs.wandb.ai/guides/integrations/lightning)
- [W&B Examples](https://github.com/wandb/examples)
- [W&B Python Library](https://github.com/wandb/wandb)

## 💡 Tips for Academic Papers

1. **Export high-quality figures** from W&B reports
2. **Share experiment links** in supplementary materials
3. **Use W&B artifacts** to version datasets and models
4. **Create comparison tables** directly from W&B
5. **Document ablation studies** with grouped experiments

---

> [!WARNING]
> Remember to add `.wandb/` to `.gitignore` to avoid committing W&B cache files.

For questions or issues, check the [W&B Community Forum](https://community.wandb.ai/).
