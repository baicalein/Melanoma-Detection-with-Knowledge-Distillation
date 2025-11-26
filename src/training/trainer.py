"""
Training utilities for teacher and student models.

Provides:
- TeacherTrainer: Train teacher model with focal loss
- StudentTrainer: Train student with knowledge distillation
- Training loop with early stopping, checkpointing, and logging
"""

import json
import logging
import pathlib
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import (
    TrainingConfig,
    KDConfig,
    WandbConfig,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    set_seed,
)
from src.models.kd_loss import create_teacher_loss, create_kd_loss, FocalLoss
from src.evaluation.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def _safe_wandb_log(metrics: Dict[str, Any], step: int = None) -> None:
    """Log metrics to W&B if available."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
    except Exception:
        pass


@dataclass
class TrainingHistory:
    """Container for training history."""
    
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    val_f1: List[float]
    val_roc_auc: List[float]
    learning_rates: List[float]
    
    # Best metrics
    best_epoch: int = 0
    best_val_roc_auc: float = 0.0
    best_val_f1: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseTrainer:
    """Base trainer with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cpu",
        experiment_name: str = "experiment",
        checkpoint_dir: Optional[pathlib.Path] = None,
        wandb_config: Optional[WandbConfig] = None,
        wandb_tags: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B integration
        self.wandb_run = None
        if wandb_config is not None and wandb_config.enabled:
            self.wandb_run = wandb_config.init_wandb(
                run_name=experiment_name,
                config={
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "max_epochs": config.max_epochs,
                    "scheduler": config.scheduler,
                    "loss_type": config.loss_type,
                    "focal_gamma": config.focal_gamma,
                    "focal_alpha": config.focal_alpha,
                    "device": device,
                },
                tags=wandb_tags,
            )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and device == "cuda" else None
        
        # Initialize history
        self.history = TrainingHistory(
            train_loss=[],
            val_loss=[],
            train_acc=[],
            val_acc=[],
            val_f1=[],
            val_roc_auc=[],
            learning_rates=[],
        )
        
        # Best tracking
        self.best_metric = 0.0
        self.epochs_no_improve = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # Maximize ROC-AUC
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=15,
                gamma=0.1,
            )
        return None
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> pathlib.Path:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else {},
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            
            # Save metadata
            meta_path = self.checkpoint_dir / f"{self.experiment_name}_best_meta.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "metrics": metrics,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)
            
            logger.info(f"Saved best checkpoint: epoch {epoch}, ROC-AUC={metrics.get('roc_auc', 0):.4f}")
            return best_path
        
        return latest_path
    
    def load_checkpoint(self, path: pathlib.Path) -> Dict[str, Any]:
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def _finish_wandb(self) -> None:
        """Finish W&B run and log final artifacts."""
        if self.wandb_run is not None:
            try:
                import wandb
                
                # Log final summary
                wandb.summary["best_epoch"] = self.history.best_epoch
                wandb.summary["best_val_roc_auc"] = self.history.best_val_roc_auc
                wandb.summary["best_val_f1"] = self.history.best_val_f1
                
                # Log model artifact if configured
                best_ckpt = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
                if best_ckpt.exists():
                    artifact = wandb.Artifact(
                        name=f"{self.experiment_name}_model",
                        type="model",
                        description=f"Best {self.experiment_name} checkpoint",
                    )
                    artifact.add_file(str(best_ckpt))
                    wandb.log_artifact(artifact)
                
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")


class TeacherTrainer(BaseTrainer):
    """Trainer for teacher model with focal loss."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cpu",
        experiment_name: str = "teacher",
        pos_weight: Optional[torch.Tensor] = None,
        wandb_config: Optional[WandbConfig] = None,
    ):
        super().__init__(
            model, train_loader, val_loader, config, device, experiment_name,
            wandb_config=wandb_config,
            wandb_tags=["teacher", config.loss_type],
        )
        
        # Create loss function
        self.criterion = create_teacher_loss(
            loss_type=config.loss_type,
            pos_weight=pos_weight.to(device) if pos_weight is not None else None,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
        )
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == targets.long()).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate and compute metrics."""
        self.model.eval()
        
        all_targets = []
        all_probs = []
        total_loss = 0.0
        
        for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())
        
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)
        
        metrics = compute_classification_metrics(y_true, y_prob)
        metrics_dict = {
            "val_loss": total_loss / len(y_true),
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "specificity": metrics.specificity,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "ece": metrics.ece,
            "specificity_at_95sens": metrics.specificity_at_target_sens,
            "ppv_at_95sens": metrics.ppv_at_target_sens,
            "npv_at_95sens": metrics.npv_at_target_sens,
        }
        
        return metrics_dict
    
    def train(self, num_epochs: Optional[int] = None) -> TrainingHistory:
        """Full training loop with early stopping."""
        num_epochs = num_epochs or self.config.max_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}, Loss: {self.config.loss_type}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_metrics["val_loss"])
            self.history.train_acc.append(train_acc)
            self.history.val_acc.append(val_metrics["accuracy"])
            self.history.val_f1.append(val_metrics["f1"])
            self.history.val_roc_auc.append(val_metrics["roc_auc"])
            self.history.learning_rates.append(current_lr)
            
            # Check for improvement (using ROC-AUC)
            is_best = val_metrics["roc_auc"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["roc_auc"]
                self.history.best_epoch = epoch + 1
                self.history.best_val_roc_auc = val_metrics["roc_auc"]
                self.history.best_val_f1 = val_metrics["f1"]
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"ECE: {val_metrics['ece']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Log to W&B
            _safe_wandb_log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_metrics["val_loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "val/specificity": val_metrics["specificity"],
                "val/roc_auc": val_metrics["roc_auc"],
                "val/pr_auc": val_metrics["pr_auc"],
                "val/ece": val_metrics["ece"],
                "val/specificity_at_95sens": val_metrics["specificity_at_95sens"],
                "learning_rate": current_lr,
                "best_roc_auc": self.best_metric,
            }, step=epoch + 1)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["roc_auc"])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.epochs_no_improve} epochs "
                    f"without improvement"
                )
                break
        
        # Save history
        history_path = LOGS_DIR / f"{self.experiment_name}_history.json"
        self.history.save(history_path)
        logger.info(f"Saved training history to {history_path}")
        
        # Finish W&B run
        self._finish_wandb()
        
        return self.history


class StudentTrainer(BaseTrainer):
    """Trainer for student model with knowledge distillation."""
    
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: TrainingConfig,
        kd_config: KDConfig,
        device: str = "cpu",
        experiment_name: str = "student",
        wandb_config: Optional[WandbConfig] = None,
    ):
        # Initialize W&B with KD-specific config
        self.kd_config = kd_config
        
        super().__init__(
            student, train_loader, val_loader, training_config, device, experiment_name,
            wandb_config=wandb_config,
            wandb_tags=["student", "kd", f"T{kd_config.temperature}", f"alpha{kd_config.alpha}"],
        )
        
        # Update W&B config with KD params
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.config.update({
                    "kd_temperature": kd_config.temperature,
                    "kd_alpha": kd_config.alpha,
                    "kd_loss_type": kd_config.loss_type,
                })
            except Exception:
                pass
        
        self.teacher = teacher.to(device)
        self.teacher.eval()
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Create KD loss
        self.criterion = create_kd_loss(kd_config, use_focal=True)
        
        logger.info(
            f"KD Config: T={kd_config.temperature}, α={kd_config.alpha}"
        )
    
    def train_epoch(self) -> Tuple[float, float, float, float]:
        """Train for one epoch with KD."""
        self.model.train()
        self.teacher.eval()
        
        total_loss = 0.0
        total_soft_loss = 0.0
        total_hard_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="KD Training", leave=False)
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Get teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            
            # Student forward pass
            if self.scaler is not None:
                with autocast():
                    student_logits = self.model(images)
                    loss_dict = self.criterion(student_logits, teacher_logits, targets)
                    loss = loss_dict["loss"]
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                student_logits = self.model(images)
                loss_dict = self.criterion(student_logits, teacher_logits, targets)
                loss = loss_dict["loss"]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * images.size(0)
            total_soft_loss += loss_dict["soft_loss"].item() * images.size(0)
            total_hard_loss += loss_dict["hard_loss"].item() * images.size(0)
            
            preds = (torch.sigmoid(student_logits) >= 0.5).long()
            correct += (preds == targets.long()).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "soft": loss_dict["soft_loss"].item(),
                "hard": loss_dict["hard_loss"].item(),
            })
        
        return (
            total_loss / total,
            total_soft_loss / total,
            total_hard_loss / total,
            correct / total,
        )
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate student model."""
        self.model.eval()
        
        all_targets = []
        all_probs = []
        
        for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device)
            
            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(targets.numpy())
        
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)
        
        metrics = compute_classification_metrics(y_true, y_prob)
        
        return {
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "ece": metrics.ece,
            "specificity_at_95sens": metrics.specificity_at_target_sens,
        }
    
    def train(self, num_epochs: Optional[int] = None) -> TrainingHistory:
        """Full KD training loop."""
        num_epochs = num_epochs or self.config.max_epochs
        
        logger.info(f"Starting KD training for {num_epochs} epochs")
        logger.info(f"Temperature: {self.kd_config.temperature}, Alpha: {self.kd_config.alpha}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, soft_loss, hard_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(0.0)  # Not computed for KD
            self.history.train_acc.append(train_acc)
            self.history.val_acc.append(val_metrics["accuracy"])
            self.history.val_f1.append(val_metrics["f1"])
            self.history.val_roc_auc.append(val_metrics["roc_auc"])
            self.history.learning_rates.append(current_lr)
            
            # Check for improvement
            is_best = val_metrics["roc_auc"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["roc_auc"]
                self.history.best_epoch = epoch + 1
                self.history.best_val_roc_auc = val_metrics["roc_auc"]
                self.history.best_val_f1 = val_metrics["f1"]
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Loss: {train_loss:.4f} (soft: {soft_loss:.4f}, hard: {hard_loss:.4f}) | "
                f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"ECE: {val_metrics['ece']:.4f}"
            )
            
            # Log to W&B
            _safe_wandb_log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/soft_loss": soft_loss,
                "train/hard_loss": hard_loss,
                "train/accuracy": train_acc,
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1"],
                "val/roc_auc": val_metrics["roc_auc"],
                "val/pr_auc": val_metrics["pr_auc"],
                "val/ece": val_metrics["ece"],
                "val/specificity_at_95sens": val_metrics["specificity_at_95sens"],
                "kd/temperature": self.kd_config.temperature,
                "kd/alpha": self.kd_config.alpha,
                "best_roc_auc": self.best_metric,
            }, step=epoch + 1)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["roc_auc"])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save history
        history_path = LOGS_DIR / f"{self.experiment_name}_history.json"
        self.history.save(history_path)
        
        # Finish W&B run
        self._finish_wandb()
        
        return self.history
