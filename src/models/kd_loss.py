"""
Knowledge Distillation Loss Functions.

Implements the standard KD loss for binary classification:

L_KD = ω * T² * KL(p_teacher^T || p_student^T) + (1 - ω) * L_BCE(y, p_student)

Where:
- T is temperature for softening logits
- ω (alpha) is the weight for KD loss vs hard label loss
- p^T = sigmoid(z/T) are temperature-scaled predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from src.config import KDConfig


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss for binary classification.
    
    Combines:
    1. Soft label loss: KL divergence between teacher and student soft predictions
    2. Hard label loss: BCE loss between student predictions and ground truth
    
    L = α * T² * L_soft + (1 - α) * L_hard
    
    The T² scaling compensates for the reduced gradient magnitude when using
    temperature scaling (Hinton et al., 2015).
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            temperature: Temperature for softening logits. Higher T = softer.
                        Recommended: T ∈ {1, 2}
            alpha: Weight for KD loss. α = 0.5 balances KD and hard labels.
                  Higher α = more weight on teacher's soft labels.
                  Recommended: α ∈ {0.5, 0.9}
            pos_weight: Positive class weight for BCE loss (handles imbalance)
        """
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        
        # Hard label loss (can use focal loss or weighted BCE)
        self.hard_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined KD loss.
        
        Args:
            student_logits: Raw logits from student (N,) or (N, 1)
            teacher_logits: Raw logits from teacher (N,) or (N, 1)
            targets: Ground truth binary labels (N,)
        
        Returns:
            Dict with 'loss' (total), 'soft_loss', 'hard_loss'
        """
        # Ensure proper shapes
        student_logits = student_logits.view(-1)
        teacher_logits = teacher_logits.view(-1)
        targets = targets.view(-1).float()
        
        # Temperature-scaled soft predictions
        student_soft = torch.sigmoid(student_logits / self.temperature)
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        
        # Soft label loss: Binary KL divergence
        # KL(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        # For numerical stability, we use BCE formulation
        soft_loss = self._binary_kl_divergence(teacher_soft, student_soft)
        soft_loss = soft_loss * (self.temperature ** 2)  # Scale by T²
        
        # Hard label loss
        hard_loss = self.hard_loss_fn(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return {
            "loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_loss,
        }
    
    def _binary_kl_divergence(
        self,
        teacher_prob: torch.Tensor,
        student_prob: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Compute binary KL divergence: KL(teacher || student).
        
        Uses numerically stable formulation.
        """
        # Clamp for numerical stability
        teacher_prob = torch.clamp(teacher_prob, eps, 1 - eps)
        student_prob = torch.clamp(student_prob, eps, 1 - eps)
        
        # KL divergence for binary distributions
        kl = (
            teacher_prob * torch.log(teacher_prob / student_prob) +
            (1 - teacher_prob) * torch.log((1 - teacher_prob) / (1 - student_prob))
        )
        
        return kl.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reduces loss for well-classified examples, focusing on hard examples.
    Particularly effective for class-imbalanced datasets like HAM10000.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: Focusing parameter. Higher γ = more focus on hard examples.
                  γ = 0 equivalent to BCE. Recommended: γ = 2
            alpha: Weight for positive class. Should be set to handle
                  class imbalance. For ~11% melanoma prevalence, α ≈ 0.75
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw logits (N,) or (N, 1)
            targets: Binary labels (N,)
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        # For positive samples: (1 - p)^γ
        # For negative samples: p^γ
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE loss (without reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # Apply focal and alpha weights
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class KDFocalLoss(nn.Module):
    """
    Knowledge Distillation with Focal Loss for hard labels.
    
    Combines:
    1. Soft label loss: KL divergence with temperature
    2. Hard label loss: Focal loss for handling class imbalance
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        kd_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ):
        """
        Args:
            temperature: KD temperature
            kd_alpha: Weight for KD loss vs focal loss
            focal_gamma: Focal loss focusing parameter
            focal_alpha: Focal loss class weight
        """
        super().__init__()
        self.temperature = temperature
        self.kd_alpha = kd_alpha
        
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined KD + Focal loss."""
        student_logits = student_logits.view(-1)
        teacher_logits = teacher_logits.view(-1)
        targets = targets.view(-1).float()
        
        # Soft label loss
        student_soft = torch.sigmoid(student_logits / self.temperature)
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        
        soft_loss = self._binary_kl_divergence(teacher_soft, student_soft)
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Hard label loss (focal)
        hard_loss = self.focal_loss(student_logits, targets)
        
        # Combined
        total_loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss
        
        return {
            "loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_loss,
        }
    
    def _binary_kl_divergence(
        self,
        teacher_prob: torch.Tensor,
        student_prob: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        teacher_prob = torch.clamp(teacher_prob, eps, 1 - eps)
        student_prob = torch.clamp(student_prob, eps, 1 - eps)
        
        kl = (
            teacher_prob * torch.log(teacher_prob / student_prob) +
            (1 - teacher_prob) * torch.log((1 - teacher_prob) / (1 - student_prob))
        )
        return kl.mean()


def create_kd_loss(
    config: Optional[KDConfig] = None,
    pos_weight: Optional[torch.Tensor] = None,
    use_focal: bool = True,
) -> nn.Module:
    """
    Factory function to create KD loss.
    
    Args:
        config: KDConfig with temperature and alpha settings
        pos_weight: Positive class weight for BCE (ignored if use_focal=True)
        use_focal: Use focal loss for hard labels (recommended for imbalanced data)
    
    Returns:
        KD loss module
    """
    config = config or KDConfig()
    
    if use_focal:
        return KDFocalLoss(
            temperature=config.temperature,
            kd_alpha=config.alpha,
            focal_gamma=2.0,
            focal_alpha=0.75,
        )
    else:
        return KnowledgeDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            pos_weight=pos_weight,
        )


def create_teacher_loss(
    loss_type: str = "focal",
    pos_weight: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
) -> nn.Module:
    """
    Create loss function for teacher training.
    
    Args:
        loss_type: 'bce', 'weighted_bce', or 'focal'
        pos_weight: Positive class weight for weighted BCE
        focal_gamma: Focusing parameter for focal loss
        focal_alpha: Class weight for focal loss
    """
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    elif loss_type == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss()
