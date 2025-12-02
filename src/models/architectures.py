"""
Teacher and Student model architectures for melanoma detection.

Teacher: ResNet-based models (ResNet18/34/50/101/152) and EfficientNet (B0-B7)
Student: MobileNetV3-Small for mobile deployment

All models output single logit for binary classification (BCEWithLogitsLoss).
"""

import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torchvision import models

from src.config import TeacherConfig, StudentConfig

logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    """
    Teacher model for melanoma detection using ResNet or EfficientNet backbone.
    
    Outputs single logit for binary classification.
    Uses global average pooling before final classifier.
    """
    
    RESNET_ARCHITECTURES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    EFFICIENTNET_ARCHITECTURES = [f"efficientnet_b{i}" for i in range(8)]  # B0-B7
    SUPPORTED_ARCHITECTURES = RESNET_ARCHITECTURES + EFFICIENTNET_ARCHITECTURES
    
    def __init__(self, config: Optional[TeacherConfig] = None):
        super().__init__()
        self.config = config or TeacherConfig()
        
        if self.config.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {self.config.architecture} not supported. "
                f"Choose from {self.SUPPORTED_ARCHITECTURES}"
            )
        
        self._is_efficientnet = self.config.architecture.startswith("efficientnet")
        
        # Load pretrained backbone
        self.backbone = self._load_backbone()
        
        # Replace classifier head based on architecture type
        if self._is_efficientnet:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=self.config.dropout),
                nn.Linear(in_features, self.config.num_classes),
            )
        else:  # ResNet
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=self.config.dropout),
                nn.Linear(in_features, self.config.num_classes),
            )
        
        logger.info(
            f"Created TeacherModel: {self.config.architecture}, "
            f"pretrained={self.config.pretrained}, "
            f"params={self.count_parameters()['total']:,}"
        )
    
    def _load_backbone(self) -> nn.Module:
        """Load pretrained ResNet or EfficientNet backbone."""
        arch = self.config.architecture
        
        if self._is_efficientnet:
            # EfficientNet weights naming convention
            weights_name = f"EfficientNet_{arch.replace('efficientnet_', '').upper()}_Weights"
            if self.config.pretrained:
                weights = getattr(models, weights_name, None)
                if weights is not None:
                    weights = weights.DEFAULT
            else:
                weights = None
            model = getattr(models, arch)(weights=weights)
        else:
            # ResNet weights naming convention
            if self.config.pretrained:
                weights = getattr(models, f"{arch.capitalize()}_Weights", None)
                if weights is not None:
                    weights = weights.DEFAULT
            else:
                weights = None
            model = getattr(models, arch)(weights=weights)
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (N,) for binary classification."""
        return self.backbone(x).squeeze(-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings before final classifier."""
        if self._is_efficientnet:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        else:  # ResNet
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        return x
    
    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def freeze_backbone(self, unfreeze_layers: Optional[list] = None) -> None:
        """
        Freeze backbone layers, optionally unfreezing specified layers.
        
        Args:
            unfreeze_layers: List of layer names to keep trainable
                           ResNet: e.g., ["layer3", "layer4", "fc"]
                           EfficientNet: e.g., ["features.7", "features.8", "classifier"]
        """
        unfreeze_layers = unfreeze_layers or self.config.unfreeze_layers
        
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze specified layers
        for name, module in self.backbone.named_children():
            if name in unfreeze_layers:
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze layer: {name}")
        
        # Also handle nested layer names like "features.7"
        for layer_name in unfreeze_layers:
            if "." in layer_name:
                parts = layer_name.split(".")
                module = self.backbone
                try:
                    for part in parts:
                        module = getattr(module, part) if not part.isdigit() else module[int(part)]
                    for param in module.parameters():
                        param.requires_grad = True
                    logger.info(f"Unfroze layer: {layer_name}")
                except (AttributeError, IndexError):
                    logger.warning(f"Layer {layer_name} not found")
        
        params = self.count_parameters()
        logger.info(f"After freezing: {params['trainable']:,} trainable / {params['total']:,} total")


class StudentModel(nn.Module):
    """
    Student model for mobile deployment using MobileNetV3.
    
    Designed to meet deployment constraints:
    - Model size < 25 MB for mobile
    - Model size < 2 MB for edge (with quantization)
    """
    
    SUPPORTED_ARCHITECTURES = ["mobilenet_v3_small", "mobilenet_v3_large"]
    
    def __init__(self, config: Optional[StudentConfig] = None):
        super().__init__()
        self.config = config or StudentConfig()
        
        if self.config.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {self.config.architecture} not supported. "
                f"Choose from {self.SUPPORTED_ARCHITECTURES}"
            )
        
        # Load pretrained backbone
        self.backbone = self._load_backbone()
        
        # Get classifier input features (from the first linear layer after avgpool)
        # MobileNetV3 classifier is: Linear -> Hardswish -> Dropout -> Linear
        # We need the input features of the FIRST linear layer
        in_features = self.backbone.classifier[0].in_features
        
        # Replace classifier with custom head for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(1024, self.config.num_classes),
        )
        
        logger.info(
            f"Created StudentModel: {self.config.architecture}, "
            f"pretrained={self.config.pretrained}, "
            f"params={self.count_parameters()['total']:,}, "
            f"size={self.get_model_size_mb():.2f} MB"
        )
    
    def _load_backbone(self) -> nn.Module:
        """Load pretrained MobileNetV3 backbone."""
        arch = self.config.architecture
        
        if self.config.pretrained:
            weights_name = "MobileNet_V3_Small_Weights" if "small" in arch else "MobileNet_V3_Large_Weights"
            weights = getattr(models, weights_name, None)
            if weights is not None:
                weights = weights.DEFAULT
        else:
            weights = None
        
        model = getattr(models, arch)(weights=weights)
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (N,) for binary classification."""
        return self.backbone(x).squeeze(-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings before final classifier."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def get_model_size_mb(self) -> float:
        """Calculate model size in megabytes (FP32)."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_bytes + buffer_bytes) / (1024 ** 2)
    
    def check_deployment_constraints(self) -> Dict[str, Any]:
        """Check if model meets deployment size constraints."""
        size_mb = self.get_model_size_mb()
        return {
            "size_mb": size_mb,
            "meets_mobile_constraint": size_mb <= self.config.max_size_mb,
            "meets_edge_constraint": size_mb <= self.config.max_edge_size_mb,
            "mobile_target_mb": self.config.max_size_mb,
            "edge_target_mb": self.config.max_edge_size_mb,
        }


def create_teacher(config: Optional[TeacherConfig] = None) -> TeacherModel:
    """Factory function to create teacher model."""
    return TeacherModel(config)


def create_student(config: Optional[StudentConfig] = None) -> StudentModel:
    """Factory function to create student model."""
    return StudentModel(config)


def load_teacher_checkpoint(
    checkpoint_path: str,
    config: Optional[TeacherConfig] = None,
    device: str = "cpu",
) -> Tuple[TeacherModel, Dict[str, Any]]:
    """
    Load teacher model from checkpoint.
    
    Returns:
        Tuple of (model, metadata)
    """
    config = config or TeacherConfig()
    model = TeacherModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        metadata = {k: v for k, v in checkpoint.items() if k not in ["state_dict", "model_state_dict"]}
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded teacher from {checkpoint_path}")
    return model, metadata


def load_student_checkpoint(
    checkpoint_path: str,
    config: Optional[StudentConfig] = None,
    device: str = "cpu",
) -> Tuple[StudentModel, Dict[str, Any]]:
    """
    Load student model from checkpoint.
    
    Returns:
        Tuple of (model, metadata)
    """
    config = config or StudentConfig()
    model = StudentModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        metadata = {k: v for k, v in checkpoint.items() if k not in ["state_dict", "model_state_dict"]}
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded student from {checkpoint_path}")
    return model, metadata
