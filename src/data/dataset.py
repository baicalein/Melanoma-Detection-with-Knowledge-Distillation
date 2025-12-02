"""
HAM10000 Dataset class with proper transforms for dermoscopic image classification.

This module provides:
- HAM10000Dataset: PyTorch Dataset for loading dermoscopic images
- Transform pipelines for training and evaluation
- Dermoscopy-specific augmentations (hair simulation, artifacts, etc.)
"""

import pathlib
from typing import Callable, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.config import DataConfig, PROCESSED_DIR, RAW_DIR


# ============================================================================
# Dermoscopy-Specific Augmentations
# ============================================================================

class GaussianNoise:
    """Add Gaussian noise to simulate sensor noise."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomSharpness:
    """Randomly adjust image sharpness."""
    
    def __init__(self, factor_range: Tuple[float, float] = (0.5, 2.0)):
        self.factor_range = factor_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        factor = np.random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)


class RandomGaussianBlur:
    """Apply random Gaussian blur to simulate focus issues."""
    
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() < self.p:
            radius = np.random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class ElasticTransform:
    """Apply elastic deformation to simulate skin stretching/movement.
    
    Note: This is a simplified version using affine transforms as approximation.
    For full elastic deformation, scipy.ndimage would be needed.
    """
    
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() < self.p:
            # Apply small random affine as approximation of elastic deformation
            angle = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.95, 1.05)
            return TF.affine(img, angle=angle, translate=[0, 0], 
                           scale=scale, shear=[0, 0])
        return img


class MicroscopeArtifacts:
    """Simulate common dermoscopy artifacts like vignetting and bubbles."""
    
    def __init__(self, vignette_p: float = 0.2, bubble_p: float = 0.1):
        self.vignette_p = vignette_p
        self.bubble_p = bubble_p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Vignetting (darker corners)
        if np.random.random() < self.vignette_p:
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_radius = np.sqrt(center_x**2 + center_y**2)
            vignette = 1 - 0.3 * (radius / max_radius) ** 2
            vignette = vignette[:, :, np.newaxis]
            img_array = img_array * vignette
        
        # Random brightness spots (simulate bubbles/reflections)
        if np.random.random() < self.bubble_p:
            num_spots = np.random.randint(1, 4)
            for _ in range(num_spots):
                cx = np.random.randint(w // 4, 3 * w // 4)
                cy = np.random.randint(h // 4, 3 * h // 4)
                radius = np.random.randint(5, 20)
                Y, X = np.ogrid[:h, :w]
                mask = ((X - cx)**2 + (Y - cy)**2) < radius**2
                img_array[mask] = np.minimum(img_array[mask] * 1.3, 255)
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


class GridDistortion:
    """Apply grid-based distortion to simulate lens distortion."""
    
    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.2):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() < self.p:
            # Simplified: use perspective transform as approximation
            width, height = img.size
            distort = self.distort_limit * min(width, height) * 0.1
            
            coeffs = [
                np.random.uniform(-distort, distort) for _ in range(4)
            ]
            
            # Apply small perspective shift
            return TF.perspective(
                img,
                startpoints=[[0, 0], [width, 0], [width, height], [0, height]],
                endpoints=[
                    [coeffs[0], coeffs[1]],
                    [width - coeffs[0], coeffs[1]],
                    [width - coeffs[2], height - coeffs[3]],
                    [coeffs[2], height - coeffs[3]]
                ]
            )
        return img


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dermoscopic Image Dataset.
    
    Loads images and binary melanoma labels from a CSV file.
    Supports configurable transforms for training/evaluation.
    
    Args:
        csv_path: Path to CSV with columns [image_path, target, lesion_id, ...]
        transform: Optional transform to apply to images
        config: DataConfig for preprocessing settings
    """
    
    def __init__(
        self,
        csv_path: pathlib.Path,
        transform: Optional[Callable] = None,
        config: Optional[DataConfig] = None,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.config = config or DataConfig()
        
        # Validate required columns
        required_cols = ["image_path", "target"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Compute class weights for weighted sampling
        self._compute_class_weights()
    
    def _compute_class_weights(self) -> None:
        """Compute inverse frequency class weights for imbalanced data."""
        counts = self.df["target"].value_counts().sort_index()
        total = len(self.df)
        # Inverse frequency weighting
        self.class_weights = torch.tensor(
            [total / (len(counts) * c) for c in counts.values],
            dtype=torch.float32
        )
        # Sample weights for WeightedRandomSampler
        self.sample_weights = torch.tensor(
            [self.class_weights[t] for t in self.df["target"].values],
            dtype=torch.float32
        )
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # Resolve image path
        img_path = pathlib.Path(row["image_path"])
        if not img_path.is_absolute():
            # Try relative to processed dir, then raw dir
            for base in [PROCESSED_DIR, RAW_DIR, pathlib.Path(".")]:
                candidate = (base / img_path).resolve()
                if candidate.exists():
                    img_path = candidate
                    break
        
        # Load and convert image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}") from e
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        target = int(row["target"])
        return image, target
    
    def get_pos_weight(self) -> torch.Tensor:
        """
        Compute positive class weight for BCEWithLogitsLoss.
        
        Returns ratio of negative to positive samples.
        """
        neg_count = (self.df["target"] == 0).sum()
        pos_count = (self.df["target"] == 1).sum()
        return torch.tensor(neg_count / pos_count, dtype=torch.float32)
    
    @property
    def num_samples(self) -> int:
        return len(self.df)
    
    @property
    def num_positive(self) -> int:
        return int((self.df["target"] == 1).sum())
    
    @property
    def num_negative(self) -> int:
        return int((self.df["target"] == 0).sum())
    
    @property
    def prevalence(self) -> float:
        """Melanoma prevalence (positive class ratio)."""
        return self.num_positive / self.num_samples


def get_train_transforms(
    config: Optional[DataConfig] = None,
    augmentation_level: str = "standard",
) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        config: DataConfig for preprocessing settings
        augmentation_level: One of 'light', 'standard', 'heavy', 'dermoscopy'
            - light: Basic flips and rotation only
            - standard: Flips, rotation, color jitter, erasing
            - heavy: All standard + aggressive color/geometric augmentation
            - dermoscopy: Domain-specific augmentations for dermoscopic images
    
    Includes:
    - Random resized crop
    - Horizontal/vertical flips
    - Rotation
    - Color jitter
    - Normalization (ImageNet stats)
    - Random erasing
    - (Optional) Dermoscopy-specific: blur, sharpness, artifacts
    """
    config = config or DataConfig()
    
    # Base transforms (always applied)
    base_transforms = [
        transforms.RandomResizedCrop(
            config.image_size,
            scale=config.random_crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        transforms.RandomVerticalFlip(p=config.vertical_flip_prob),
    ]
    
    # Light augmentation
    if augmentation_level == "light":
        transform_list = base_transforms + [
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
        ]
        return transforms.Compose(transform_list)
    
    # Standard augmentation
    color_jitter = transforms.ColorJitter(
        brightness=config.color_jitter_brightness,
        contrast=config.color_jitter_contrast,
        saturation=config.color_jitter_saturation,
        hue=config.color_jitter_hue,
    )
    
    standard_transforms = base_transforms + [
        transforms.RandomRotation(config.rotation_degrees),
        transforms.RandomApply([color_jitter], p=0.5),
    ]
    
    if augmentation_level == "standard":
        transform_list = standard_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            transforms.RandomErasing(p=config.random_erasing_prob, scale=(0.02, 0.25)),
        ]
        return transforms.Compose(transform_list)
    
    # Heavy augmentation (more aggressive)
    if augmentation_level == "heavy":
        heavy_color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        )
        transform_list = base_transforms + [
            transforms.RandomRotation(45),
            transforms.RandomApply([heavy_color_jitter], p=0.7),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
            GaussianNoise(std=0.03),
        ]
        return transforms.Compose(transform_list)
    
    # Dermoscopy-specific augmentation
    if augmentation_level == "dermoscopy":
        transform_list = [
            # Geometric transforms
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.7, 1.0),  # More aggressive crop for lesion focus
                ratio=(0.9, 1.1),  # Keep roughly square
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),  # Full rotation - lesions have no orientation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=(-5, 5),
            ),
            
            # Dermoscopy-specific augmentations
            RandomSharpness(factor_range=(0.5, 1.5)),
            RandomGaussianBlur(radius_range=(0.5, 1.5), p=0.2),
            MicroscopeArtifacts(vignette_p=0.15, bubble_p=0.05),
            GridDistortion(p=0.15),
            
            # Color augmentation (important for skin tones)
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.25,
                hue=0.08,  # Slightly more hue for skin tone variation
            ),
            
            # To tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            
            # Post-tensor augmentations
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            GaussianNoise(std=0.02),
        ]
        return transforms.Compose(transform_list)
    
    # Default: standard
    transform_list = standard_transforms + [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
        transforms.RandomErasing(p=config.random_erasing_prob, scale=(0.02, 0.25)),
    ]
    return transforms.Compose(transform_list)


def get_eval_transforms(config: Optional[DataConfig] = None) -> transforms.Compose:
    """
    Get evaluation transforms (no augmentation).
    
    Includes:
    - Resize to slightly larger than target
    - Center crop to target size
    - Normalization (ImageNet stats)
    """
    config = config or DataConfig()
    
    # Resize to 256, then center crop to 224 (standard practice)
    resize_size = int(config.image_size * 256 / 224)
    
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std,
        ),
    ])


def get_tta_transforms(config: Optional[DataConfig] = None) -> List[transforms.Compose]:
    """
    Get Test-Time Augmentation (TTA) transforms.
    
    Returns multiple transform pipelines that will be averaged at inference.
    TTA can improve predictions by ~1-2% on dermoscopy tasks.
    
    Returns:
        List of 5 transform pipelines:
        - Original (center crop)
        - Horizontal flip
        - Vertical flip
        - Horizontal + Vertical flip
        - Slightly zoomed in
    """
    config = config or DataConfig()
    resize_size = int(config.image_size * 256 / 224)
    
    base_transforms = [
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.image_size),
    ]
    
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
    ]
    
    tta_list = [
        # Original
        transforms.Compose(base_transforms + normalize),
        
        # Horizontal flip
        transforms.Compose(base_transforms + [transforms.RandomHorizontalFlip(p=1.0)] + normalize),
        
        # Vertical flip
        transforms.Compose(base_transforms + [transforms.RandomVerticalFlip(p=1.0)] + normalize),
        
        # Both flips
        transforms.Compose(base_transforms + [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ] + normalize),
        
        # Slight zoom (crop more)
        transforms.Compose([
            transforms.Resize(int(resize_size * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.image_size),
        ] + normalize),
    ]
    
    return tta_list


def create_dataloaders(
    train_csv: pathlib.Path,
    val_csv: pathlib.Path,
    config: Optional[DataConfig] = None,
    use_weighted_sampling: bool = True,
    augmentation_level: str = "standard",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_csv: Path to training data CSV
        val_csv: Path to validation data CSV
        config: DataConfig for batch size, workers, etc.
        use_weighted_sampling: Use weighted random sampling for class balance
        augmentation_level: One of 'light', 'standard', 'heavy', 'dermoscopy'
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    config = config or DataConfig()
    
    train_dataset = HAM10000Dataset(
        train_csv,
        transform=get_train_transforms(config, augmentation_level=augmentation_level),
        config=config,
    )
    
    val_dataset = HAM10000Dataset(
        val_csv,
        transform=get_eval_transforms(config),
        config=config,
    )
    
    # Optional weighted sampling for training
    train_sampler = None
    shuffle = True
    if use_weighted_sampling:
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,  # Keep workers alive between epochs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    
    return train_loader, val_loader
