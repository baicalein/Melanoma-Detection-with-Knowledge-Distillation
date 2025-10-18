"""
Data transformation and augmentation utilities.

This module provides common data transformations for deep learning projects,
including normalization, augmentation, and preprocessing pipelines.
"""

from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms as T


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """
    Get training data augmentation pipeline.

    Args:
        image_size: Target size for images (height, width)

    Returns:
        Composed transformations for training data

    Example:
        >>> transform = get_train_transforms((256, 256))
        >>> augmented_image = transform(image)
    """
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """
    Get validation/test data transformation pipeline.

    Args:
        image_size: Target size for images (height, width)

    Returns:
        Composed transformations for validation/test data

    Example:
        >>> transform = get_val_transforms((256, 256))
        >>> processed_image = transform(image)
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def normalize_tensor(
    tensor: torch.Tensor,
    mean: List[float],
    std: List[float]
) -> torch.Tensor:
    """
    Normalize a tensor with given mean and standard deviation.

    Args:
        tensor: Input tensor to normalize
        mean: Mean values for each channel
        std: Standard deviation values for each channel

    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (tensor - mean) / std


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: List[float],
    std: List[float]
) -> torch.Tensor:
    """
    Denormalize a tensor (reverse normalization).

    Useful for visualization of normalized images.

    Args:
        tensor: Normalized input tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


class CustomTransform:
    """
    Template for custom data transformations.

    Implement __call__ method for your specific transformation.

    Example:
        >>> transform = CustomTransform(param=value)
        >>> transformed_data = transform(data)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize transform with parameters."""
        self.params = kwargs

    def __call__(self, data: Any) -> Any:
        """
        Apply transformation to data.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        # TODO: Implement your transformation logic
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
