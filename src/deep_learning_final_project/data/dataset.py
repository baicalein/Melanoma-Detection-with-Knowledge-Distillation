"""
Dataset classes for PyTorch data loading.

This module contains custom PyTorch Dataset classes for loading and preprocessing data.
Extend these classes for your specific data needs.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset class for deep learning projects.

    This class provides a template for creating custom datasets. Override the
    __len__ and __getitem__ methods for your specific use case.

    Args:
        data_dir: Path to the data directory
        transform: Optional transform to apply to samples
        target_transform: Optional transform to apply to targets

    Example:
        >>> dataset = BaseDataset(data_dir="data/processed")
        >>> sample, label = dataset[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform

        # TODO: Load your data here
        self.data = []
        self.targets = []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        Get a sample and its target from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (sample, target)
        """
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


class ImageDataset(BaseDataset):
    """
    Dataset for loading images from a directory.

    This is a template for image classification tasks. Customize as needed.

    Args:
        data_dir: Path to the directory containing images
        transform: Transformations to apply to images
        target_transform: Transformations to apply to labels
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(data_dir, transform, target_transform)

        # TODO: Implement image loading logic
        # Example: Load image paths and labels
        self.image_paths = []
        self.labels = []

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return an image and its label."""
        # TODO: Implement image loading
        # Example:
        # image = Image.open(self.image_paths[idx])
        # label = self.labels[idx]
        #
        # if self.transform:
        #     image = self.transform(image)
        #
        # return image, label
        pass
