"""
Example model architecture.

This is a template model to demonstrate the structure. Replace with your actual
model architectures for your research project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class SimpleCNN(BaseModel):
    """
    A simple CNN architecture for image classification.

    This is a template model. Customize it according to your research needs.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        dropout_rate: Dropout probability for regularization

    Example:
        >>> model = SimpleCNN(num_classes=10, in_channels=3)
        >>> x = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([4, 10])
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolution

    Example:
        >>> block = ResidualBlock(64, 64)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> output = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out
