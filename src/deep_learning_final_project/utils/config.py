"""
Configuration management utilities.

This module provides functions for loading and managing experiment configurations,
ensuring reproducibility across research runs.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid

    Example:
        >>> config = load_config("configs/experiments/baseline.yaml")
        >>> print(config["learning_rate"])
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], output_path: str | Path) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration

    Example:
        >>> config = {"learning_rate": 0.001, "batch_size": 32}
        >>> save_config(config, "results/experiment_config.yaml")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base values

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {"lr": 0.001, "epochs": 100}
        >>> override = {"lr": 0.01}
        >>> merged = merge_configs(base, override)
        >>> print(merged)  # {"lr": 0.01, "epochs": 100}
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_config_from_env(prefix: str = "DL_") -> Dict[str, str]:
    """
    Extract configuration from environment variables with a given prefix.

    Args:
        prefix: Prefix for environment variables to extract

    Returns:
        Dictionary of configuration values from environment

    Example:
        >>> # If DL_BATCH_SIZE=32 and DL_LR=0.001 are set
        >>> config = get_config_from_env("DL_")
        >>> print(config)  # {"BATCH_SIZE": "32", "LR": "0.001"}
    """
    config = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):]
            config[config_key] = value
    return config


class Config:
    """
    Configuration class for managing experiment parameters.

    This class provides attribute-style access to configuration parameters
    and supports nested configurations.

    Example:
        >>> config = Config({"model": {"hidden_size": 256}, "lr": 0.001})
        >>> print(config.lr)  # 0.001
        >>> print(config.model.hidden_size)  # 256
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize Config from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    @classmethod
    def from_file(cls, config_path: str | Path) -> "Config":
        """
        Create Config from a YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.from_file("configs/experiments/exp1.yaml")
        """
        config_dict = load_config(config_path)
        return cls(config_dict)
