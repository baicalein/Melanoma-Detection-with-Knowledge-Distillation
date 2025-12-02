"""
Lesion-aware stratified data splitting for HAM10000.

CRITICAL: HAM10000 contains multiple images per lesion. Images from the same
lesion must NOT appear in different splits to avoid data leakage and overly
optimistic performance estimates.

This module provides:
- Lesion-level stratified splitting
- Proper train/val/holdout partitioning
- Split verification and statistics
"""

import logging
import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DataConfig, PROCESSED_DIR, RAW_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


def create_lesion_level_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
    lesion_col: str = "lesion_id",
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/holdout splits at the LESION level, not image level.
    
    This is critical for HAM10000 which has multiple images per lesion.
    Splitting at image level would cause data leakage.
    
    Args:
        df: Full dataset DataFrame with lesion_id and target columns
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        holdout_ratio: Fraction for holdout/test (default 0.15)
        random_seed: Random seed for reproducibility
        lesion_col: Column name for lesion identifier
        target_col: Column name for binary target
    
    Returns:
        Tuple of (train_df, val_df, holdout_df)
    
    Raises:
        ValueError: If ratios don't sum to 1.0 or required columns missing
    """
    # Validate ratios
    total = train_ratio + val_ratio + holdout_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Validate columns
    if lesion_col not in df.columns:
        raise ValueError(f"Lesion column '{lesion_col}' not found. Available: {df.columns.tolist()}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Create lesion-level DataFrame
    # Each lesion gets its majority target label for stratification
    lesion_df = df.groupby(lesion_col).agg({
        target_col: lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        "image_id": "count"  # Count images per lesion
    }).reset_index()
    lesion_df.columns = [lesion_col, target_col, "n_images"]
    
    logger.info(f"Total lesions: {len(lesion_df)}")
    logger.info(f"Melanoma lesions: {(lesion_df[target_col] == 1).sum()}")
    logger.info(f"Non-melanoma lesions: {(lesion_df[target_col] == 0).sum()}")
    
    # First split: train vs (val + holdout)
    train_lesions, remaining_lesions = train_test_split(
        lesion_df[lesion_col].values,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        stratify=lesion_df[target_col].values,
    )
    
    # Second split: val vs holdout (relative to remaining)
    remaining_df = lesion_df[lesion_df[lesion_col].isin(remaining_lesions)]
    rel_val_ratio = val_ratio / (val_ratio + holdout_ratio)
    
    val_lesions, holdout_lesions = train_test_split(
        remaining_df[lesion_col].values,
        test_size=(1.0 - rel_val_ratio),
        random_state=random_seed,
        stratify=remaining_df[target_col].values,
    )
    
    # Map lesions back to images
    train_df = df[df[lesion_col].isin(train_lesions)].copy()
    val_df = df[df[lesion_col].isin(val_lesions)].copy()
    holdout_df = df[df[lesion_col].isin(holdout_lesions)].copy()
    
    # Verify no leakage
    train_lesion_set = set(train_df[lesion_col])
    val_lesion_set = set(val_df[lesion_col])
    holdout_lesion_set = set(holdout_df[lesion_col])
    
    assert len(train_lesion_set & val_lesion_set) == 0, "Leakage between train and val!"
    assert len(train_lesion_set & holdout_lesion_set) == 0, "Leakage between train and holdout!"
    assert len(val_lesion_set & holdout_lesion_set) == 0, "Leakage between val and holdout!"
    
    logger.info("=" * 60)
    logger.info("SPLIT STATISTICS (Lesion-Level Stratified)")
    logger.info("=" * 60)
    _log_split_stats(train_df, val_df, holdout_df, target_col)
    
    return train_df, val_df, holdout_df


def _log_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
) -> None:
    """Log detailed statistics about the splits."""
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Holdout", holdout_df)]:
        n = len(df)
        n_pos = (df[target_col] == 1).sum()
        n_neg = (df[target_col] == 0).sum()
        prevalence = n_pos / n if n > 0 else 0
        n_lesions = df["lesion_id"].nunique()
        
        logger.info(f"{name:8s}: N={n:5d} images, {n_lesions:4d} lesions | "
                   f"Melanoma: {n_pos:4d} ({prevalence:.1%}) | "
                   f"Non-melanoma: {n_neg:4d}")


def load_or_create_splits(
    base_csv: Optional[pathlib.Path] = None,
    output_dir: Optional[pathlib.Path] = None,
    config: Optional[DataConfig] = None,
    force_recreate: bool = False,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Load existing splits or create new lesion-aware splits.
    
    Args:
        base_csv: Path to full labeled dataset (default: labeled_ham10000.csv)
        output_dir: Directory to save splits (default: PROCESSED_DIR)
        config: DataConfig with split ratios
        force_recreate: If True, recreate even if files exist
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of paths to (train_csv, val_csv, holdout_csv)
    """
    config = config or DataConfig()
    base_csv = base_csv or (PROCESSED_DIR / "labeled_ham10000.csv")
    output_dir = output_dir or PROCESSED_DIR
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_data.csv"
    val_path = output_dir / "val_data.csv"
    holdout_path = output_dir / "holdout_data.csv"
    
    # Check if splits already exist
    all_exist = train_path.exists() and val_path.exists() and holdout_path.exists()
    
    if all_exist and not force_recreate:
        logger.info("Loading existing splits...")
        # Verify no leakage in existing splits
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        holdout_df = pd.read_csv(holdout_path)
        
        if "lesion_id" in train_df.columns:
            train_lesions = set(train_df["lesion_id"])
            val_lesions = set(val_df["lesion_id"])
            holdout_lesions = set(holdout_df["lesion_id"])
            
            if (train_lesions & val_lesions) or (train_lesions & holdout_lesions) or (val_lesions & holdout_lesions):
                logger.warning("Existing splits have lesion leakage! Recreating...")
                force_recreate = True
        
        if not force_recreate:
            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Holdout: {len(holdout_df)}")
            return train_path, val_path, holdout_path
    
    # Create new splits
    logger.info(f"Creating new lesion-aware splits from {base_csv}")
    df = pd.read_csv(base_csv)
    
    train_df, val_df, holdout_df = create_lesion_level_splits(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        holdout_ratio=config.holdout_ratio,
        random_seed=random_seed,
    )
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    
    logger.info(f"Saved train to {train_path}")
    logger.info(f"Saved val to {val_path}")
    logger.info(f"Saved holdout to {holdout_path}")
    
    # Save split metadata
    metadata = {
        "random_seed": random_seed,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "holdout_ratio": config.holdout_ratio,
        "train_n": len(train_df),
        "val_n": len(val_df),
        "holdout_n": len(holdout_df),
        "train_n_lesions": train_df["lesion_id"].nunique(),
        "val_n_lesions": val_df["lesion_id"].nunique(),
        "holdout_n_lesions": holdout_df["lesion_id"].nunique(),
        "train_melanoma_n": int((train_df["target"] == 1).sum()),
        "val_melanoma_n": int((val_df["target"] == 1).sum()),
        "holdout_melanoma_n": int((holdout_df["target"] == 1).sum()),
    }
    
    import json
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return train_path, val_path, holdout_path


def verify_no_leakage(
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    holdout_path: pathlib.Path,
) -> bool:
    """
    Verify that no lesions appear in multiple splits.
    
    Returns True if no leakage detected.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    holdout_df = pd.read_csv(holdout_path)
    
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    holdout_lesions = set(holdout_df["lesion_id"])
    
    train_val_overlap = train_lesions & val_lesions
    train_holdout_overlap = train_lesions & holdout_lesions
    val_holdout_overlap = val_lesions & holdout_lesions
    
    if train_val_overlap:
        logger.error(f"LEAKAGE: {len(train_val_overlap)} lesions in both train and val")
        return False
    if train_holdout_overlap:
        logger.error(f"LEAKAGE: {len(train_holdout_overlap)} lesions in both train and holdout")
        return False
    if val_holdout_overlap:
        logger.error(f"LEAKAGE: {len(val_holdout_overlap)} lesions in both val and holdout")
        return False
    
    logger.info("✓ No lesion leakage detected across splits")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Recreate splits with proper lesion-level stratification
    load_or_create_splits(force_recreate=True)
