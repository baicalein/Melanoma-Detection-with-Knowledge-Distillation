import os
import pathlib

import dotenv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import RAW_DIR, PROCESSED_DIR, RANDOM_SEED

# set a global random seed for reproducibility

dotenv.load_dotenv()
torch.manual_seed(int(os.getenv("RANDOM_SEED", RANDOM_SEED)))

OUT_HAM_DIR = PROCESSED_DIR

def split_dataset(df_raw, train_size=0.7, test_size=0.15, holdout_size=0.15, random_seed=int(os.getenv("RANDOM_SEED", 42))):
    """
    Splits a dataset into training, test, and holdout sets.

    Args:
        df_raw: The dataframe to split.
        train_size: Proportion for the training set (default 0.7).
        test_size: Proportion for the test set (default 0.15).
        holdout_size: Proportion for the holdout set (default 0.15).
        random_seed: Random seed for reproducibility.

    Returns:
        train_df, test_df, holdout_df
    """
    # basic validation
    total = float(train_size) + float(test_size) + float(holdout_size)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/test/holdout sizes must sum to 1.0")

    # first split off training portion
    train_df, remaining = train_test_split(
        df_raw, test_size=(1.0 - train_size), random_state=random_seed, stratify=df_raw["target"]
    )
    # split remaining into test and holdout (proportionally)
    # fraction of remaining to assign to test = test_size / (test_size + holdout_size)
    rel_test_fraction = test_size / (test_size + holdout_size)
    test_df, holdout_df = train_test_split(
        remaining, test_size=(1.0 - rel_test_fraction), random_state=random_seed, stratify=remaining["target"]
    )

    # save to processed folder
    train_df.to_csv(OUT_HAM_DIR / "train_data.csv", index=False)
    test_df.to_csv(OUT_HAM_DIR / "test_data.csv", index=False)
    holdout_df.to_csv(OUT_HAM_DIR / "holdout_data.csv", index=False)
    print(f"saving training data to {OUT_HAM_DIR / 'train_data.csv'}")
    print(f"saving test data to {OUT_HAM_DIR / 'test_data.csv'}")
    print(f"saving holdout data to {OUT_HAM_DIR / 'holdout_data.csv'}")
    return train_df, test_df, holdout_df



def build_data_loaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size=os.getenv("BATCH_SIZE", 32),
    num_workers=4,
):
    """
    Builds training and validation data loaders.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
    Returns:
        train_loader: DataLoader for the training dataset."""

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def create_base_df(base_dir: pathlib.Path = None):
    """
    Creates the base dataframe by loading metadata and mapping image paths and lesion types.

    Args:
        base_dir: Base directory containing the skin images and metadata.
                  Defaults to RAW_DIR from config.
    Returns:
        df: The base dataframe with image paths and lesion types.
    """
    if base_dir is None:
        base_dir = RAW_DIR
    base_dir = pathlib.Path(base_dir)

    df = pd.read_csv(base_dir / "HAM10000_metadata.csv")

    lession_type_dict = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    df["lesion_type"] = df["dx"].map(lession_type_dict)
    df["image_path"] = df["image_id"].apply(lambda x: str(base_dir / "images" / f"{x}.jpg"))
    df["target"] = np.where(df["lesion_type"] == "Melanoma", 1, 0)
    df["cell_type_idx"] = pd.Categorical(df["lesion_type"]).codes
    df.to_csv(OUT_HAM_DIR / "labeled_ham10000.csv", index=False)
    print(f"saving output to {OUT_HAM_DIR / 'labeled_ham10000.csv'}")
    return df




if __name__ == "__main__":
    # Example usage
    df = create_base_df()
    train_df, test_df, holdout_df = split_dataset(df)

    print("Training set size:", len(train_df))
    print("Test set size:", len(test_df))
    print("Holdout set size:", len(holdout_df))