import os
import dotenv

import torch
import pytorch.nn as nn
from sklearn.model_selection import train_test_split

# set a global random seed for reproducibility

dotenv.load_dotenv()
torch.manual_seed(int(os.getenv("RANDOM_SEED", 42)))

def split_dataset(df_raw, val_size=0.2, random_seed=int(os.getenv("RANDOM_SEED", 42))):
    """
    Splits a dataset into training and validation sets.
    
    Args:
        dataset: The dataset to split.
        val_size: Proportion of the dataset to include in the validation split.
        random_seed: Random seed for reproducibility."""

    train_df, val_df = train_test_split(
        df_raw,
        test_size=val_size,
        random_state=random_seed,
        stratify=df_raw['label']  # assuming 'label' is the target column
    )
    return train_df, val_df


