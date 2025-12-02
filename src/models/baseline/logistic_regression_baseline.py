import argparse
import json
import logging
import os
import pathlib
import time
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROC_DIR / "train_data.csv"
VAL_CSV = PROC_DIR / "val_data.csv"
OUT_DIR_TBLS = ROOT / "artifacts" /"tbls" / "01_baselines" / "logistic_regression"
LOG_PATH = ROOT / "models" / "logs" / "logistic_regression.log"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))


class ImageCSVDataSet(Datasets):
    """Dataset that loads images and labels from a CSV with columns image_path and target."""

    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        target = int(row["target"])
        p = pathlib.Path(img_path)
        if not p.is_absolute():
            p = (PROC_DIR / img_path).resolve()
        if not p.exists():
            logger.error("Image file not found: %s (resolved: %s)", img_path, p)
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_transforms() -> transforms.Compose:
    """Transforms used for feature extraction (deterministic)."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val CSVs; raise if missing."""
    if TRAIN_CSV.exists() and VAL_CSV.exists():
        logger.info("Loading splits: %s and %s", TRAIN_CSV, VAL_CSV)
        return pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)
    raise FileNotFoundError("Train/Val CSVs not found. Run data build pipeline.")


def build_feature_extractor(device: str = "cpu") -> nn.Module:
    """Return a ResNet18 with final fc replaced by identity (frozen)."""
    # use pretrained weights for strong features
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        # fallback if torchvision older
        model = models.resnet18(pretrained=True)
    # replace fc so forward returns feature vector
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_features(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Run images through model and collect features and labels."""
    features = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Extract features", leave=False):
            imgs = imgs.to(device)
            out = model(imgs)  # shape (B, feat_dim)
            out = out.cpu().numpy()
            features.append(out)
            labels.append(labs.numpy())
    X = np.vstack(features)
    y = np.concatenate(labels)
    return X, y


def train_logistic(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train a sklearn pipeline (scaler + logistic) and return it."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Return accuracy and ROC AUC (if available)."""
    y_pred = pipe.predict(X)
    acc = float(accuracy_score(y, y_pred))
    result = {"accuracy": acc}
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)[:, 1]
            result["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None
    return result


def _cache_paths(debug_subset: Optional[int], augmentations: bool = False) -> Tuple[pathlib.Path, pathlib.Path]:
    """Return cache file paths for train and val features."""
    features_dir = OUT_DIR_TBLS / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_dbg{debug_subset}" if debug_subset else ""
    if augmentations:
        suffix += "_aug"
    train_cache = features_dir / f"features_train{suffix}.npz"
    val_cache = features_dir / f"features_val{suffix}.npz"
    return train_cache, val_cache


# Augmentation transforms to try when aggregating features
AUGMENTATION_TRANSFORMS: List[transforms.Compose] = [
    # deterministic center crop
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # random resized crop + horizontal flip
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # center crop with color jitter
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


def extract_features_with_augmentations(
    model: nn.Module,
    df: pd.DataFrame,
    transforms_list: List[transforms.Compose],
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for each augmentation transform and aggregate per image.

    Returns averaged features and labels.
    """
    all_feats: List[np.ndarray] = []
    labels_ref = None
    for i, t in enumerate(transforms_list):
        logger.info("Extracting features for augmentation %d/%d", i + 1, len(transforms_list))
        ds = ImageCSVDataSet(df, transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        X_i, y_i = extract_features(model, loader, device=device)
        all_feats.append(X_i)
        if labels_ref is None:
            labels_ref = y_i
        else:
            if not np.array_equal(labels_ref, y_i):
                logger.warning("Label arrays differ across augmentations; using first occurrence")
    # stack along new axis (n_augs, n_samples, feat_dim)
    stacked = np.stack(all_feats, axis=0)
    # average over augmentations axis
    X_avg = np.mean(stacked, axis=0)
    return X_avg, labels_ref


def train_baseline_and_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int = RANDOM_SEED,
    OUT_DIR_TBLS: pathlib.Path = OUT_DIR_TBLS,
) -> dict:
    """Train baseline logistic, perform GridSearchCV with StratifiedKFold, evaluate and save artifacts.

    Returns a dict with metrics and artifact paths.
    """
    results = {"timestamp": time.time()}
    # Baseline pipeline
    baseline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, random_state=random_seed))])
    baseline.fit(X_train, y_train)
    baseline_train = evaluate_model(baseline, X_train, y_train)
    baseline_val = evaluate_model(baseline, X_val, y_val)

    baseline_path = OUT_DIR_TBLS / "logistic_baseline.joblib"
    joblib.dump({"pipeline": baseline, "metrics_train": baseline_train, "metrics_val": baseline_val}, baseline_path)
    logger.info("Saved baseline pipeline to %s", baseline_path)

    results["baseline"] = {"path": str(baseline_path), "train": baseline_train, "val": baseline_val}

    # Hyperparameter tuning: GridSearchCV
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2", "l1"],
        # use saga for l1 with multinomial / large datasets; liblinear doesn't support multinomial
        "clf__solver": ["liblinear", "saga"],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    pipe_for_search = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=random_seed))])

    gs = GridSearchCV(pipe_for_search, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1, verbose=1)
    logger.info("Starting GridSearchCV for LogisticRegression")
    gs.fit(X_train, y_train)
    logger.info("GridSearchCV complete. Best params: %s", gs.best_params_)

    tuned = gs.best_estimator_
    tuned_train = evaluate_model(tuned, X_train, y_train)
    tuned_val = evaluate_model(tuned, X_val, y_val)

    tuned_path = OUT_DIR_TBLS / "logistic_tuned_gridsearch.joblib"
    joblib.dump({"pipeline": tuned, "best_params": gs.best_params_, "metrics_train": tuned_train, "metrics_val": tuned_val}, tuned_path)
    logger.info("Saved tuned pipeline to %s", tuned_path)

    results["tuned"] = {"path": str(tuned_path), "best_params": gs.best_params_, "train": tuned_train, "val": tuned_val}

    # Also save full cv results for inspection
    try:
        cvres_path = OUT_DIR_TBLS / "gridsearch_cv_results.json"
        # convert numpy types to native
        cv = gs.cv_results_.copy()
        for k, v in cv.items():
            try:
                cv[k] = v.tolist()
            except Exception:
                cv[k] = str(v)
        with open(cvres_path, "w") as fh:
            json.dump(cv, fh)
        logger.info("Saved GridSearch CV results to %s", cvres_path)
        results["grid_cv_results"] = str(cvres_path)
    except Exception as e:
        logger.warning("Failed to save cv results: %s", e)

    return results


def save_npz_features(path: pathlib.Path, X: np.ndarray, y: np.ndarray) -> None:
    """Save features and labels to a compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    logger.info("Saved features to %s", path)


def load_npz_features(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a .npz file."""
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    return X, y


def main(debug_subset: Optional[int] = None, device: Optional[str] = None, batch_size: int = BATCH_SIZE, augmentations: bool = False) -> None:
    """Main routine: extract or load features, train baseline and tuned logistic, evaluate and save artifacts."""
    if device is None:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info("Using device: %s", device)

    train_df, val_df = load_splits()
    logger.info("Train rows: %d, Val rows: %d", len(train_df), len(val_df))

    if debug_subset:
        n = int(debug_subset)
        train_df = train_df.sample(n=min(n, len(train_df)), random_state=RANDOM_SEED).reset_index(drop=True)
        val_df = val_df.sample(n=min(n, len(val_df)), random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info("Using debug subset: Train %d, Val %d", len(train_df), len(val_df))

    feat_model = build_feature_extractor(device=device)

    train_cache, val_cache = _cache_paths(debug_subset, augmentations)

    if train_cache.exists():
        logger.info("Loading cached train features from %s", train_cache)
        X_train, y_train = load_npz_features(train_cache)
    else:
        if augmentations:
            X_train, y_train = extract_features_with_augmentations(feat_model, train_df, AUGMENTATION_TRANSFORMS, device=device, batch_size=batch_size)
        else:
            transform = get_transforms()
            train_ds = ImageCSVDataSet(train_df, transform=transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            X_train, y_train = extract_features(feat_model, train_loader, device=device)
        save_npz_features(train_cache, X_train, y_train)

    if val_cache.exists():
        logger.info("Loading cached val features from %s", val_cache)
        X_val, y_val = load_npz_features(val_cache)
    else:
        if augmentations:
            X_val, y_val = extract_features_with_augmentations(feat_model, val_df, AUGMENTATION_TRANSFORMS, device=device, batch_size=batch_size)
        else:
            transform = get_transforms()
            val_ds = ImageCSVDataSet(val_df, transform=transform)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            X_val, y_val = extract_features(feat_model, val_loader, device=device)
        save_npz_features(val_cache, X_val, y_val)

    logger.info("Features shapes: X_train=%s y_train=%s | X_val=%s y_val=%s", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # Train baseline and tuned models, save artifacts and metrics
    artifacts = train_baseline_and_tune(X_train, y_train, X_val, y_val, random_seed=RANDOM_SEED, OUT_DIR_TBLS=OUT_DIR_TBLS)

    # save summary artifacts
    summary_path = OUT_DIR_TBLS / "artifacts_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(artifacts, fh, indent=2)
    logger.info("Saved artifacts summary to %s", summary_path)

    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression on image features (ResNet18).")
    parser.add_argument("--debug_subset", type=int, default=None, help="Use small subset of examples for quick test")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu/cuda/mps)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--augmentations", action="store_true", help="Use augmentation-based feature extraction and aggregate features")
    args = parser.parse_args()
    main(debug_subset=args.debug_subset, device=args.device, batch_size=args.batch_size, augmentations=args.augmentations)