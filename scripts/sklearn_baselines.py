"""Sklearn baseline benchmarks for melanoma classification.

These traditional ML baselines help contextualize deep learning performance.
Uses HOG features, color histograms, or flattened pixels as features.

Includes dermoscopy-specific preprocessing and feature engineering:
- Hair removal (black-hat morphology)
- Contrast enhancement (CLAHE)
- Color space features (LAB, HSV)
- ABCD rule features (Asymmetry, Border, Color, Dermoscopic structures)
- GLCM texture features
- Shape descriptors

Usage:
    python scripts/sklearn_baselines.py
    python scripts/sklearn_baselines.py --features hog --model logistic_regression
    python scripts/sklearn_baselines.py --features combined --model all
    python scripts/sklearn_baselines.py --features dermoscopy --model random_forest
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, closing, black_tophat
from skimage.measure import regionprops, label
from skimage.color import rgb2lab, rgb2hsv
from skimage.exposure import equalize_adapthist
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.config import (
    LOGS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TABLES_DIR,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "sklearn_baselines.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Image Preprocessing (Dermoscopy-specific)
# ============================================================================

def remove_hair(image: np.ndarray, kernel_size: int = 17) -> np.ndarray:
    """Remove hair artifacts using black-hat morphological filtering.
    
    Hair appears as thin dark structures. Black-hat transform highlights
    dark structures smaller than the structuring element.
    
    Args:
        image: RGB or grayscale image
        kernel_size: Size of morphological kernel (should be larger than hair width)
        
    Returns:
        Image with hair artifacts reduced
    """
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image.copy()
    
    # Black-hat transform to detect hair
    kernel = disk(kernel_size)
    hair_mask = black_tophat(gray, kernel)
    
    # Threshold to get binary hair mask
    thresh = threshold_otsu(hair_mask) if hair_mask.max() > 0 else 0
    hair_binary = hair_mask > max(thresh, 10)
    
    # Inpaint by replacing hair pixels with local median
    result = image.copy()
    if len(image.shape) == 3:
        for c in range(3):
            channel = result[:, :, c].astype(float)
            # Use median filter on hair regions
            median_filtered = ndimage.median_filter(channel, size=5)
            channel[hair_binary] = median_filtered[hair_binary]
            result[:, :, c] = channel.astype(np.uint8)
    else:
        median_filtered = ndimage.median_filter(result.astype(float), size=5)
        result[hair_binary] = median_filtered[hair_binary]
    
    return result


def enhance_contrast(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Improves local contrast while limiting noise amplification.
    Particularly useful for dermoscopic images with varying illumination.
    
    Args:
        image: RGB image (values 0-255)
        clip_limit: Clipping limit for CLAHE
        
    Returns:
        Contrast-enhanced image
    """
    # Normalize to 0-1 range for skimage
    img_float = image.astype(np.float64) / 255.0
    
    # Apply CLAHE to each channel
    enhanced = np.zeros_like(img_float)
    for c in range(3):
        enhanced[:, :, c] = equalize_adapthist(img_float[:, :, c], clip_limit=clip_limit)
    
    return (enhanced * 255).astype(np.uint8)


def segment_lesion(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Segment the lesion from background using Otsu thresholding.
    
    Returns binary mask and the largest connected component properties.
    
    Args:
        image: RGB image
        
    Returns:
        Tuple of (binary mask, regionprops of largest region or None)
    """
    # Convert to grayscale
    gray = np.mean(image, axis=2).astype(np.uint8)
    
    # Otsu thresholding (lesions are typically darker)
    thresh = threshold_otsu(gray)
    binary = gray < thresh
    
    # Clean up with morphological operations
    binary = opening(binary, disk(3))
    binary = closing(binary, disk(5))
    
    # Get largest connected component
    labeled = label(binary)
    regions = regionprops(labeled)
    
    if not regions:
        return binary, None
    
    # Find largest region
    largest = max(regions, key=lambda r: r.area)
    
    # Create mask with only largest component
    mask = labeled == largest.label
    
    return mask, largest


def preprocess_image(
    image: np.ndarray,
    remove_hair_artifacts: bool = True,
    enhance: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for dermoscopic images.
    
    Args:
        image: RGB image
        remove_hair_artifacts: Whether to apply hair removal
        enhance: Whether to apply contrast enhancement
        
    Returns:
        Preprocessed image
    """
    result = image.copy()
    
    if remove_hair_artifacts:
        result = remove_hair(result)
    
    if enhance:
        result = enhance_contrast(result)
    
    return result


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_color_histogram(
    image: np.ndarray,
    bins: int = 32,
) -> np.ndarray:
    """Extract color histogram features from RGB image.
    
    Args:
        image: RGB image array (H, W, 3)
        bins: Number of bins per channel
        
    Returns:
        Feature vector of length bins * 3
    """
    features = []
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)  # Normalize
        features.append(hist)
    return np.concatenate(features)


def extract_color_statistics(image: np.ndarray) -> np.ndarray:
    """Extract statistical color features from multiple color spaces.
    
    Uses RGB, LAB, and HSV color spaces for more robust color description.
    LAB is particularly useful as it separates luminance from color.
    
    Args:
        image: RGB image (0-255)
        
    Returns:
        Feature vector with color statistics
    """
    features = []
    
    # RGB statistics
    for c in range(3):
        channel = image[:, :, c].flatten()
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
            skew(channel),
            kurtosis(channel),
        ])
    
    # Convert to LAB (better for perceptual color differences)
    img_float = image.astype(np.float64) / 255.0
    lab = rgb2lab(img_float)
    
    for c in range(3):
        channel = lab[:, :, c].flatten()
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
        ])
    
    # Convert to HSV (good for color segmentation)
    hsv = rgb2hsv(img_float)
    
    for c in range(3):
        channel = hsv[:, :, c].flatten()
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
        ])
    
    return np.array(features)


def extract_glcm_features(
    image: np.ndarray,
    distances: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
) -> np.ndarray:
    """Extract GLCM (Gray-Level Co-occurrence Matrix) texture features.
    
    GLCM captures spatial relationships between pixel intensities,
    useful for characterizing texture patterns in lesions.
    
    Args:
        image: Grayscale image (0-255)
        distances: Pixel distances to consider
        angles: Angles to consider (in radians)
        
    Returns:
        Feature vector with GLCM properties
    """
    if distances is None:
        distances = [1, 3, 5]
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Quantize to fewer gray levels for efficiency
    quantized = (image / 16).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=16,
        symmetric=True,
        normed=True,
    )
    
    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    
    for prop in properties:
        prop_values = graycoprops(glcm, prop)
        features.extend([
            prop_values.mean(),
            prop_values.std(),
            prop_values.min(),
            prop_values.max(),
        ])
    
    return np.array(features)


def extract_shape_features(mask: np.ndarray, region) -> np.ndarray:
    """Extract shape descriptors from lesion segmentation.
    
    Implements features related to the ABCD rule:
    - Asymmetry
    - Border irregularity
    - Compactness
    
    Args:
        mask: Binary lesion mask
        region: regionprops object for the lesion
        
    Returns:
        Feature vector with shape descriptors
    """
    if region is None:
        return np.zeros(12)
    
    features = []
    
    # Basic shape metrics
    features.append(region.area)
    features.append(region.perimeter)
    features.append(region.equivalent_diameter)
    features.append(region.major_axis_length)
    features.append(region.minor_axis_length)
    
    # Compactness (circularity) - melanomas tend to be less circular
    compactness = (4 * np.pi * region.area) / (region.perimeter ** 2 + 1e-8)
    features.append(compactness)
    
    # Eccentricity - how elongated
    features.append(region.eccentricity)
    
    # Solidity - ratio of area to convex hull area (border irregularity)
    features.append(region.solidity)
    
    # Extent - ratio of area to bounding box area
    features.append(region.extent)
    
    # Asymmetry - compare lesion to its flipped versions
    # Horizontal asymmetry
    flipped_h = np.fliplr(mask)
    h_asym = np.sum(mask != flipped_h) / (np.sum(mask) + 1e-8)
    features.append(h_asym)
    
    # Vertical asymmetry
    flipped_v = np.flipud(mask)
    v_asym = np.sum(mask != flipped_v) / (np.sum(mask) + 1e-8)
    features.append(v_asym)
    
    # Perimeter irregularity (ratio of perimeter to convex hull perimeter)
    features.append(region.perimeter / (region.convex_area ** 0.5 * 4 + 1e-8))
    
    return np.array(features)


def extract_hog_features(
    image: np.ndarray,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9,
) -> np.ndarray:
    """Extract HOG (Histogram of Oriented Gradients) features.
    
    Args:
        image: Grayscale image array (H, W)
        pixels_per_cell: Size of a cell
        cells_per_block: Number of cells per block
        orientations: Number of gradient orientations
        
    Returns:
        HOG feature vector
    """
    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """Extract simple texture features using local binary patterns approximation.
    
    Args:
        image: Grayscale image array (H, W)
        
    Returns:
        Texture feature vector
    """
    # Simple texture: local variance in patches
    patch_size = 16
    h, w = image.shape
    features = []
    
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            features.extend([
                np.mean(patch),
                np.std(patch),
                np.percentile(patch, 25),
                np.percentile(patch, 75),
            ])
    
    return np.array(features[:256])  # Limit to fixed size


def extract_dermoscopy_features(
    image: np.ndarray,
    include_shape: bool = True,
) -> np.ndarray:
    """Extract dermoscopy-specific features based on ABCD rule.
    
    Combines:
    - Color statistics in multiple color spaces
    - GLCM texture features
    - Shape descriptors (asymmetry, border, compactness)
    
    Args:
        image: RGB image
        include_shape: Whether to include shape features (requires segmentation)
        
    Returns:
        Feature vector
    """
    features = []
    
    # Color statistics (RGB + LAB + HSV)
    color_stats = extract_color_statistics(image)
    features.append(color_stats)
    
    # GLCM texture features
    gray = np.mean(image, axis=2).astype(np.uint8)
    glcm_feats = extract_glcm_features(gray)
    features.append(glcm_feats)
    
    # Color histogram (for color variation - "C" in ABCD)
    color_hist = extract_color_histogram(image, bins=16)
    features.append(color_hist)
    
    # Shape features (requires segmentation)
    if include_shape:
        mask, region = segment_lesion(image)
        shape_feats = extract_shape_features(mask, region)
        features.append(shape_feats)
    
    return np.concatenate(features)


def extract_features(
    image_path: pathlib.Path,
    feature_type: str = "combined",
    target_size: Tuple[int, int] = (128, 128),
    preprocess: bool = True,
) -> np.ndarray:
    """Extract features from a single image.
    
    Args:
        image_path: Path to image file
        feature_type: One of 'color', 'hog', 'texture', 'pixels', 'combined', 
                     'dermoscopy', 'dermoscopy_full'
        target_size: Resize image to this size
        preprocess: Whether to apply preprocessing (hair removal, contrast)
        
    Returns:
        Feature vector
    """
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Apply preprocessing for dermoscopy images
    if preprocess and feature_type in ["dermoscopy", "dermoscopy_full", "combined"]:
        img_array = preprocess_image(img_array, remove_hair_artifacts=True, enhance=True)
    
    # Grayscale for HOG/texture
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    
    features = []
    
    if feature_type in ["color", "combined"]:
        color_feats = extract_color_histogram(img_array)
        features.append(color_feats)
    
    if feature_type in ["hog", "combined"]:
        hog_feats = extract_hog_features(gray)
        features.append(hog_feats)
    
    if feature_type in ["texture", "combined"]:
        texture_feats = extract_texture_features(gray)
        features.append(texture_feats)
    
    if feature_type == "pixels":
        # Flattened pixels (downsampled)
        small = np.array(img.resize((32, 32), Image.Resampling.LANCZOS))
        features.append(small.flatten().astype(np.float32) / 255.0)
    
    if feature_type == "dermoscopy":
        # Dermoscopy features without shape (faster)
        derm_feats = extract_dermoscopy_features(img_array, include_shape=False)
        features.append(derm_feats)
    
    if feature_type == "dermoscopy_full":
        # Full dermoscopy features including shape analysis
        derm_feats = extract_dermoscopy_features(img_array, include_shape=True)
        features.append(derm_feats)
    
    return np.concatenate(features) if features else np.array([])


def load_features_and_labels(
    csv_path: pathlib.Path,
    feature_type: str = "combined",
    max_samples: Optional[int] = None,
    preprocess: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images and extract features.
    
    Args:
        csv_path: Path to data CSV (train_data.csv, val_data.csv, etc.)
        feature_type: Type of features to extract
        max_samples: Limit number of samples (for debugging)
        preprocess: Whether to apply preprocessing
        
    Returns:
        Tuple of (features array, labels array)
    """
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    # Find image directory
    image_dirs = [
        RAW_DIR / "HAM10000_images_part_1",
        RAW_DIR / "HAM10000_images_part_2", 
        RAW_DIR / "images",
    ]
    
    features_list = []
    labels_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {feature_type} features"):
        image_id = row["image_id"]
        target_label = row["target"] if "target" in row else row["dx_binary"]
        
        # Find image file
        image_path = None
        for img_dir in image_dirs:
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = img_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path:
                break
        
        if image_path is None:
            logger.warning("Image not found: %s", image_id)
            continue
        
        try:
            feats = extract_features(image_path, feature_type, preprocess=preprocess)
            features_list.append(feats)
            labels_list.append(target_label)
        except Exception as e:
            logger.warning("Error processing %s: %s", image_id, e)
    
    return np.array(features_list), np.array(labels_list)


# ============================================================================
# Models
# ============================================================================

def get_sklearn_models() -> Dict[str, object]:
    """Get dictionary of sklearn classifiers to benchmark."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    }


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),  # Sensitivity
        "specificity": recall_score(y_test, y_pred, pos_label=0),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run sklearn baseline benchmarks")
    parser.add_argument(
        "--features",
        type=str,
        default="combined",
        choices=["color", "hog", "texture", "pixels", "combined", 
                 "dermoscopy", "dermoscopy_full"],
        help="Feature extraction method. 'dermoscopy' uses ABCD-inspired features, "
             "'dermoscopy_full' adds shape analysis (slower).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["logistic_regression", "random_forest", "gradient_boosting", "svm_rbf", "all"],
        help="Model to train",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocessing (hair removal, contrast enhancement)",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Apply PCA dimensionality reduction",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for debugging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    preprocess = not args.no_preprocess
    
    print("=" * 70)
    print("SKLEARN BASELINE BENCHMARKS")
    print("=" * 70)
    print(f"Features: {args.features}")
    print(f"Model(s): {args.model}")
    print(f"Preprocessing: {'enabled (hair removal + contrast)' if preprocess else 'disabled'}")
    print(f"PCA: {args.use_pca} ({args.pca_components} components)" if args.use_pca else "PCA: disabled")
    print()
    
    # Load data
    train_path = PROCESSED_DIR / "train_data.csv"
    val_path = PROCESSED_DIR / "val_data.csv"
    holdout_path = PROCESSED_DIR / "holdout_data.csv"
    
    for path in [train_path, val_path, holdout_path]:
        if not path.exists():
            logger.error("Data file not found: %s", path)
            logger.error("Run 'make splits' first to create data splits.")
            sys.exit(1)
    
    print(">>> Loading and extracting features...")
    start_time = time.time()
    
    X_train, y_train = load_features_and_labels(train_path, args.features, args.max_samples, preprocess)
    X_val, y_val = load_features_and_labels(val_path, args.features, args.max_samples, preprocess)
    X_holdout, y_holdout = load_features_and_labels(holdout_path, args.features, args.max_samples, preprocess)
    
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Holdout: {X_holdout.shape}")
    print(f"Feature extraction time: {time.time() - start_time:.1f}s")
    print(f"Class balance (train): {y_train.mean():.1%} positive (melanoma)")
    
    # Standardize features
    print("\n>>> Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_holdout = scaler.transform(X_holdout)
    
    # Optional PCA
    if args.use_pca:
        print(f">>> Applying PCA ({args.pca_components} components)...")
        pca = PCA(n_components=min(args.pca_components, X_train.shape[1]))
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_holdout = pca.transform(X_holdout)
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    
    # Get models to train
    all_models = get_sklearn_models()
    if args.model == "all":
        models_to_train = all_models
    else:
        models_to_train = {args.model: all_models[args.model]}
    
    # Train and evaluate
    results = []
    
    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION")
    print("=" * 70)
    
    for model_name, model in models_to_train.items():
        print(f"\n>>> Training {model_name}...")
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Evaluate on validation
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # Evaluate on holdout
        holdout_metrics = evaluate_model(model, X_holdout, y_holdout)
        
        print(f"   Train time: {train_time:.1f}s")
        print(f"   Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"   Holdout ROC-AUC: {holdout_metrics['roc_auc']:.4f}")
        print(f"   Holdout PR-AUC: {holdout_metrics['pr_auc']:.4f}")
        print(f"   Holdout F1: {holdout_metrics['f1']:.4f}")
        print(f"   Holdout Sensitivity: {holdout_metrics['recall']:.4f}")
        print(f"   Holdout Specificity: {holdout_metrics['specificity']:.4f}")
        
        results.append({
            "model": model_name,
            "features": args.features,
            "preprocessing": preprocess,
            "use_pca": args.use_pca,
            "n_features": X_train.shape[1],
            "train_time_s": train_time,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "holdout_roc_auc": holdout_metrics["roc_auc"],
            "holdout_pr_auc": holdout_metrics["pr_auc"],
            "holdout_f1": holdout_metrics["f1"],
            "holdout_precision": holdout_metrics["precision"],
            "holdout_recall": holdout_metrics["recall"],
            "holdout_specificity": holdout_metrics["specificity"],
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    df = df.sort_values("holdout_roc_auc", ascending=False)
    
    print("\nResults (sorted by holdout ROC-AUC):")
    print(df[["model", "features", "holdout_roc_auc", "holdout_pr_auc", "holdout_f1"]].to_string(index=False))
    
    # Save results
    output_path = args.output or (TABLES_DIR / "sklearn_baselines.csv")
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to existing results if file exists
    if output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["model", "features", "preprocessing", "use_pca"], keep="last")
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Also save as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")
    
    print("\n" + "=" * 70)
    print("COMPARISON NOTES")
    print("=" * 70)
    print("""
FEATURE OPTIONS:
  - color:           RGB color histograms (96 features)
  - hog:             Histogram of Oriented Gradients (edge patterns)
  - texture:         Local statistics (mean, std, quartiles)
  - pixels:          Flattened pixel values (1024 features)
  - combined:        color + hog + texture (default)
  - dermoscopy:      ABCD-inspired features (color stats, GLCM, histograms)
  - dermoscopy_full: dermoscopy + shape features (asymmetry, border, etc.)

PREPROCESSING (enabled by default):
  - Hair removal via black-hat morphology
  - Contrast enhancement via CLAHE

TIPS FOR BETTER RESULTS:
  1. Use 'dermoscopy' or 'dermoscopy_full' features for best baseline performance
  2. Random Forest typically works best with hand-crafted features
  3. Try --use-pca with SVM to reduce overfitting
  4. The gap to deep learning shows the value of learned representations
""")


if __name__ == "__main__":
    main()
