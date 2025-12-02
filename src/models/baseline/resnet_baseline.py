import os
import argparse
import json
import logging
import pathlib
import time
from typing import Optional, Tuple, List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROC_DIR / "train_data.csv"
test_CSV = PROC_DIR / "test_data.csv"
OUT_DIR_TBLS = ROOT / "artifacts" / "tbls" / "01_baselines" / "resnet"
OUT_DIR_IMGS = ROOT / "artifacts" / "imgs" / "01_baselines" / "resnet"
OUT_DIR_TBLS.mkdir(parents=True, exist_ok=True)
OUT_DIR_IMGS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "models" / "logs" / "resnet_baseline.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
# use batch size 64 by default as requested
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))


class ImageCSVDataSet(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        p = pathlib.Path(r["image_path"])
        if not p.is_absolute():
            p = (PROC_DIR / r["image_path"]).resolve()
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(r["target"])


def get_transforms(train: bool = False) -> transforms.Compose:
    if train:
        # stronger augmentation pipeline for fine-tuning
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # random erasing operates on tensors
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(name: str, num_classes: int, device: str = "cpu") -> nn.Module:
    # Build pretrained backbone and replace pooling + head with a simple global average pool + linear head
    # Prefer loading a local pretrained state (if present) to mirror known_working_finetune behaviour
    try:
        # try using a plain torchvision constructor (no weights) and then load local checkpoint
        model = getattr(models, name)(weights=None)
    except Exception:
        # fallback to using any available pretrained weights
        try:
            weight_enum_name = f"{name.capitalize()}_Weights"
            weights = None
            if hasattr(models, weight_enum_name):
                weights = getattr(models, weight_enum_name).DEFAULT
            model = getattr(models, name)(weights=weights)
        except Exception:
            model = getattr(models, name)(pretrained=True)

    # attempt to load a local pretrained state dict if it exists (matches known_working approach)
    try:
        local_ckpt_candidates = [
            ROOT / "models" / f"{name}-pretrained.pth",
            ROOT / "models" / f"{name}_pretrained.pth",
            ROOT / "models" / "pretrained" / name / f"{name}_pretrained.pth",
            ROOT / "models" / "pretrained" / name / f"{name}-pretrained.pth",
        ]
        for pth in local_ckpt_candidates:
            if pth.exists():
                try:
                    state = torch.load(pth, map_location="cpu")
                    # if file contains 'state_dict' key (common), extract it
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    model.load_state_dict(state)
                    logger.info("Loaded local pretrained weights for %s from %s", name, pth)
                    break
                except Exception:
                    # try next candidate
                    continue
    except Exception:
        pass

    # ensure global average pooling (ResNet already has adaptive avgpool)
    if hasattr(model, "avgpool"):
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # For binary classification use single logit output (BCEWithLogitsLoss); for multiclass use num_classes outputs
    out_features = 1 if num_classes == 2 else num_classes
    # replace classifier / head
    try:
        in_features = model.fc.in_features
    except Exception:
        in_features = None
    if in_features is None and hasattr(model, "fc"):
        in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, out_features)
    model = model.to(device)
    return model


def count_params(m: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def measure_inference_time(model: nn.Module, loader: DataLoader, device: str = "cpu", max_batches: Optional[int] = 10) -> Dict[str, Any]:
    model.eval()
    times = []
    n = 0
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            imgs = imgs.to(device)
            t0 = time.time()
            _ = model(imgs)
            t1 = time.time()
            times.append(t1 - t0)
            n += imgs.size(0)
    total = sum(times)
    avg_per_batch = total / max(1, len(times))
    avg_per_image = total / max(1, n)
    throughput = n / total if total > 0 else None
    return {"total_s": total, "avg_per_batch_s": avg_per_batch, "avg_per_image_s": avg_per_image, "throughput_images_per_s": throughput}


def evaluate_pytorch_model(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, Any]:
    model.eval()
    ys = []
    preds = []
    probs = []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            # handle binary (single-logit) and multiclass outputs
            if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
                p = torch.sigmoid(out.view(-1)).cpu().numpy()
                pred = (p >= 0.5).astype(int)
            else:
                p = torch.softmax(out, dim=1).cpu().numpy()
                pred = p.argmax(axis=1)
            ys.append(y.numpy())
            preds.append(pred)
            probs.append(p)
    y = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_proba = np.concatenate(probs)
    res = {"accuracy": float((y == y_pred).mean())}
    # precision / recall / f1 / ROC-AUC / PR-AUC
    try:
        # detect binary positive-probabilities vector
        if (y_proba.ndim == 1) or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
            pos_probs = y_proba.ravel()
            # basic metrics
            res["precision"] = float(precision_score(y, y_pred, average="binary", zero_division=0))
            res["recall"] = float(recall_score(y, y_pred, average="binary", zero_division=0))
            res["f1"] = float(f1_score(y, y_pred, average="binary", zero_division=0))
            # ROC AUC
            try:
                res["roc_auc"] = float(roc_auc_score(y, pos_probs))
            except Exception:
                res["roc_auc"] = None
            # PR AUC
            try:
                precision, recall, thresholds = precision_recall_curve(y, pos_probs)
                res["pr_auc"] = float(auc(recall, precision))
            except Exception:
                res["pr_auc"] = None
            # choose threshold achieving >=95% sensitivity (recall) and report specificity, PPV, NPV
            try:
                # precision_recall_curve returns arrays where thresholds length = len(precision)-1
                precision, recall, thresholds = precision_recall_curve(y, pos_probs)
                # find thresholds where recall >= 0.95 (sensitivity)
                if len(thresholds) > 0:
                    # align recalls to thresholds by ignoring the last precision/recall value
                    recall_for_thresh = recall[:-1]
                    mask = recall_for_thresh >= 0.95
                    if mask.any():
                        # choose the threshold with the highest precision among those meeting sensitivity
                        candidate_idxs = np.where(mask)[0]
                        best_idx = candidate_idxs[np.argmax(precision[:-1][candidate_idxs])]
                        thresh = thresholds[best_idx]
                    else:
                        # no threshold achieves 95% recall; use the lowest threshold to maximize sensitivity
                        thresh = float(thresholds.min()) if thresholds.size > 0 else 0.5
                else:
                    thresh = 0.5
                y_pred_thresh = (pos_probs >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, y_pred_thresh).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else None
                ppv = tp / (tp + fp) if (tp + fp) > 0 else None
                npv = tn / (tn + fn) if (tn + fn) > 0 else None
                res["sensitivity_threshold"] = float(thresh)
                res["specificity_at_95_sens"] = None if specificity is None else float(specificity)
                res["ppv_at_95_sens"] = None if ppv is None else float(ppv)
                res["npv_at_95_sens"] = None if npv is None else float(npv)
            except Exception:
                # don't fail whole evaluation over this
                res.update({"sensitivity_threshold": None, "specificity_at_95_sens": None, "ppv_at_95_sens": None, "npv_at_95_sens": None})
        else:
            # multiclass (use macro averages)
            res["precision_macro"] = float(precision_score(y, y_pred, average="macro", zero_division=0))
            res["recall_macro"] = float(recall_score(y, y_pred, average="macro", zero_division=0))
            res["f1_macro"] = float(f1_score(y, y_pred, average="macro", zero_division=0))
            try:
                res["roc_auc_macro"] = float(roc_auc_score(y, y_proba, multi_class="ovo", average="macro"))
            except Exception:
                res["roc_auc_macro"] = None
    except Exception:
        res.update({"precision": None, "recall": None, "f1": None})
    return res


def quick_train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: str = "cpu", epochs: int = 1, save_best: bool = False, model_name: Optional[str] = None, checkpoint_dir: Optional[pathlib.Path] = None, lr: float = 1e-5, weight_decay: float = 1e-5) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    model = model.to(device)
    # Simplify optimizer to match known_working behaviour: single group, use passed LR for finetuning
    # enable full finetuning (allow gradients for all params) so model can learn
    for p in model.parameters():
        p.requires_grad = True
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    # choose criterion based on model head (binary single-logit vs multiclass)
    is_binary_head = hasattr(model, "fc") and getattr(model.fc, "out_features", None) == 1

    # optionally compute pos_weight for class imbalance for BCEWithLogitsLoss
    crit = None
    if is_binary_head:
        try:
            counts = torch.zeros(2, dtype=torch.float64)
            # small pass over train_loader to get class counts
            for _, y in train_loader:
                yv = y.view(-1).to(torch.long)
                binc = torch.bincount(yv, minlength=2).to(torch.float64)
                counts += binc
            if counts[1] > 0:
                # compute positive weight as float then construct tensor on the target device
                ratio = float((counts[0] / counts[1]).item())
                pos_weight_tensor = torch.tensor(ratio, dtype=torch.float32, device=device)
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                crit = nn.BCEWithLogitsLoss()
        except Exception:
            crit = nn.BCEWithLogitsLoss()
    else:
        crit = nn.CrossEntropyLoss()

    # LR scheduler to reduce lr on plateau
    # older PyTorch versions may not accept 'verbose' kwarg
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    except TypeError:
        scheduler = None

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "test_f1": []}
    # track best validation F1 (we optimize for F1) and implement early stopping
    best_f1 = -1.0
    best_epoch = None
    best_info = None
    patience = 5
    no_improve = 0
    if save_best and checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: no early stopping implemented here — training will run for all epochs provided
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for imgs, y in train_iter:
            imgs = imgs.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(imgs)
            # ensure target dtype/shape for BCEWithLogitsLoss
            if is_binary_head:
                # flatten model output to (N,) and targets to float (N,)
                if out.dim() == 2 and out.size(1) == 1:
                    out_flat = out.view(-1)
                else:
                    out_flat = out.view(-1)
                y_in = y.view(-1).float()
                loss = crit(out_flat, y_in)
            else:
                loss = crit(out, y)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
            opt.step()
            running_loss += loss.item() * imgs.size(0)
            # compute predictions for accuracy reporting
            if is_binary_head:
                preds = (torch.sigmoid(out.view(-1)) >= 0.5).long()
            else:
                preds = out.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            total += imgs.size(0)
            # update progress bar with current loss/acc
            if total:
                train_iter.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{(preds==y).float().mean().item():.3f}"})
        train_loss = running_loss / max(1, total)
        train_acc = running_correct / max(1, total)
        # val
        model.eval()
        test_loss = 0.0; test_correct = 0; test_total = 0
        val_ys = []
        val_preds = []
        with torch.no_grad():
            test_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
            for imgs, y in test_iter:
                imgs = imgs.to(device); y = y.to(device)
                out = model(imgs)
                if is_binary_head:
                    y_in = y.view(-1, 1).float()
                else:
                    y_in = y
                loss = crit(out, y_in)
                test_loss += loss.item() * imgs.size(0)
                if is_binary_head:
                    test_pred = (torch.sigmoid(out.view(-1)) >= 0.5).long()
                else:
                    test_pred = out.argmax(dim=1)
                test_correct += (test_pred == y).sum().item()
                test_total += imgs.size(0)
                # collect for F1 computation
                val_ys.append(y.cpu().numpy())
                val_preds.append(test_pred.cpu().numpy())
                if test_total:
                    test_iter.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{(test_pred==y).float().mean().item():.3f}"})
        val_loss = test_loss / max(1, test_total)
        val_acc = test_correct / max(1, test_total)
        try:
            if len(val_ys) > 0:
                y_true = np.concatenate(val_ys)
                y_pred_arr = np.concatenate(val_preds)
                if is_binary_head:
                    val_f1 = float(f1_score(y_true, y_pred_arr, average="binary", zero_division=0))
                else:
                    val_f1 = float(f1_score(y_true, y_pred_arr, average="macro", zero_division=0))
            else:
                val_f1 = None
        except Exception:
            val_f1 = None
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(val_loss)
        history["test_acc"].append(val_acc)
        history["test_f1"].append(val_f1)
        logger.info("Epoch %d: train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f test_f1=%s", epoch+1, train_loss, train_acc, val_loss, val_acc, str(val_f1))

        # step scheduler on validation loss
        try:
            if scheduler is not None:
                scheduler.step(val_loss)
        except Exception:
            pass

        # checkpoint best model (based on validation F1) and implement early stopping
        try:
            improved = False
            if val_f1 is not None:
                if val_f1 > best_f1:
                    improved = True
            # update best and save checkpoint when improved
            if improved:
                best_f1 = float(val_f1)
                best_epoch = epoch + 1
                best_info = {"epoch": best_epoch, "val_f1": best_f1, "val_acc": float(val_acc) if val_acc is not None else None, "timestamp": time.time(), "model_name": model_name}
                no_improve = 0
                if save_best and checkpoint_dir is not None and model_name is not None:
                    try:
                        sd = {k: v.cpu() for k, v in model.state_dict().items()}
                        ckpt_path = checkpoint_dir / f"{model_name}_best.pth"
                        torch.save(sd, str(ckpt_path))
                        meta_path = checkpoint_dir / f"{model_name}_best_meta.json"
                        with open(meta_path, "w") as fh:
                            json.dump(best_info, fh)
                        best_info["ckpt_path"] = str(ckpt_path)
                        best_info["meta_path"] = str(meta_path)
                        logger.info("Saved best checkpoint for %s to %s (val_f1=%.4f)", model_name, ckpt_path, best_f1)
                    except Exception as e:
                        logger.exception("Failed to save checkpoint for %s: %s", model_name, e)
            else:
                no_improve += 1
            # early stopping based on patience
            if no_improve >= patience:
                logger.info("Early stopping: no improvement in validation F1 for %d epochs (patience=%d). Stopping training.", no_improve, patience)
                break
        except Exception:
            pass
    return history, best_info


def model_memory_footprint(m: nn.Module) -> Dict[str, float]:
    """Compute approximate memory footprint for a torch module's parameters and buffers.
    Returns sizes in bytes and megabytes.
    """
    try:
        param_bytes = sum(int(p.numel()) * int(p.element_size()) for p in m.parameters())
        buffer_bytes = sum(int(b.numel()) * int(b.element_size()) for b in m.buffers())
        total_bytes = param_bytes + buffer_bytes
        return {
            "param_bytes": int(param_bytes),
            "buffer_bytes": int(buffer_bytes),
            "total_bytes": int(total_bytes),
            "param_mb": float(param_bytes) / (1024 ** 2),
            "buffer_mb": float(buffer_bytes) / (1024 ** 2),
            "total_mb": float(total_bytes) / (1024 ** 2),
        }
    except Exception:
        return {"param_bytes": None, "buffer_bytes": None, "total_bytes": None, "param_mb": None, "buffer_mb": None, "total_mb": None}


def main(models_list: Optional[List[str]] = None,
         debug_subset: Optional[int] = None,
         device: Optional[str] = None,
         batch_size: int = BATCH_SIZE,
         do_train: bool = False,
         epochs: int = 1,
         lr: float = 1e-5,
         weight_decay: float = 1e-5,
         random_seed: Optional[int] = None) -> Dict[str, Any]:
    # apply random seed
    if random_seed is None:
        random_seed = RANDOM_SEED
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    try:
        torch.cuda.manual_seed_all(int(random_seed))
    except Exception:
        pass
    # start timer for whole script
    start_time = time.time()
    logger.info("Hyperparams - seed: %s epochs: %d batch_size: %d lr: %g weight_decay: %g", random_seed, epochs, batch_size, lr, weight_decay)

    if device is None:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info("Device: %s", device)

    if models_list is None:
        models_list = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    train_df, test_df = pd.read_csv(TRAIN_CSV), pd.read_csv(test_CSV)
    if debug_subset:
        train_df = train_df.sample(n=min(debug_subset, len(train_df)), random_state=RANDOM_SEED).reset_index(drop=True)
        test_df = test_df.sample(n=min(debug_subset, len(test_df)), random_state=RANDOM_SEED).reset_index(drop=True)
    num_classes = int(train_df["target"].nunique())

    transform = get_transforms(train=False)
    train_t = get_transforms(train=True)
    # ensure training dataset uses augmentation transforms
    train_ds = ImageCSVDataSet(train_df, transform=train_t)
    test_ds = ImageCSVDataSet(test_df, transform=transform)
    # small loaders for quick runs
    quick_train_loader = DataLoader(ImageCSVDataSet(train_df.sample(n=min(64, len(train_df)), random_state=int(random_seed)), transform=train_t), batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    quick_test_loader = DataLoader(ImageCSVDataSet(test_df.sample(n=min(64, len(test_df)), random_state=int(random_seed)), transform=transform), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    results: Dict[str, Any] = {}
    for name in models_list:
        logger.info("Building model: %s", name)
        model = build_model(name, num_classes=num_classes, device=device)
        # parameter counts before freezing
        params_total = count_params(model)
        # compute memory footprint
        mem = model_memory_footprint(model)
        # quick verification forward pass (before freezing)
        verify_metrics = measure_inference_time(model, quick_test_loader, device=device, max_batches=5)
        # freeze backbone parameters, keep final FC and last residual blocks trainable for fine-tuning
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze last residual blocks if present (resnet.layer3 + resnet.layer4)
        if hasattr(model, "layer3"):
            for p in model.layer3.parameters():
                p.requires_grad = True
        if hasattr(model, "layer4"):
            for p in model.layer4.parameters():
                p.requires_grad = True
        # ensure head is trainable
        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        params_trainable = count_params(model)
        etest_metrics = None
        try:
            etest_metrics = {}
            # report total params and trainable params (after freezing)
            etest_metrics.update({"params_total": params_total, "params_trainable": params_trainable})
            etest_metrics.update(verify_metrics)
            # include memory footprint
            etest_metrics.update({"memory": mem})
            # run a short eval for accuracy/roc using quick_test_loader
            model.eval()
            ys = []
            preds = []
            probs = []
            with torch.no_grad():
                for imgs, y in quick_test_loader:
                    imgs = imgs.to(device)
                    out = model(imgs)
                    # handle binary (single-logit) and multiclass outputs
                    if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
                        p = torch.sigmoid(out.view(-1)).cpu().numpy()
                        pred = (p >= 0.5).astype(int)
                    else:
                        p = torch.softmax(out, dim=1).cpu().numpy()
                        pred = p.argmax(axis=1)
                    ys.append(y.numpy()); preds.append(pred); probs.append(p)
            y = np.concatenate(ys); y_pred = np.concatenate(preds); y_proba = np.concatenate(probs)
            etest_metrics["accuracy"] = float((y == y_pred).mean())
            # precision / recall / f1
            try:
                if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
                    # binary
                    etest_metrics["precision"] = float(precision_score(y, y_pred, average="binary", zero_division=0))
                    etest_metrics["recall"] = float(recall_score(y, y_pred, average="binary", zero_division=0))
                    etest_metrics["f1"] = float(f1_score(y, y_pred, average="binary", zero_division=0))
                    try:
                        etest_metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
                    except Exception:
                        etest_metrics["roc_auc"] = None
                else:
                    # multiclass (use macro averages)
                    etest_metrics["precision_macro"] = float(precision_score(y, y_pred, average="macro", zero_division=0))
                    etest_metrics["recall_macro"] = float(recall_score(y, y_pred, average="macro", zero_division=0))
                    etest_metrics["f1_macro"] = float(f1_score(y, y_pred, average="macro", zero_division=0))
                    try:
                        etest_metrics["roc_auc_macro"] = float(roc_auc_score(y, y_proba, multi_class="ovo", average="macro"))
                    except Exception:
                        etest_metrics["roc_auc_macro"] = None
            except Exception:
                etest_metrics.update({"precision": None, "recall": None, "f1": None})
        except Exception as e:
            logger.warning("Eval metrics failed: %s", e)
        # optional short train
        history = None
        if do_train:
            # quick_train uses trainable params opt creation inside; override LR via global var by setting attr
            history, best_info = quick_train(model, quick_train_loader, quick_test_loader, device=device, epochs=epochs, save_best=True, model_name=name, checkpoint_dir=ROOT / "models" / "checkpoints", lr=lr, weight_decay=weight_decay)
            # save training curves to images folder
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history["train_loss"], label="train_loss")
            ax[0].plot(history["test_loss"], label="test_loss")
            ax[0].set_title(f"{name} - Training and Validation Loss")
            ax[0].legend()
            ax[1].plot(history["train_acc"], label="train_acc")
            ax[1].plot(history["test_acc"], label="test_acc")
            ax[1].set_title(f"{name} - Training and Validation Accuracy")
            ax[1].legend()
            # annotate max test accuracy with dotted red line
            try:
                if history and "test_acc" in history and len(history["test_acc"]) > 0:
                    max_acc = max(history["test_acc"])
                    max_epoch = history["test_acc"].index(max_acc) + 1
                    ax[1].axhline(max_acc, color="red", linestyle=":", linewidth=2.5)
                    # place label slightly above the line using a small offset based on axis range
                    ylim = ax[1].get_ylim()
                    y_offset = (ylim[1] - ylim[0]) * 0.02
                    ax[1].text(len(history["test_acc"]) - 0.5, max_acc + y_offset, f"  max test acc={max_acc:.3f} (ep{max_epoch})", color="red", va="bottom", ha="right", fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))
            except Exception as e:
                logger.warning("Failed to annotate max test accuracy for %s: %s", name, e)
            try:
                out_file = OUT_DIR_IMGS / f"{name}_training_curves.png"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(out_file), bbox_inches="tight")
                logger.info("Saved training curves to %s", out_file)
            except Exception as e:
                logger.exception("Failed to save training curves for %s: %s", name, e)
            plt.close(fig)
        # save metrics
        # report final validation metrics from the last training epoch (if training ran) rather than best-so-far
        final_val_loss = None
        final_val_acc = None
        final_val_f1 = None
        if history:
            try:
                if "test_loss" in history and len(history["test_loss"]) > 0:
                    final_val_loss = history["test_loss"][-1]
                if "test_acc" in history and len(history["test_acc"]) > 0:
                    final_val_acc = history["test_acc"][-1]
                if "test_f1" in history and len(history["test_f1"]) > 0:
                    final_val_f1 = history["test_f1"][-1]
            except Exception:
                pass
        logger.info("Final validation for %s: val_loss=%s val_acc=%s val_f1=%s", name, str(final_val_loss), str(final_val_acc), str(final_val_f1))
        # memory footprint
        mem_footprint = model_memory_footprint(model)
        res = {"model": name, "params_total": params_total, "params_trainable": params_trainable, "verify": verify_metrics, "eval": etest_metrics, "history": history, "final_val_loss": final_val_loss, "final_val_acc": final_val_acc, "final_val_f1": final_val_f1, "memory_footprint": mem_footprint}
        # attach memory to result top-level for easy access
        res["memory"] = mem
        if best_info is not None:
            res["best_info"] = best_info
        results[name] = res
        # collect summary row for df
        try:
            summary_row = {
                "model": name,
                "params_total": params_total["total"],
                "params_trainable": params_trainable["trainable"],
                "memory_mb": mem.get("total_mb"),
                "inference_avg_per_image_s": verify_metrics.get("avg_per_image_s"),
                "throughput_ips": verify_metrics.get("throughput_images_per_s"),
                "final_val_acc": final_val_acc,
                "final_val_f1": final_val_f1,
                "etest_roc_auc": etest_metrics.get("roc_auc") if etest_metrics else None,
            }
            if "summary_rows" not in locals():
                summary_rows = []
            summary_rows.append(summary_row)
        except Exception:
            pass
        # do not save model state (brief or full) per configuration
        # free GPU memory before next model to avoid OOM
        try:
            del model
            # clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # small sleep
        time.sleep(0.5)

    # record total runtime
    total_s = time.time() - start_time
    results["script_runtime_s"] = float(total_s)
    logger.info("Total script runtime: %.2fs", total_s)

    # write summary
    out_path = OUT_DIR_TBLS / "resnet_summary.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote resnet summary to %s", out_path)

    # save dataframe summary
    try:
        if 'summary_rows' in locals() and len(summary_rows) > 0:
            df_sum = pd.DataFrame(summary_rows)
            df_out = OUT_DIR_TBLS / "resnet_summary_tbl.csv"
            df_sum.to_csv(df_out, index=False)
            logger.info("Wrote resnet summary table to %s", df_out)
    except Exception:
        logger.exception("Failed to write summary table")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick ResNet baseline verifier and metrics")
    p.add_argument("--models", type=str, nargs="*", default=None, help="ResNet variants to test")
    p.add_argument("--debug_subset", type=int, default=None, help="Quick subset size")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda/mps")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--train", action="store_true", help="Run brief training to produce curves")
    p.add_argument("--epochs", type=int, default=20, help="Number of epochs for quick training")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate for optimizer")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    p.add_argument("--random_seed", type=int, default=None, help="Random seed override")
    args = p.parse_args()
    main(models_list=args.models, debug_subset=args.debug_subset, device=args.device, batch_size=args.batch_size, do_train=args.train, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, random_seed=args.random_seed)
