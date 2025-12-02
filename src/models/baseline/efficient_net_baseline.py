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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROC_DIR / "train_data.csv"
TEST_CSV = PROC_DIR / "test_data.csv"
PRETRAIN_DIR = ROOT / "models" / "pretrained" / "efficient_net"
OUT_DIR_TBLS = ROOT / "artifacts" / "tbls" / "01_baselines" / "efficient_net"
OUT_DIR_IMGS = ROOT / "artifacts" / "imgs" / "01_baselines" / "efficient_net"
OUT_DIR_TBLS.mkdir(parents=True, exist_ok=True)
OUT_DIR_IMGS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "models" / "logs" / "efficient_net_baseline.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(name: str, num_classes: int, device: str = "cpu", pretrained_path: Optional[pathlib.Path] = None) -> nn.Module:
    # construct model via torchvision factory
    ctor = getattr(models, name, None)
    if ctor is None:
        raise ValueError(f"Unknown model: {name}")
    try:
        # try default constructor
        model = ctor(pretrained=False)
    except Exception:
        model = ctor()
    model = model.to(device)
    # load pretrained state_dict if provided into the original model BEFORE replacing classifier
    if pretrained_path is not None and pretrained_path.exists():
        try:
            sd = torch.load(pretrained_path, map_location="cpu")
            # support full model or state_dict
            if isinstance(sd, dict):
                model_sd = model.state_dict()
                cleaned = {}
                for k, v in sd.items():
                    # handle module. prefix from DataParallel checkpoints
                    k2 = k.replace("module.", "")
                    if k2 in model_sd:
                        try:
                            if isinstance(v, torch.Tensor) and v.shape == model_sd[k2].shape:
                                cleaned[k2] = v
                            else:
                                logger.info("Skipping pretrained param %s due to shape mismatch (%s != %s)", k2, getattr(v, 'shape', type(v)), model_sd[k2].shape)
                        except Exception:
                            logger.info("Skipping pretrained param %s (unable to compare shapes)", k2)
                    else:
                        logger.debug("Pretrained param %s not found in model, skipping", k2)
                # load filtered state dict
                model.load_state_dict(cleaned, strict=False)
            else:
                # fallback: try to load directly
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    logger.warning("Pretrained file %s could not be loaded strictly", pretrained_path)
            logger.info("Loaded pretrained weights from %s", pretrained_path)
        except Exception as e:
            logger.warning("Failed to load pretrained %s: %s", pretrained_path, e)
    # now replace the classifier to match the desired num_classes (after loading pretrained backbone)
    if hasattr(model, "classifier"):
        try:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        except Exception:
            model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(model.classifier[-1].in_features, num_classes))
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
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


def quick_train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: str = "cpu", epochs: int = 1, save_best: bool = False, model_name: Optional[str] = None, checkpoint_dir: Optional[pathlib.Path] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    model = model.to(device)
    # only final classifier parameters should be trainable
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_val = -1.0
    best_epoch = None
    best_info = None
    if save_best and checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: there is no early stopping here — training will always run for the full number
    # of epochs provided and metrics for every epoch will be recorded regardless of
    # whether validation accuracy improves.
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for imgs, y in train_iter:
            imgs = imgs.to(device); y = y.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, y)
            loss.backward(); opt.step()
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            total += imgs.size(0)
            if total:
                train_iter.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{(preds==y).float().mean().item():.3f}"})
        train_loss = running_loss / max(1, total)
        train_acc = running_correct / max(1, total)
        # val
        model.eval()
        test_loss = 0.0; test_correct = 0; test_total = 0
        with torch.no_grad():
            test_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
            for imgs, y in test_iter:
                imgs = imgs.to(device); y = y.to(device)
                out = model(imgs)
                loss = crit(out, y)
                test_loss += loss.item() * imgs.size(0)
                test_correct += (out.argmax(dim=1) == y).sum().item()
                test_total += imgs.size(0)
                if test_total:
                    test_iter.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{(out.argmax(dim=1)==y).float().mean().item():.3f}"})
        val_loss = test_loss / max(1, test_total)
        val_acc = test_correct / max(1, test_total)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(val_loss)
        history["test_acc"].append(val_acc)
        logger.info("Epoch %d: train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f", epoch+1, train_loss, train_acc, val_loss, val_acc)

        # checkpoint best model (based on val_acc)
        try:
            if val_acc > best_val:
                best_val = float(val_acc)
                best_epoch = epoch + 1
                best_info = {"epoch": best_epoch, "val_acc": best_val, "timestamp": time.time(), "model_name": model_name}
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
                        logger.info("Saved best checkpoint for %s to %s", model_name, ckpt_path)
                    except Exception as e:
                        logger.exception("Failed to save checkpoint for %s: %s", model_name, e)
        except Exception:
            pass
    return history, best_info


def main(models_list: Optional[List[str]] = None, debug_subset: Optional[int] = None, device: Optional[str] = None, batch_size: int = BATCH_SIZE, do_train: bool = False, epochs: int = 1, random_seed: Optional[int] = None) -> Dict[str, Any]:
    if random_seed is None:
        random_seed = RANDOM_SEED
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    try:
        torch.cuda.manual_seed_all(int(random_seed))
    except Exception:
        pass
    start_time = time.time()
    logger.info("Hyperparams - seed: %s epochs: %d batch_size: %d", random_seed, epochs, batch_size)

    if device is None:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info("Device: %s", device)

    if models_list is None:
        models_list = [f"efficientnet_b{i}" for i in range(8)]

    train_df, test_df = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    if debug_subset:
        train_df = train_df.sample(n=min(debug_subset, len(train_df)), random_state=RANDOM_SEED).reset_index(drop=True)
        test_df = test_df.sample(n=min(debug_subset, len(test_df)), random_state=RANDOM_SEED).reset_index(drop=True)
    num_classes = int(train_df["target"].nunique())

    transform = get_transforms(train=False)
    train_t = get_transforms(train=True)
    # small loaders for quick runs
    quick_train_loader = DataLoader(ImageCSVDataSet(train_df.sample(n=min(64, len(train_df)), random_state=int(random_seed)), transform=train_t), batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    quick_test_loader = DataLoader(ImageCSVDataSet(test_df.sample(n=min(64, len(test_df)), random_state=int(random_seed)), transform=transform), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    results: Dict[str, Any] = {}
    for name in models_list:
        logger.info("Building model: %s", name)
        pretrained_path = PRETRAIN_DIR / f"{name}_pretrained.pth"
        model = build_model(name, num_classes=num_classes, device=device, pretrained_path=pretrained_path)
        params_total = count_params(model)
        # verify
        verify_metrics = measure_inference_time(model, quick_test_loader, device=device, max_batches=5)
        # freeze then fine-tune only classifier
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        params_trainable = count_params(model)

        # quick eval
        eval_metrics = None
        try:
            eval_metrics = {}
            eval_metrics.update({"params_total": params_total, "params_trainable": params_trainable})
            eval_metrics.update(verify_metrics)
            model.eval()
            ys = []
            preds = []
            probs = []
            with torch.no_grad():
                for imgs, y in quick_test_loader:
                    imgs = imgs.to(device)
                    out = model(imgs)
                    if out.dim()==1 or out.size(1)==1:
                        p = torch.sigmoid(out.view(-1)).cpu().numpy()
                        pred = (p>=0.5).astype(int)
                    else:
                        p = torch.softmax(out, dim=1).cpu().numpy()
                        pred = p.argmax(axis=1)
                    ys.append(y.numpy()); preds.append(pred); probs.append(p)
            y = np.concatenate(ys); y_pred = np.concatenate(preds); y_proba = np.concatenate(probs)
            eval_metrics["accuracy"] = float((y == y_pred).mean())
            try:
                if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
                    eval_metrics["precision"] = float(precision_score(y, y_pred, average="binary", zero_division=0))
                    eval_metrics["recall"] = float(recall_score(y, y_pred, average="binary", zero_division=0))
                    eval_metrics["f1"] = float(f1_score(y, y_pred, average="binary", zero_division=0))
                    try:
                        eval_metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
                    except Exception:
                        eval_metrics["roc_auc"] = None
                else:
                    eval_metrics["precision_macro"] = float(precision_score(y, y_pred, average="macro", zero_division=0))
                    eval_metrics["recall_macro"] = float(recall_score(y, y_pred, average="macro", zero_division=0))
                    eval_metrics["f1_macro"] = float(f1_score(y, y_pred, average="macro", zero_division=0))
                    try:
                        eval_metrics["roc_auc_macro"] = float(roc_auc_score(y, y_proba, multi_class="ovo", average="macro"))
                    except Exception:
                        eval_metrics["roc_auc_macro"] = None
            except Exception:
                eval_metrics.update({"precision": None, "recall": None, "f1": None})
        except Exception as e:
            logger.warning("Eval metrics failed: %s", e)

        history = None
        best_info = None
        if do_train:
            history, best_info = quick_train(model, quick_train_loader, quick_test_loader, device=device, epochs=epochs, save_best=True, model_name=name, checkpoint_dir=ROOT / "models" / "checkpoints")
            # save training curves
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history["train_loss"], label="train_loss")
            ax[0].plot(history["test_loss"], label="test_loss")
            ax[0].set_title("Loss")
            ax[0].legend()
            ax[1].plot(history["train_acc"], label="train_acc")
            ax[1].plot(history["test_acc"], label="test_acc")
            ax[1].set_title("Accuracy")
            ax[1].legend()
            try:
                if history and "test_acc" in history and len(history["test_acc"]) > 0:
                    max_acc = max(history["test_acc"])
                    max_epoch = history["test_acc"].index(max_acc) + 1
                    ax[1].axhline(max_acc, color="red", linestyle=":", linewidth=2.5)
                    ax[1].text(len(history["test_acc"]) - 0.5, max_acc, f"  max test acc={max_acc:.3f} (ep{max_epoch})", color="red", va="center", ha="right", fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.3'))
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

        res = {"model": name, "params_total": params_total, "params_trainable": params_trainable, "verify": verify_metrics, "eval": eval_metrics, "history": history}
        if best_info is not None:
            res["best_info"] = best_info
        results[name] = res

        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(0.5)

    total_s = time.time() - start_time
    results["script_runtime_s"] = float(total_s)
    logger.info("Total script runtime: %.2fs", total_s)

    out_path = OUT_DIR_TBLS / "efficientnet_summary.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote efficientnet summary to %s", out_path)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick EfficientNet baseline verifier and metrics")
    p.add_argument("--models", type=str, nargs="*", default=None, help="EfficientNet variants to test")
    p.add_argument("--debug_subset", type=int, default=None, help="Quick subset size")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda/mps")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--train", action="store_true", help="Run brief training to produce curves")
    p.add_argument("--epochs", type=int, default=3, help="Number of epochs for quick training")
    p.add_argument("--random_seed", type=int, default=None, help="Random seed override")
    args = p.parse_args()
    main(models_list=args.models, debug_subset=args.debug_subset, device=args.device, batch_size=args.batch_size, do_train=args.train, epochs=args.epochs, random_seed=args.random_seed)
