import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import pandas as pd


def _deep_get(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """Try to find a value in dict `d` by multiple candidate keys (including nested)."""
    for key in keys:
        # direct key
        if key in d:
            return d[key]
        # nested common places
        for top in ("metrics", "summary", "history", "validation", "val_metrics"):
            if top in d and isinstance(d[top], dict) and key in d[top]:
                return d[top][key]
        # history style: lists of epoch metrics
        for hist_key in ("history", "train_history", "val_history"):
            hist = d.get(hist_key)
            if isinstance(hist, dict) and key in hist:
                v = hist[key]
                if isinstance(v, list) and v:
                    return v[-1]
    return None


def _nested_get(d: Dict[str, Any], *path: str) -> Optional[Any]:
    """Traverse nested dict `d` by path components and return value or None."""
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def extract_metrics_from_json(path: Path) -> Dict[str, Any]:
    """Load a json file and extract a normalized set of baseline metrics.

    Enhanced to pull metrics out of nested `baseline` and `tuned` sections
    (common for non-deep-learning artifacts like sklearn model summaries).
    """
    with path.open("r") as f:
        data = json.load(f)

    def find(*candidates):
        return _deep_get(data, candidates)

    def ng(*parts):
        return _nested_get(data, *parts)

    metrics = {
        "source_file": str(path),
        "run_id": data.get("run_id") or data.get("id") or path.stem,
        "timestamp": data.get("timestamp"),

        # prefer generic find, fallback to nested baseline/tuned sections
        "val_accuracy": (
            find("best_val_accuracy", "best_val_acc", "val_accuracy", "val_acc", "validation_accuracy")
            or ng("baseline", "val", "accuracy")
            or ng("tuned", "val", "accuracy")
        ),
        "train_accuracy": (
            find("train_accuracy", "train_acc")
            or ng("baseline", "train", "accuracy")
            or ng("tuned", "train", "accuracy")
        ),
        "val_roc_auc": (
            find("val_roc_auc", "validation_roc_auc")
            or ng("baseline", "val", "roc_auc")
            or ng("tuned", "val", "roc_auc")
        ),
        "train_roc_auc": (
            ng("baseline", "train", "roc_auc")
            or ng("tuned", "train", "roc_auc")
        ),
        "val_loss": find("best_val_loss", "val_loss", "validation_loss"),
        "train_loss": find("train_loss", "loss"),
        "best_epoch": find("best_epoch", "epoch_of_best", "best_val_epoch"),

        "tuned_best_params": ng("tuned", "best_params") or ng("best_params"),
        "grid_cv_results": data.get("grid_cv_results"),
    }

    # try to capture any additional top-level numeric/string/bool metrics for debugging/comparison
    for k, v in data.items():
        if k in metrics or k in ("metrics", "summary", "history", "baseline", "tuned"):
            continue
        if isinstance(v, (int, float, str, bool)):
            metrics[k] = v

    return metrics


def build_metrics_table(json_dir: str | Path, pattern: str = "**/*.json") -> pd.DataFrame:
    """
    Scan json_dir for JSON summary/artifact files, extract metrics and return a sorted DataFrame.
    Rows are sorted with best validation accuracy on top. Missing val_accuracy are treated as NaN.
    """
    p = Path(json_dir)
    files = sorted(p.glob(pattern))
    rows = []
    for f in files:
        try:
            row = extract_metrics_from_json(f)
            rows.append(row)
        except Exception:
            # skip unreadable / invalid json but continue
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ensure numeric types where possible
    for col in ("val_accuracy", "train_accuracy", "val_loss", "train_loss", "best_epoch"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sort highest validation accuracy first, push NaNs to the bottom
    if "val_accuracy" in df.columns:
        df = df.sort_values(by="val_accuracy", ascending=False, na_position="last").reset_index(drop=True)

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build metric table from json artifacts/summaries.")
    parser.add_argument("json_dir", nargs="?", default=".", help="Directory to scan for JSON files (default: current dir)")
    parser.add_argument("--out", "-o", help="Optional CSV path to write the table")
    args = parser.parse_args()

    table = build_metrics_table(args.json_dir)
    if table.empty:
        print("No JSON metric files found or no parsable metrics extracted.")
    else:
        pd.set_option("display.max_columns", None)
        print(table)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"Wrote table to {args.out}")