"""
This script understands exploratory data analysis (EDA) 
"""

import json
import logging
import pathlib
from typing import Optional, Dict, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
DEFAULT_CSV = PROC_DIR / "labeled_ham10000.csv"
OUT_DIR = ROOT / "models" / "figures" / "target_profile"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set(style="whitegrid")


def profile_target(
    csv_path: Optional[pathlib.Path] = None,
    out_dir: Optional[pathlib.Path] = None,
    top_k_localization: int = 10,
) -> Dict[str, Any]:
    """
    Profile the 'target' column and related covariates.

    Produces and saves:
      - target count bar plot
      - lesion_type counts (top categories) bar plot
      - age distribution by target (box + histogram)
      - sex distribution by target (stacked bar)
      - top localizations bar plot

    Returns a summary dict with counts, proportions and basic stats.
    """
    csv_path = pathlib.Path(csv_path) if csv_path is not None else DEFAULT_CSV
    if out_dir is None:
        out_dir = OUT_DIR
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)

    summary: Dict[str, Any] = {}
    # Basic target counts
    target_counts = df["target"].value_counts().sort_index()
    target_props = (target_counts / target_counts.sum()).round(4)
    summary["target_counts"] = target_counts.to_dict()
    summary["target_props"] = target_props.to_dict()

    # Lesion type distribution
    if "lesion_type" in df.columns:
        lesion_counts = df["lesion_type"].value_counts()
        summary["lesion_counts_top10"] = lesion_counts.head(10).to_dict()
    else:
        lesion_counts = pd.Series(dtype=int)

    # Age stats
    age_stats = {}
    if "age" in df.columns:
        age_series = pd.to_numeric(df["age"], errors="coerce")
        age_stats["overall"] = {
            "count": int(age_series.count()),
            "mean": float(np.nanmean(age_series)),
            "std": float(np.nanstd(age_series)),
            "min": float(np.nanmin(age_series)),
            "max": float(np.nanmax(age_series)),
        }
        age_by_target = age_series.groupby(df["target"]).agg(["count", "mean", "std", "min", "max"]).to_dict()
        age_stats["by_target"] = {k: {stat: float(v) for stat, v in vals.items()} for k, vals in age_by_target.items()}
    summary["age_stats"] = age_stats

    # Sex distribution
    sex_ct = df["sex"].fillna("unknown").value_counts()
    sex_by_target = pd.crosstab(df["sex"].fillna("unknown"), df["target"])
    summary["sex_counts"] = sex_ct.to_dict()
    summary["sex_by_target"] = sex_by_target.to_dict()

    # Localization top-k
    if "localization" in df.columns:
        loc_top = df["localization"].fillna("unknown").value_counts().head(top_k_localization)
        summary["localization_top"] = loc_top.to_dict()
    else:
        loc_top = pd.Series(dtype=int)

    # PLOTTING
    try:
        # target bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, palette="muted", ax=ax)
        ax.set_title("Target counts")
        ax.set_xlabel("target")
        ax.set_ylabel("count")
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + max(target_counts.values) * 0.01, str(int(v)), ha="center")
        fig.savefig(out_dir / "target_counts.png", bbox_inches="tight")
        plt.close(fig)

        # lesion type top categories
        if not lesion_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            top = lesion_counts.head(10)
            sns.barplot(y=top.index, x=top.values, palette="viridis", ax=ax)
            ax.set_title("Top 10 lesion types")
            ax.set_xlabel("count")
            ax.set_ylabel("")
            fig.savefig(out_dir / "lesion_type_top10.png", bbox_inches="tight")
            plt.close(fig)

        # age distribution by target: box + violin
        if "age" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            age_df = df.copy()
            age_df["age_num"] = pd.to_numeric(age_df["age"], errors="coerce")
            sns.boxplot(x="target", y="age_num", data=age_df, ax=axes[0], palette="pastel")
            axes[0].set_title("Age by target (boxplot)")
            sns.histplot(data=age_df, x="age_num", hue="target", kde=True, ax=axes[1], palette="dark")
            axes[1].set_title("Age distribution by target")
            fig.savefig(out_dir / "age_by_target.png", bbox_inches="tight")
            plt.close(fig)

        # sex distribution stacked bar
        fig, ax = plt.subplots(figsize=(6, 4))
        sex_by_target_norm = sex_by_target.div(sex_by_target.sum(axis=0), axis=1)
        sex_by_target_norm.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title("Sex distribution by target (proportion)")
        ax.set_xlabel("sex")
        ax.set_ylabel("proportion")
        fig.savefig(out_dir / "sex_by_target.png", bbox_inches="tight")
        plt.close(fig)

        # localization top-k
        if not loc_top.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(y=loc_top.index, x=loc_top.values, palette="magma", ax=ax)
            ax.set_title(f"Top {top_k_localization} localizations")
            ax.set_xlabel("count")
            ax.set_ylabel("")
            fig.savefig(out_dir / "localization_topk.png", bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # Save summary json
    summary_path = out_dir / "target_profile_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved profile summary and figures to %s", out_dir)

    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Profile target and related covariates in HAM10000 CSV.")
    p.add_argument("--csv", type=str, default=None, help="Path to processed CSV (defaults to labeled_ham10000.csv)")
    p.add_argument("--out", type=str, default=None, help="Output directory for figures")
    p.add_argument("--topk", type=int, default=10, help="Top-K localizations to show")
    args = p.parse_args()
    profile_target(csv_path=pathlib.Path(args.csv) if args.csv else None, out_dir=pathlib.Path(args.out) if args.out else None, top_k_localization=args.topk)