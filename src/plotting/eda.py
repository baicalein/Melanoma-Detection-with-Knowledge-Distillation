"""
This script understands exploratory data analysis (EDA) 
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

ROOT = pathlib.Path(__file__).parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
DEFAULT_CSV = PROC_DIR / "labeled_ham10000.csv"
OUT_DIR = ROOT / "artifacts"
LOG_PATH = ROOT / "models" / "logs" / "eda.log"
SRC_DIR = ROOT / "src"

sys.path.append(str(SRC_DIR))
from utils import uva_colors  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["savefig.dpi"] = 300


def get_uva_palette(n: int):
    """Return a list of colors based on uva_colors that is safe for seaborn/matplotlib.

    Handles cases where uva_colors may be a dict, list, map, or palette name string.
    """
    try:
        # if uva_colors is a string palette name, let seaborn handle it
        if isinstance(uva_colors, str):
            return sns.color_palette(uva_colors, n_colors=n)
        # if it's a mapping (dict), use its values
        if isinstance(uva_colors, dict):
            base = list(uva_colors.values())
        else:
            # try to coerce to list (handles map object, generator, list, tuple)
            base = list(uva_colors)
        # if base contains color names/hex, create palette
        return sns.color_palette(base, n_colors=n)
    except Exception:
        # fallback to a default seaborn color palette
        return sns.color_palette("muted", n_colors=n)


# ensure image subfolders exist before saving
IMG_SUBDIR = out_dir = OUT_DIR / "imgs" / "00_eda"
IMG_SUBDIR.mkdir(parents=True, exist_ok=True)


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
        colors = get_uva_palette(len(target_counts))
        x_pos = list(range(len(target_counts)))
        ax.bar(x_pos, target_counts.values, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_counts.index.astype(str))
        ax.set_title("Target counts")
        ax.set_xlabel("target")
        ax.set_ylabel("count")
        # format y-axis ticks with thousands separator
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + max(target_counts.values) * 0.01, f"{int(v):,}", ha="center")
        fig.savefig(IMG_SUBDIR / "target_counts.png", bbox_inches="tight")
        plt.close(fig)

        # lesion type top categories
        if not lesion_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            top = lesion_counts.head(10)
            colors = get_uva_palette(len(top))
            # plot horizontal bars with consistent palette
            y_pos = list(range(len(top)))[::-1]
            ax.barh(y_pos, top.values[::-1], color=colors[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top.index[::-1])
            ax.set_title("Top 10 lesion types")
            ax.set_xlabel("count")
            ax.set_ylabel("")
            fig.savefig(IMG_SUBDIR / "lesion_type_top10.png", bbox_inches="tight")
            plt.close(fig)

        # age distribution by target: box + violin
        if "age" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            age_df = df.copy()
            age_df["age_num"] = pd.to_numeric(age_df["age"], errors="coerce")
            # use a temporary palette to avoid passing `palette` without `hue` (deprecated)
            tmp_palette = get_uva_palette(age_df["target"].nunique())
            prev = sns.color_palette()
            sns.set_palette(tmp_palette)
            sns.boxplot(x="target", y="age_num", data=age_df, ax=axes[0])
            sns.set_palette(prev)
            axes[1].set_title("Boxplot")
            sns.histplot(data=age_df, x="age_num", hue="target", kde=True, ax=axes[1], palette=get_uva_palette(age_df["target"].nunique()))
            axes[0].set_title("Age distribution by target")
            fig.savefig(IMG_SUBDIR / "age_by_target.png", bbox_inches="tight")
            plt.close(fig)

        # sex distribution stacked bar
        fig, ax = plt.subplots(figsize=(6, 4))
        # compute proportions of sexes within each target, then plot per-target stacked bars
        sex_by_target_prop = sex_by_target.div(sex_by_target.sum(axis=0), axis=1)  # columns are targets
        df_plot = sex_by_target_prop.T  # index=target, columns=sex categories
        color_list = get_uva_palette(df_plot.shape[1])
        df_plot.plot(kind="bar", stacked=True, ax=ax, color=color_list)
        ax.set_title("Sex distribution by target (proportion)")
        ax.set_xlabel("target")
        ax.set_ylabel("proportion")
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        fig.savefig(IMG_SUBDIR / "sex_by_target.png", bbox_inches="tight")
        plt.close(fig)

        # localization top-k
        if not loc_top.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = get_uva_palette(len(loc_top))
            y_pos = list(range(len(loc_top)))[::-1]
            ax.barh(y_pos, loc_top.values[::-1], color=colors[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(loc_top.index[::-1])
            ax.set_title(f"Top {top_k_localization} localizations")
            ax.set_xlabel("count")
            ax.set_ylabel("")
            fig.savefig(IMG_SUBDIR / "localization_topk.png", bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # Save summary json
    summary_path = out_dir / "target_profile_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved profile summary to %s", out_dir / "tbls" / "00_eda" / "")

    return summary

# run eda if called as script
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Profile target and related covariates in HAM10000 CSV.")
    p.add_argument("--csv", type=str, default=None, help="Path to processed CSV (defaults to labeled_ham10000.csv)")
    p.add_argument("--out", type=str, default=None, help="Output directory for figures")
    p.add_argument("--topk", type=int, default=10, help="Top-K localizations to show")
    args = p.parse_args()
    profile_target(csv_path=pathlib.Path(args.csv) if args.csv else None, out_dir=pathlib.Path(args.out) if args.out else None, top_k_localization=args.topk)