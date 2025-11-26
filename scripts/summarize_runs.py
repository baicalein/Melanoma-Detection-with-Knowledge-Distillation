#!/usr/bin/env python
"""
Summarize all training runs and identify missing experiments.

This script:
1. Scans checkpoints and logs for completed runs
2. Loads metrics from saved JSON files
3. Generates a summary table (console + CSV)
4. Identifies which experiments still need to be run

Usage:
    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --output artifacts/tbls/run_summary.csv
"""

import argparse
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd

from src.config import CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Expected Experiments (based on project requirements)
# ============================================================================

# Teacher architectures to evaluate
TEACHER_ARCHITECTURES = ["resnet34"]
TEACHER_LOSS_TYPES = ["focal"]

# KD hyperparameter grid (focused search per feedback)
KD_TEMPERATURES = [1.0, 2.0]
KD_ALPHAS = [0.5, 0.9]

# Quantization methods
QUANTIZATION_METHODS = ["dynamic"]


@dataclass
class ExperimentStatus:
    """Status of an experiment."""
    name: str
    experiment_type: str  # "teacher", "student", "quantization"
    completed: bool
    checkpoint_path: Optional[str] = None
    metrics_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None


def find_teacher_runs() -> List[ExperimentStatus]:
    """Find all completed teacher training runs."""
    runs = []
    
    for arch in TEACHER_ARCHITECTURES:
        for loss in TEACHER_LOSS_TYPES:
            exp_name = f"teacher_{arch}_{loss}"
            
            # Check for checkpoint
            ckpt_path = CHECKPOINTS_DIR / f"{exp_name}_best.pth"
            meta_path = CHECKPOINTS_DIR / f"{exp_name}_best_meta.json"
            metrics_path = FIGURES_DIR / "training" / exp_name / "holdout_metrics.json"
            
            completed = ckpt_path.exists()
            
            metrics = None
            timestamp = None
            
            # Load metrics if available
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                except Exception:
                    pass
            
            # Get timestamp from meta
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                        timestamp = meta.get("timestamp")
                except Exception:
                    pass
            
            runs.append(ExperimentStatus(
                name=exp_name,
                experiment_type="teacher",
                completed=completed,
                checkpoint_path=str(ckpt_path) if completed else None,
                metrics_path=str(metrics_path) if metrics_path.exists() else None,
                metrics=metrics,
                timestamp=timestamp,
            ))
    
    return runs


def find_student_runs() -> List[ExperimentStatus]:
    """Find all completed student KD training runs."""
    runs = []
    
    for temp, alpha in product(KD_TEMPERATURES, KD_ALPHAS):
        exp_name = f"student_T{temp}_alpha{alpha}"
        
        # Check for checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"{exp_name}_best.pth"
        meta_path = CHECKPOINTS_DIR / f"{exp_name}_best_meta.json"
        metrics_path = FIGURES_DIR / "training" / exp_name / "student_holdout_metrics.json"
        
        completed = ckpt_path.exists()
        
        metrics = None
        timestamp = None
        
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except Exception:
                pass
        
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                    timestamp = meta.get("timestamp")
            except Exception:
                pass
        
        runs.append(ExperimentStatus(
            name=exp_name,
            experiment_type="student",
            completed=completed,
            checkpoint_path=str(ckpt_path) if completed else None,
            metrics_path=str(metrics_path) if metrics_path.exists() else None,
            metrics=metrics,
            timestamp=timestamp,
        ))
    
    return runs


def find_quantization_runs() -> List[ExperimentStatus]:
    """Find all completed quantization runs."""
    runs = []
    
    for method in QUANTIZATION_METHODS:
        exp_name = f"quantized_{method}"
        
        ckpt_path = CHECKPOINTS_DIR / f"{exp_name}_quantized.pth"
        metrics_path = FIGURES_DIR / "quantization" / exp_name / "comparison.json"
        
        completed = ckpt_path.exists()
        
        metrics = None
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                    # Extract INT8 metrics
                    metrics = data.get("int8", {})
            except Exception:
                pass
        
        runs.append(ExperimentStatus(
            name=exp_name,
            experiment_type="quantization",
            completed=completed,
            checkpoint_path=str(ckpt_path) if completed else None,
            metrics_path=str(metrics_path) if metrics_path.exists() else None,
            metrics=metrics,
        ))
    
    return runs


def generate_summary_table(runs: List[ExperimentStatus]) -> pd.DataFrame:
    """Generate summary DataFrame from experiment statuses."""
    rows = []
    
    for run in runs:
        row = {
            "Experiment": run.name,
            "Type": run.experiment_type,
            "Completed": "✓" if run.completed else "✗",
            "Timestamp": run.timestamp or "",
        }
        
        # Add metrics if available
        if run.metrics:
            row["ROC-AUC"] = run.metrics.get("roc_auc", "")
            row["PR-AUC"] = run.metrics.get("pr_auc", "")
            row["F1"] = run.metrics.get("f1", "")
            row["ECE"] = run.metrics.get("ece", "")
            row["Spec@95%Sens"] = run.metrics.get("specificity_at_target_sens", 
                                                   run.metrics.get("specificity_at_95sens", ""))
        else:
            row["ROC-AUC"] = ""
            row["PR-AUC"] = ""
            row["F1"] = ""
            row["ECE"] = ""
            row["Spec@95%Sens"] = ""
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, runs: List[ExperimentStatus]) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    
    # Group by type
    for exp_type in ["teacher", "student", "quantization"]:
        type_df = df[df["Type"] == exp_type]
        if len(type_df) == 0:
            continue
        
        print(f"\n{'─' * 90}")
        print(f"  {exp_type.upper()} EXPERIMENTS")
        print(f"{'─' * 90}")
        
        # Print table
        print(type_df.to_string(index=False))
    
    # Summary statistics
    completed = sum(1 for r in runs if r.completed)
    total = len(runs)
    
    print(f"\n{'=' * 90}")
    print(f"PROGRESS: {completed}/{total} experiments completed ({100*completed/total:.0f}%)")
    print("=" * 90)
    
    # Missing experiments
    missing = [r for r in runs if not r.completed]
    if missing:
        print("\n⚠️  MISSING EXPERIMENTS:")
        print("-" * 50)
        for r in missing:
            print(f"  • {r.name} ({r.experiment_type})")
        
        print("\n📋 COMMANDS TO RUN MISSING EXPERIMENTS:")
        print("-" * 50)
        
        for r in missing:
            if r.experiment_type == "teacher":
                parts = r.name.replace("teacher_", "").split("_")
                arch = parts[0]
                loss = parts[1] if len(parts) > 1 else "focal"
                print(f"""
python scripts/train_teacher.py \\
    --architecture {arch} \\
    --loss {loss} \\
    --epochs 50 \\
    --seed 42
""")
            
            elif r.experiment_type == "student":
                # Parse T and alpha from name like "student_T2.0_alpha0.5"
                parts = r.name.replace("student_T", "").split("_alpha")
                temp = parts[0]
                alpha = parts[1]
                print(f"""
python scripts/train_student.py \\
    --teacher-ckpt models/checkpoints/teacher_resnet34_focal_best.pth \\
    --temperature {temp} \\
    --alpha {alpha} \\
    --epochs 50 \\
    --seed 42
""")
            
            elif r.experiment_type == "quantization":
                method = r.name.replace("quantized_", "")
                print(f"""
python scripts/quantize_model.py \\
    --model-ckpt models/checkpoints/student_T2.0_alpha0.5_best.pth \\
    --method {method}
""")
    else:
        print("\n✅ ALL EXPERIMENTS COMPLETED!")
    
    # Best results
    print("\n" + "=" * 90)
    print("BEST RESULTS")
    print("=" * 90)
    
    # Best teacher
    teacher_runs = [r for r in runs if r.experiment_type == "teacher" and r.metrics]
    if teacher_runs:
        best_teacher = max(teacher_runs, key=lambda r: r.metrics.get("roc_auc", 0))
        print(f"\n🏆 Best Teacher: {best_teacher.name}")
        print(f"   ROC-AUC: {best_teacher.metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"   ECE: {best_teacher.metrics.get('ece', 'N/A'):.4f}")
    
    # Best student
    student_runs = [r for r in runs if r.experiment_type == "student" and r.metrics]
    if student_runs:
        best_student = max(student_runs, key=lambda r: r.metrics.get("roc_auc", 0))
        print(f"\n🏆 Best Student (KD): {best_student.name}")
        print(f"   ROC-AUC: {best_student.metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"   ECE: {best_student.metrics.get('ece', 'N/A'):.4f}")
        
        # Compare to teacher
        if teacher_runs:
            teacher_auc = best_teacher.metrics.get("roc_auc", 0)
            student_auc = best_student.metrics.get("roc_auc", 0)
            delta = student_auc - teacher_auc
            print(f"   Δ ROC-AUC vs Teacher: {delta:+.4f}")


def generate_markdown_report(df: pd.DataFrame, runs: List[ExperimentStatus], output_path: pathlib.Path) -> None:
    """Generate a Markdown report file."""
    with open(output_path, "w") as f:
        f.write("# Experiment Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Progress
        completed = sum(1 for r in runs if r.completed)
        total = len(runs)
        f.write(f"## Progress: {completed}/{total} ({100*completed/total:.0f}%)\n\n")
        
        # Tables by type
        for exp_type in ["teacher", "student", "quantization"]:
            type_df = df[df["Type"] == exp_type]
            if len(type_df) == 0:
                continue
            
            f.write(f"### {exp_type.title()} Experiments\n\n")
            f.write(type_df.to_markdown(index=False))
            f.write("\n\n")
        
        # Missing experiments
        missing = [r for r in runs if not r.completed]
        if missing:
            f.write("## Missing Experiments\n\n")
            for r in missing:
                f.write(f"- [ ] {r.name} ({r.experiment_type})\n")
            f.write("\n")
        
        # Best results
        f.write("## Best Results\n\n")
        
        student_runs = [r for r in runs if r.experiment_type == "student" and r.metrics]
        if student_runs:
            best = max(student_runs, key=lambda r: r.metrics.get("roc_auc", 0))
            f.write(f"**Best Student Configuration**: {best.name}\n")
            f.write(f"- ROC-AUC: {best.metrics.get('roc_auc', 'N/A'):.4f}\n")
            f.write(f"- PR-AUC: {best.metrics.get('pr_auc', 'N/A'):.4f}\n")
            f.write(f"- ECE: {best.metrics.get('ece', 'N/A'):.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize training runs")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: artifacts/tbls/run_summary.csv)",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Output JSON path for programmatic access",
    )
    
    args = parser.parse_args()
    
    # Find all runs
    logger.info("Scanning for completed experiments...")
    
    teacher_runs = find_teacher_runs()
    student_runs = find_student_runs()
    quant_runs = find_quantization_runs()
    
    all_runs = teacher_runs + student_runs + quant_runs
    
    # Generate summary
    df = generate_summary_table(all_runs)
    
    # Print to console
    print_summary(df, all_runs)
    
    # Save CSV
    csv_path = pathlib.Path(args.output) if args.output else TABLES_DIR / "run_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")
    
    # Save Markdown if requested
    if args.markdown:
        md_path = pathlib.Path(args.markdown)
        generate_markdown_report(df, all_runs, md_path)
        logger.info(f"Saved Markdown report to {md_path}")
    
    # Save JSON if requested
    if args.json:
        json_path = pathlib.Path(args.json)
        json_data = {
            "generated": datetime.now().isoformat(),
            "experiments": [
                {
                    "name": r.name,
                    "type": r.experiment_type,
                    "completed": r.completed,
                    "metrics": r.metrics,
                }
                for r in all_runs
            ],
            "summary": {
                "total": len(all_runs),
                "completed": sum(1 for r in all_runs if r.completed),
                "missing": [r.name for r in all_runs if not r.completed],
            },
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved JSON summary to {json_path}")
    
    # Return exit code based on completion
    missing = sum(1 for r in all_runs if not r.completed)
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
