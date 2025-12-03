"""真实流信号与检测事件可视化脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实数据集信号与检测事件分析")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="逗号分隔的真实数据集名称（例如 Electricity,NOAA）",
    )
    parser.add_argument(
        "--model_variant_pattern",
        type=str,
        default="ts_drift_adapt",
        help="可选的 model_variant 模式匹配（默认只看 ts_drift_adapt）",
    )
    parser.add_argument("--output_dir", type=str, default="results/phaseB_real_analysis")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_x_col(df: pd.DataFrame) -> str:
    if "sample_idx" in df.columns:
        return "sample_idx"
    if "seen_samples" in df.columns:
        return "seen_samples"
    return "step"


def find_logs(logs_root: Path, dataset: str, pattern: str) -> List[Path]:
    dataset_dir = logs_root / dataset
    if not dataset_dir.exists():
        return []
    logs = []
    for csv in dataset_dir.glob(f"{dataset}__*__seed*.csv"):
        if pattern and pattern not in csv.stem:
            continue
        logs.append(csv)
    return sorted(logs)


def load_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df = df.sort_values(by="step").reset_index(drop=True)
    return df


def extract_run_info(csv_path: Path) -> tuple[str, str, str]:
    parts = csv_path.stem.split("__")
    if len(parts) >= 3:
        dataset = parts[0]
        model = parts[1]
        seed = parts[2].replace("seed", "")
    else:
        dataset = csv_path.parent.name
        model = "unknown"
        seed = "0"
    return dataset, model, seed


def plot_run(df: pd.DataFrame, run_id: str, drifts: Sequence[int], out_path: Path, x_col: str) -> None:
    keys = [
        ("student_error_rate", "Student Error Rate"),
        ("teacher_entropy", "Teacher Entropy"),
        ("divergence_js", "Teacher-Student JS"),
        ("metric_accuracy", "Accuracy"),
    ]
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 9), sharex=True)
    x = df[x_col].to_numpy()
    detections = df.loc[df["drift_flag"] == 1, x_col].tolist() if "drift_flag" in df else []
    for ax, (col, title) in zip(axes, keys):
        if col not in df:
            ax.set_title(f"{title} (missing)")
            continue
        ax.plot(x, df[col], label=col, linewidth=1.2)
        ax.set_ylabel(title)
        for d in detections:
            ax.axvline(d, color="blue", linestyle="--", alpha=0.4)
    axes[-1].set_xlabel(x_col)
    fig.suptitle(run_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_run(df: pd.DataFrame, dataset: str, model: str, seed: str) -> Dict[str, object]:
    detection_steps = (
        df.loc[df["drift_flag"] == 1, "step"].tolist() if "drift_flag" in df else []
    )
    severity_col = "monitor_severity" if "monitor_severity" in df else None
    if severity_col is None and "drift_severity" in df and "drift_severity_raw" in df:
        # 新日志中 drift_severity 表示严重度感知值，因此 detector severity 落在 monitor_severity。
        severity_col = "drift_severity_raw"
    elif severity_col is None and "drift_severity" in df:
        severity_col = "drift_severity"
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "total_steps": int(df["step"].iloc[-1]) if not df.empty else 0,
        "detected_events": len(detection_steps),
        "mean_severity": float(df[severity_col].mean()) if severity_col else float("nan"),
        "acc_final": float(df["metric_accuracy"].iloc[-1]) if "metric_accuracy" in df else float("nan"),
        "acc_mean": float(df["metric_accuracy"].mean()) if "metric_accuracy" in df else float("nan"),
        "detection_steps": detection_steps,
    }


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    records: List[Dict[str, object]] = []
    run_count = 0

    for dataset in datasets:
        logs = find_logs(logs_root, dataset, args.model_variant_pattern)
        if not logs:
            print(f"[warn] no logs found for dataset {dataset}")
            continue
        for log_path in logs:
            dataset_name, model_variant, seed = extract_run_info(log_path)
            df = load_log(log_path)
            if df.empty:
                print(f"[warn] empty log {log_path}")
                continue
            x_col = infer_x_col(df)
            run_id = f"{dataset_name}__{model_variant}__seed{seed}"
            out_path = plots_dir / f"{run_id}.png"
            plot_run(df, run_id, [], out_path, x_col=x_col)
            records.append(summarize_run(df, dataset_name, model_variant, seed))
            print(f"[info] saved plot: {out_path}")
            run_count += 1

    if records:
        summary_df = pd.DataFrame(records)
        summary_path = output_dir / "summary_detection_stats.csv"
        ensure_dir(summary_path.parent)
        summary_df.to_csv(summary_path, index=False)
        print(f"[done] summary saved to {summary_path}")
    else:
        print("[warn] no records collected")
    print(f"[done] total processed runs = {run_count}")


if __name__ == "__main__":
    main()
