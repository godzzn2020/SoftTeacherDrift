"""合成流三信号与真值漂移对齐分析脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合成流漂移信号 vs 真值分析")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--synthetic_root", type=str, default="data/synthetic")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="逗号分隔的数据集名称，例如 sea_abrupt4,sine_abrupt4",
    )
    parser.add_argument(
        "--model_variant_pattern",
        type=str,
        default="",
        help="可选，仅保留 model_variant 名称中包含该子串的 run",
    )
    parser.add_argument("--output_dir", type=str, default="results/phaseA_synth_analysis")
    parser.add_argument("--window", type=int, default=50, help="pre/post 统计窗口大小（以 step 计）")
    return parser.parse_args()


def find_logs(
    logs_root: Path,
    dataset: str,
    model_pattern: str,
) -> List[Path]:
    dataset_dir = logs_root / dataset
    if not dataset_dir.exists():
        print(f"[warn] dataset logs not found: {dataset_dir}")
        return []
    matches: List[Path] = []
    for csv_path in dataset_dir.glob(f"{dataset}__*__seed*.csv"):
        if model_pattern and model_pattern not in csv_path.stem:
            continue
        matches.append(csv_path)
    return sorted(matches)


def load_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values(by="step").reset_index(drop=True)


def parse_meta(meta_path: Path) -> Tuple[List[int], Dict[str, any]]:
    meta = json.loads(meta_path.read_text())
    drifts = []
    if isinstance(meta, dict):
        if "drifts" in meta:
            drifts = [int(item.get("start", item.get("step", item.get("t", 0)))) for item in meta["drifts"]]
        elif "concept_segments" in meta:
            segments = meta["concept_segments"]
            for seg in segments[1:]:
                drifts.append(int(seg.get("start", 0)))
        else:
            print(f"[warn] 未能在 meta 中找到 'drifts' 或 'concept_segments'，请检查字段: {meta_path}")
    else:
        print(f"[warn] meta 不是 dict 格式: {meta_path}")
    drifts = sorted(set(drifts))
    return drifts, meta


def extract_run_info(csv_path: Path) -> Tuple[str, str, str]:
    # logs/{dataset}/{dataset}__{model_variant}__seed{seed}.csv
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


def plot_run(
    df: pd.DataFrame,
    drifts: Sequence[int],
    run_id: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        ("student_error_rate", "Student Error Rate"),
        ("teacher_entropy", "Teacher Entropy"),
        ("divergence_js", "Teacher-Student JS"),
        ("metric_accuracy", "Accuracy"),
    ]
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 9), sharex=True)
    for ax, (col, title) in zip(axes, keys):
        if col not in df:
            ax.set_title(f"{title} (missing)")
            continue
        ax.plot(df["step"], df[col], label=col, linewidth=1.2)
        ax.set_ylabel(title)
        for d in drifts:
            ax.axvline(d, color="red", linestyle="--", alpha=0.5)
        if col == "metric_accuracy" and "drift_flag" in df:
            flagged = df.loc[df["drift_flag"] == 1, "step"].tolist()
            for f in flagged:
                ax.axvline(f, color="blue", linestyle=":", alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(run_id)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_id}.png", dpi=150)
    plt.close(fig)


def compute_pre_post_stats(
    df: pd.DataFrame,
    drifts: Sequence[int],
    dataset: str,
    model: str,
    seed: str,
    window: int,
) -> List[Dict[str, any]]:
    stats: List[Dict[str, any]] = []
    if not drifts or df.empty:
        return stats
    signals = ["student_error_rate", "teacher_entropy", "divergence_js", "metric_accuracy"]
    steps = df["step"].to_numpy()
    for idx, drift_step in enumerate(drifts):
        nearest_idx = (np.abs(steps - drift_step)).argmin()
        left = max(0, nearest_idx - window)
        right = min(len(df), nearest_idx + window)
        pre_slice = df.iloc[left:nearest_idx]
        post_slice = df.iloc[nearest_idx:right]
        for sig in signals:
            if sig not in df:
                continue
            pre_mean = float(pre_slice[sig].mean()) if not pre_slice.empty else float("nan")
            post_mean = float(post_slice[sig].mean()) if not post_slice.empty else float("nan")
            delta = post_mean - pre_mean if not (np.isnan(pre_mean) or np.isnan(post_mean)) else float("nan")
            stats.append(
                {
                    "dataset_name": dataset,
                    "model_variant": model,
                    "seed": seed,
                    "drift_index": idx,
                    "true_drift_step": int(drift_step),
                    "signal_name": sig,
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "delta": delta,
                }
            )
    return stats


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    logs_root = Path(args.logs_root)
    synth_root = Path(args.synthetic_root)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    stats_records: List[Dict[str, any]] = []
    total_runs = 0

    for dataset in datasets:
        log_paths = find_logs(logs_root, dataset, args.model_variant_pattern)
        if not log_paths:
            print(f"[warn] no logs found for {dataset}")
            continue
        for log_path in log_paths:
            dataset_name, model_variant, seed = extract_run_info(log_path)
            df = load_log(log_path)
            if df.empty:
                print(f"[warn] empty log for {log_path}")
                continue
            meta_path = synth_root / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
            if not meta_path.exists():
                print(f"[warn] missing meta ({meta_path}), skip run {log_path.name}")
                continue
            drifts, meta_raw = parse_meta(meta_path)
            if not drifts:
                print(f"[warn] run {log_path.name} 无真实漂移记录，跳过")
                continue
            run_id = f"{dataset_name}__{model_variant}__seed{seed}"
            print(f"[info] run={run_id}, drift_count={len(drifts)}")
            plot_run(df, drifts, run_id, plots_dir)
            stats_records.extend(
                compute_pre_post_stats(
                    df,
                    drifts,
                    dataset_name,
                    model_variant,
                    seed,
                    args.window,
                )
            )
            total_runs += 1

    if stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_path = output_dir / "summary_pre_post_stats.csv"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"[done] 保存 pre/post summary: {stats_path}")
    else:
        print("[warn] 未生成任何统计结果")
    print(f"[done] total processed runs = {total_runs}")


if __name__ == "__main__":
    main()
