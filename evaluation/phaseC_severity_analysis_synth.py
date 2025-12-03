"""合成流三信号严重度与性能掉落分析脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C：信号变化与性能掉落分析（合成流）")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--synthetic_root", type=str, default="data/synthetic")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="逗号分隔的数据集名称，例如 sea_abrupt4,sine_abrupt4,stagger_abrupt3",
    )
    parser.add_argument(
        "--model_variant_pattern",
        type=str,
        default="",
        help="仅保留 model_variant 名称中包含该子串的 run（例如 ts_drift_adapt）",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="漂移前后窗口大小（以样本数计，例如 500 表示前后各 500 个样本）",
    )
    parser.add_argument("--output_dir", type=str, default="results/phaseC_severity_analysis")
    return parser.parse_args()


def list_datasets(spec: str) -> List[str]:
    return [d.strip() for d in spec.split(",") if d.strip()]


def find_logs(logs_root: Path, dataset: str, model_pattern: str) -> List[Path]:
    dataset_dir = logs_root / dataset
    if not dataset_dir.exists():
        return []
    logs: List[Path] = []
    for csv_path in dataset_dir.glob(f"{dataset}__*__seed*.csv"):
        if model_pattern and model_pattern not in csv_path.stem:
            continue
        logs.append(csv_path)
    return sorted(logs)


def extract_run_info(csv_path: Path) -> Tuple[str, str, str]:
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


def load_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values(by="step").reset_index(drop=True)


def parse_meta(meta_path: Path) -> Tuple[List[int], Dict[str, Any]]:
    meta = json.loads(meta_path.read_text())
    drifts: List[int] = []
    if isinstance(meta, dict):
        if "drifts" in meta:
            drifts = [int(item.get("start", item.get("step", 0))) for item in meta["drifts"]]
        elif "concept_segments" in meta:
            segments = meta["concept_segments"]
            for seg in segments[1:]:
                drifts.append(int(seg.get("start", 0)))
    drifts = sorted(set(drifts))
    return drifts, meta


def choose_time_axis(df: pd.DataFrame) -> pd.Series:
    if "sample_idx" in df.columns:
        return df["sample_idx"]
    if "seen_samples" in df.columns:
        return df["seen_samples"]
    return df["step"]


def compute_drift_stats_for_run(
    df: pd.DataFrame,
    true_drifts: Sequence[int],
    dataset: str,
    model_variant: str,
    seed: str,
    window: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if df.empty or not true_drifts:
        return records
    time = choose_time_axis(df)
    t = time.to_numpy()
    signals = ["student_error_rate", "teacher_entropy", "divergence_js", "metric_accuracy"]
    for idx, drift_step in enumerate(true_drifts):
        pre_mask = (t >= drift_step - window) & (t < drift_step)
        post_mask = (t >= drift_step) & (t < drift_step + window)
        pre_slice = df.loc[pre_mask]
        post_slice = df.loc[post_mask]
        if pre_slice.empty or post_slice.empty:
            continue
        record: Dict[str, Any] = {
            "dataset_name": dataset,
            "model_variant": model_variant,
            "seed": seed,
            "drift_index": idx,
            "true_drift_step": int(drift_step),
            "window": window,
        }
        for sig in ["student_error_rate", "teacher_entropy", "divergence_js", "metric_accuracy"]:
            if sig not in df:
                record[f"{sig}_pre"] = float("nan")
                record[f"{sig}_post"] = float("nan")
                record[f"delta_{sig}"] = float("nan")
                continue
            pre_val = float(pre_slice[sig].mean())
            post_val = float(post_slice[sig].mean())
            record[f"{sig}_pre"] = pre_val
            record[f"{sig}_post"] = post_val
            record[f"delta_{sig}"] = post_val - pre_val
        if "metric_accuracy" in df:
            acc_pre_mean = float(pre_slice["metric_accuracy"].mean())
            acc_post_min = float(post_slice["metric_accuracy"].min())
            record["delta_acc"] = record.get("delta_metric_accuracy")
            record["drop_min_acc"] = acc_pre_mean - acc_post_min
        records.append(record)
    return records


def compute_correlations(per_drift_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    datasets = per_drift_df["dataset_name"].unique()
    for dataset in datasets:
        df = per_drift_df[per_drift_df["dataset_name"] == dataset].copy()
        usable = df.dropna(
            subset=[
                "delta_student_error_rate",
                "delta_teacher_entropy",
                "delta_divergence_js",
                "delta_metric_accuracy",
                "drop_min_acc",
            ]
        )
        if usable.empty:
            continue
        def corr(a: str, b: str) -> float:
            if usable[a].empty or usable[b].empty:
                return float("nan")
            return float(np.corrcoef(usable[a], usable[b])[0, 1])
        rows.append(
            {
                "dataset_name": dataset,
                "n_drifts_used": len(usable),
                "corr_delta_error__delta_acc": corr("delta_student_error_rate", "delta_metric_accuracy"),
                "corr_delta_entropy__delta_acc": corr("delta_teacher_entropy", "delta_metric_accuracy"),
                "corr_delta_divergence__delta_acc": corr("delta_divergence_js", "delta_metric_accuracy"),
                "corr_delta_error__drop_min_acc": corr("delta_student_error_rate", "drop_min_acc"),
                "corr_delta_entropy__drop_min_acc": corr("delta_teacher_entropy", "drop_min_acc"),
                "corr_delta_divergence__drop_min_acc": corr("delta_divergence_js", "drop_min_acc"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    datasets = list_datasets(args.datasets)
    logs_root = Path(args.logs_root)
    synth_root = Path(args.synthetic_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    for dataset in datasets:
        log_paths = find_logs(logs_root, dataset, args.model_variant_pattern)
        if not log_paths:
            print(f"[warn] no logs found for dataset={dataset}")
            continue
        for log_path in log_paths:
            dataset_name, model_variant, seed = extract_run_info(log_path)
            meta_path = synth_root / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
            if not meta_path.exists():
                print(f"[warn] missing meta ({meta_path}), skip run {log_path.name}")
                continue
            drifts, _ = parse_meta(meta_path)
            if not drifts:
                continue
            df = load_log(log_path)
            records = compute_drift_stats_for_run(
                df,
                drifts,
                dataset_name,
                model_variant,
                seed,
                args.window,
            )
            all_records.extend(records)
    if not all_records:
        print("[warn] no drift stats collected")
        return
    per_drift_df = pd.DataFrame(all_records)
    per_drift_path = output_dir / "per_drift_stats.csv"
    per_drift_df.to_csv(per_drift_path, index=False)
    corr_df = compute_correlations(per_drift_df)
    corr_path = output_dir / "severity_correlations_by_dataset.csv"
    corr_df.to_csv(corr_path, index=False)
    for _, row in corr_df.iterrows():
        print(
            f"[info] dataset={row['dataset_name']}: n_drifts={row['n_drifts_used']}, "
            f"corr(delta_entropy, delta_acc)={row['corr_delta_entropy__delta_acc']:.3f}, "
            f"corr(delta_divergence, delta_acc)={row['corr_delta_divergence__delta_acc']:.3f}"
        )
    print(f"[done] per-drift stats saved to {per_drift_path}")
    print(f"[done] correlation summary saved to {corr_path}")


if __name__ == "__main__":
    main()
