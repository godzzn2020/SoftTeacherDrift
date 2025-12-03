"""合成流三信号检测消融评估脚本。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from drift.detectors import build_default_monitor


DEFAULT_PRESETS = [
    "error_only_ph_meta",
    "entropy_only_ph_meta",
    "divergence_only_ph_meta",
    "error_entropy_ph_meta",
    "error_divergence_ph_meta",
    "entropy_divergence_ph_meta",
    "all_signals_ph_meta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合成流信号组合漂移检测消融评估")
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
        help="仅保留文件名中包含此字符串的 model_variant",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=DEFAULT_PRESETS,
        help="monitor preset 名称列表，可空格或逗号分隔",
    )
    parser.add_argument("--output_dir", type=str, default="results/phaseB_ablation_synth")
    parser.add_argument(
        "--match_tolerance",
        type=int,
        default=500,
        help="将检测事件与真值漂移匹配时允许的最大延迟（样本数）",
    )
    parser.add_argument(
        "--min_separation",
        type=int,
        default=200,
        help="若两次检测间隔小于该值（样本数），视为同一次报警",
    )
    return parser.parse_args()


def ensure_list(value: object) -> List[str]:
    items: List[str] = []
    if isinstance(value, (list, tuple)):
        sources = value
    else:
        sources = [value]
    for src in sources:
        if isinstance(src, str):
            items.extend([item.strip() for item in src.split(",") if item.strip()])
    return items


def find_logs(logs_root: Path, dataset: str, pattern: str) -> List[Path]:
    dataset_dir = logs_root / dataset
    if not dataset_dir.exists():
        print(f"[warn] dataset log dir missing: {dataset_dir}")
        return []
    matches: List[Path] = []
    for csv in dataset_dir.glob(f"{dataset}__*__seed*.csv"):
        if pattern and pattern not in csv.stem:
            continue
        matches.append(csv)
    return sorted(matches)


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


def load_synth_meta(meta_path: Path) -> Tuple[List[int], int]:
    meta = json.loads(meta_path.read_text())
    drifts: List[int] = []
    if "drifts" in meta:
        drifts = [int(item.get("start", item.get("step", 0))) for item in meta["drifts"]]
    elif "concept_segments" in meta:
        segments = meta["concept_segments"]
        for seg in segments[1:]:
            drifts.append(int(seg.get("start", 0)))
    drifts = sorted(set(drifts))
    horizon = int(meta.get("n_samples", 0))
    return drifts, horizon


def infer_x_col(df: pd.DataFrame) -> str:
    if "sample_idx" in df.columns:
        return "sample_idx"
    if "seen_samples" in df.columns:
        return "seen_samples"
    return "step"


def build_signal_dict(row: pd.Series) -> Dict[str, float]:
    return {
        "error_rate": float(row.get("student_error_rate", float("nan"))),
        "teacher_entropy": float(row.get("teacher_entropy", float("nan"))),
        "divergence": float(row.get("divergence_js", float("nan"))),
    }


def run_monitor(df: pd.DataFrame, preset: str, x_col: str) -> List[float]:
    monitor = build_default_monitor(preset)
    detections: List[float] = []
    x_values = df[x_col].to_numpy()
    steps = df["step"].to_numpy()
    for idx, row in df.iterrows():
        signals = build_signal_dict(row)
        drift_flag, _ = monitor.update(signals, int(steps[idx]))
        if drift_flag:
            detections.append(float(x_values[idx]))
    return detections


def merge_detections(events: Sequence[float], min_sep: int) -> List[float]:
    if not events:
        return []
    events = sorted(events)
    merged = [events[0]]
    for det in events[1:]:
        if det - merged[-1] < min_sep:
            continue
        merged.append(det)
    return merged


def compute_metrics(
    drifts: Sequence[int],
    detections: Sequence[float],
    horizon: int,
    tolerance: int,
) -> Dict[str, float]:
    drifts = sorted(int(d) for d in drifts)
    detections = sorted(float(d) for d in detections)
    matched: set[int] = set()
    delays: List[float] = []
    missed = 0
    for drift in drifts:
        match_idx: Optional[int] = None
        for idx, det in enumerate(detections):
            if idx in matched:
                continue
            if det < drift:
                continue
            if det <= drift + tolerance:
                match_idx = idx
                break
            if det > drift + tolerance:
                break
        if match_idx is not None:
            matched.add(match_idx)
            delays.append(detections[match_idx] - drift)
        else:
            missed += 1
    false_detections = [detections[i] for i in range(len(detections)) if i not in matched]
    mdr = missed / len(drifts) if drifts else math.nan
    mtd = float(np.mean(delays)) if delays else math.nan
    if len(false_detections) >= 2:
        gaps = [b - a for a, b in zip(false_detections[:-1], false_detections[1:])]
        mtfa = float(np.mean(gaps))
    elif len(false_detections) == 1:
        mtfa = float(horizon - false_detections[0]) if horizon > 0 else float("nan")
    else:
        mtfa = math.nan
    if (
        delays
        and not math.isnan(mtd)
        and not math.isnan(mtfa)
        and mdr is not None
        and mdr < 1.0
        and mtd > 0
    ):
        mtr = mtfa / (mtd * (1 - mdr))
    else:
        mtr = math.nan
    return {
        "MDR": mdr,
        "MTD": mtd,
        "MTFA": mtfa,
        "MTR": mtr,
        "n_true_drifts": len(drifts),
        "n_detected": len(delays),
        "n_false_alarms": len(false_detections),
        "n_missed_drifts": missed,
    }


def main() -> None:
    args = parse_args()
    datasets = ensure_list(args.datasets)
    presets = ensure_list(args.presets)
    logs_root = Path(args.logs_root)
    synth_root = Path(args.synthetic_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_records: List[Dict[str, object]] = []

    for dataset in datasets:
        log_paths = find_logs(logs_root, dataset, args.model_variant_pattern)
        if not log_paths:
            print(f"[warn] no logs found for dataset={dataset}")
            continue
        for log_path in log_paths:
            dataset_name, model_variant, seed = extract_run_info(log_path)
            meta_path = synth_root / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
            if not meta_path.exists():
                print(f"[warn] missing meta {meta_path}, skip {log_path.name}")
                continue
            drifts, horizon = load_synth_meta(meta_path)
            if not drifts:
                print(f"[warn] meta for {dataset_name} seed={seed} 无漂移记录，跳过")
                continue
            df = load_log(log_path)
            if df.empty:
                continue
            x_col = infer_x_col(df)
            max_x = int(df[x_col].iloc[-1]) + 1 if horizon <= 0 else horizon
            for preset in presets:
                detections = run_monitor(df, preset, x_col)
                detections = merge_detections(detections, args.min_separation)
                metrics = compute_metrics(drifts, detections, max_x, args.match_tolerance)
                record = {
                    "dataset_name": dataset_name,
                    "model_variant": model_variant,
                    "preset": preset,
                    "seed": seed,
                    **metrics,
                }
                run_records.append(record)

    if not run_records:
        print("[warn] 没有生成任何 run 级指标，请确认日志与 meta 是否存在。")
        return

    run_df = pd.DataFrame(run_records)
    run_csv = output_dir / "run_level_metrics.csv"
    run_df.to_csv(run_csv, index=False)
    agg_dict = {}
    metrics = ["MDR", "MTD", "MTFA", "MTR", "n_detected", "n_false_alarms", "n_missed_drifts"]
    for m in metrics:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    summary_df = run_df.groupby(["dataset_name", "preset"]).agg(**agg_dict).reset_index()
    summary_csv = output_dir / "summary_metrics_by_dataset_preset.csv"
    summary_df.to_csv(summary_csv, index=False)
    for _, row in summary_df.iterrows():
        print(
            f"[info] dataset={row['dataset_name']}, preset={row['preset']}: "
            f"MDR={row['MDR_mean']:.3f}±{row['MDR_std']:.3f}, "
            f"MTD={row['MTD_mean']:.1f}, MTFA={row['MTFA_mean']:.1f}"
        )
    print(f"[done] run-level metrics: {run_csv}")
    print(f"[done] summary metrics: {summary_csv}")


if __name__ == "__main__":
    main()
