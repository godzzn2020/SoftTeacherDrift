"""Phase C3：严重度调度 vs baseline 的合成流消融。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from soft_drift.utils.run_paths import create_experiment_run, resolve_log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C3：scheduler 消融（合成流）")
    parser.add_argument(
        "--logs_root",
        type=str,
        default="logs",
        help="训练日志根目录（logs/{dataset}/{dataset}__{model}__seed{seed}.csv）",
    )
    parser.add_argument(
        "--synthetic_root",
        type=str,
        default="data/synthetic",
        help="合成流 meta 根目录",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="逗号分隔的数据集名称，例如 sea_abrupt4,sine_abrupt4",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5",
        help="逗号或空格分隔的随机种子列表",
    )
    parser.add_argument(
        "--model_variants",
        type=str,
        default="ts_drift_adapt,ts_drift_adapt_severity",
        help="逗号分隔的模型变体列表",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="漂移后计算 drop_min_acc 的窗口（样本数）",
    )
    parser.add_argument(
        "--final_window",
        type=int,
        default=200,
        help="final accuracy 所使用的尾部窗口（步数/批次数）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phaseC_scheduler_ablation_synth",
        help="评估结果根目录（内部会再创建 run_id 子目录）",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="phaseC_scheduler_ablation_synth",
        help="评估实验名称，用于目录分组",
    )
    parser.add_argument("--run_name", type=str, default=None, help="评估 run 的别名")
    parser.add_argument("--run_id", type=str, default=None, help="覆盖自动生成的评估 run_id")
    parser.add_argument(
        "--log_experiment",
        type=str,
        default="stage1_multi_seed",
        help="训练日志所属实验名称，用于推断新目录结构",
    )
    parser.add_argument(
        "--log_run_id",
        type=str,
        default=None,
        help="训练 run_id（若缺省则使用最新目录或旧版路径）",
    )
    return parser.parse_args()


def parse_int_list(spec: str) -> List[int]:
    tokens = spec.replace(",", " ").split()
    return [int(tok) for tok in tokens if tok.strip()]


def parse_str_list(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_run_info(csv_path: Path) -> Tuple[str, str, str]:
    parts = csv_path.stem.split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2].replace("seed", "")
    return csv_path.parent.name, "unknown", "0"


def infer_x_col(df: pd.DataFrame) -> str:
    if "sample_idx" in df.columns:
        return "sample_idx"
    if "seen_samples" in df.columns:
        return "seen_samples"
    return "step"


def load_meta(dataset: str, seed: int, synthetic_root: Path) -> Tuple[List[int], Optional[Path]]:
    meta_path = synthetic_root / dataset / f"{dataset}__seed{seed}_meta.json"
    if not meta_path.exists():
        return [], None
    meta = json.loads(meta_path.read_text())
    drifts: List[int] = []
    if isinstance(meta, dict):
        if "drifts" in meta:
            for item in meta["drifts"]:
                if isinstance(item, dict):
                    start = item.get("start") or item.get("step")
                else:
                    start = item
                if start is not None:
                    drifts.append(int(start))
        elif "concept_segments" in meta:
            for seg in meta["concept_segments"][1:]:
                start = seg.get("start")
                if start is not None:
                    drifts.append(int(start))
    return sorted(drifts), meta_path


def compute_drop_stats(
    df: pd.DataFrame,
    drifts: Sequence[int],
    x_col: str,
    window: int,
) -> Tuple[float, float, int]:
    if not drifts or "metric_accuracy" not in df:
        return float("nan"), float("nan"), 0
    x = df[x_col].to_numpy()
    acc = df["metric_accuracy"].to_numpy()
    drops: List[float] = []
    for drift_step in drifts:
        pre_mask = (x >= drift_step - window) & (x < drift_step)
        post_mask = (x >= drift_step) & (x < drift_step + window)
        if not pre_mask.any() or not post_mask.any():
            continue
        pre_mean = float(acc[pre_mask].mean())
        post_min = float(acc[post_mask].min())
        drops.append(pre_mean - post_min)
    if not drops:
        return float("nan"), float("nan"), 0
    drops_arr = np.asarray(drops, dtype=np.float64)
    return float(drops_arr.mean()), float(drops_arr.max()), len(drops)


def summarize_run(
    df: pd.DataFrame,
    dataset: str,
    model_variant: str,
    seed: int,
    drifts: Sequence[int],
    window: int,
    final_window: int,
) -> Dict[str, float]:
    x_col = infer_x_col(df)
    mean_acc = float(df["metric_accuracy"].mean()) if "metric_accuracy" in df else float("nan")
    if "metric_accuracy" in df and len(df) > 0:
        tail = df["metric_accuracy"].tail(final_window)
        final_acc = float(tail.mean())
    else:
        final_acc = float("nan")
    mean_drop, max_drop, valid_drifts = compute_drop_stats(df, drifts, x_col, window)
    severity_series = df.loc[df["drift_flag"] == 1, "drift_severity"] if "drift_severity" in df else pd.Series(dtype=float)
    mean_severity = float(severity_series.mean()) if not severity_series.empty else float("nan")
    return {
        "dataset_name": dataset,
        "model_variant": model_variant,
        "seed": seed,
        "mean_acc": mean_acc,
        "final_acc": final_acc,
        "mean_drop_min_acc": mean_drop,
        "max_drop_min_acc": max_drop,
        "mean_drift_severity": mean_severity,
        "n_valid_drifts": valid_drifts,
        "n_true_drifts": len(drifts),
    }


def summarize_group(run_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    grouped = run_df.groupby(["dataset_name", "model_variant"])
    for (dataset, variant), group in grouped:
        stats = {
            "dataset_name": dataset,
            "model_variant": variant,
        }
        for col in ["mean_acc", "final_acc", "mean_drop_min_acc", "max_drop_min_acc", "mean_drift_severity"]:
            values = group[col].astype(float)
            stats[f"{col}_mean"] = float(values.mean())
            stats[f"{col}_std"] = float(values.std(ddof=0))
        stats["runs"] = int(len(group))
        records.append(stats)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    model_variants = parse_str_list(args.model_variants)
    synth_root = Path(args.synthetic_root)
    experiment_run = create_experiment_run(
        experiment_name=args.experiment_name,
        results_root=args.output_dir,
        logs_root=args.logs_root,
        run_name=args.run_name,
        run_id=args.run_id,
    )
    output_dir = experiment_run.summary_dir()
    print(f"[run] {experiment_run.describe()} -> {output_dir}")
    run_records: List[Dict[str, float]] = []

    for dataset in datasets:
        for model_variant in model_variants:
            for seed in seeds:
                log_path = resolve_log_path(
                    logs_root=args.logs_root,
                    experiment_name=args.log_experiment,
                    dataset_name=dataset,
                    model_variant=model_variant,
                    seed=seed,
                    run_id=args.log_run_id,
                )
                if not log_path:
                    print(f"[warn] missing log for dataset={dataset} model={model_variant} seed={seed}, skip")
                    continue
                df = pd.read_csv(log_path)
                if df.empty:
                    print(f"[warn] empty log {log_path}, skip")
                    continue
                drifts, _ = load_meta(dataset, seed, synth_root)
                if not drifts:
                    print(f"[warn] missing meta or drifts for {dataset} seed={seed}")
                record = summarize_run(
                    df=df.sort_values(by="step").reset_index(drop=True),
                    dataset=dataset,
                    model_variant=model_variant,
                    seed=seed,
                    drifts=drifts,
                    window=args.window,
                    final_window=args.final_window,
                )
                run_records.append(record)
                print(
                    f"[run] dataset={dataset} model={model_variant} seed={seed} "
                    f"mean_drop={record['mean_drop_min_acc']:.4f} final_acc={record['final_acc']:.4f}"
                )

    if not run_records:
        print("[warn] no runs processed")
        return
    run_df = pd.DataFrame(run_records)
    run_path = output_dir / "run_level_metrics.csv"
    run_df.to_csv(run_path, index=False)
    summary_df = summarize_group(run_df)
    summary_path = output_dir / "summary_metrics_by_dataset_variant.csv"
    summary_df.to_csv(summary_path, index=False)

    for _, row in summary_df.iterrows():
        print(
            f"[info] dataset={row['dataset_name']} variant={row['model_variant']}: "
            f"final_acc={row['final_acc_mean']:.4f}±{row['final_acc_std']:.4f}, "
            f"mean_drop={row['mean_drop_min_acc_mean']:.4f}"
        )
    print(f"[done] run-level metrics saved to {run_path}")
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
