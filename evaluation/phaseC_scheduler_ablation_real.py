"""Phase C3：真实流 ts_drift_adapt vs ts_drift_adapt_severity 调度消融。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from soft_drift.utils.run_paths import create_experiment_run, resolve_log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C3：真实流调度消融评估")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Electricity,NOAA,INSECTS_abrupt_balanced,Airlines",
        help="逗号分隔的真实数据集名称",
    )
    parser.add_argument(
        "--model_variants",
        type=str,
        default="ts_drift_adapt,ts_drift_adapt_severity",
        help="逗号分隔的模型名称",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="逗号或空格分隔的随机种子列表",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="漂移后窗口大小（按 sample_idx/seen_samples）",
    )
    parser.add_argument(
        "--final_window",
        type=int,
        default=200,
        help="final accuracy 的尾部窗口（按行数）",
    )
    parser.add_argument(
        "--min_separation",
        type=int,
        default=200,
        help="将检测事件合并为漂移的最小间隔（同样使用 sample 轴）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phaseC_scheduler_ablation_real",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="phaseC_scheduler_ablation_real",
        help="评估 run 的 experiment_name",
    )
    parser.add_argument("--run_name", type=str, default=None, help="评估 run 的别名")
    parser.add_argument("--run_id", type=str, default=None, help="覆盖评估 run_id")
    parser.add_argument(
        "--log_experiment",
        type=str,
        default="run_real_adaptive",
        help="训练日志所属实验名称",
    )
    parser.add_argument(
        "--log_run_id",
        type=str,
        default=None,
        help="训练 run_id（缺省时尝试使用最新或旧版路径）",
    )
    return parser.parse_args()


def parse_int_list(spec: str) -> List[int]:
    tokens = spec.replace(",", " ").split()
    return [int(tok) for tok in tokens if tok.strip()]


def parse_str_list(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_x_col(df: pd.DataFrame) -> str:
    if "sample_idx" in df.columns:
        return "sample_idx"
    if "seen_samples" in df.columns:
        return "seen_samples"
    return "step"


def load_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    return df.sort_values(by="step").reset_index(drop=True)


def dedup_drift_events(values: Sequence[float], min_sep: int) -> List[int]:
    if not values:
        return []
    deduped: List[int] = []
    for v in values:
        if not deduped or v - deduped[-1] >= min_sep:
            deduped.append(int(v))
    return deduped


def extract_detection_drifts(df: pd.DataFrame, x_col: str, min_sep: int) -> List[int]:
    if "drift_flag" not in df:
        return []
    events = df.loc[df["drift_flag"] == 1, x_col].tolist()
    if not events:
        return []
    events.sort()
    return dedup_drift_events(events, min_sep=min_sep)


def compute_drop_stats(df: pd.DataFrame, drifts: Sequence[int], x_col: str, window: int) -> Dict[str, float]:
    if not drifts or "metric_accuracy" not in df:
        return {"mean_drop": float("nan"), "max_drop": float("nan"), "count": 0}
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
        return {"mean_drop": float("nan"), "max_drop": float("nan"), "count": 0}
    arr = np.asarray(drops, dtype=np.float64)
    return {"mean_drop": float(arr.mean()), "max_drop": float(arr.max()), "count": len(arr)}


def summarize_run(
    df: pd.DataFrame,
    dataset: str,
    model_variant: str,
    seed: int,
    window: int,
    final_window: int,
    min_separation: int,
) -> Dict[str, float]:
    x_col = infer_x_col(df)
    drifts = extract_detection_drifts(df, x_col, min_separation)
    drops = compute_drop_stats(df, drifts, x_col, window)
    mean_acc = float(df["metric_accuracy"].mean()) if "metric_accuracy" in df else float("nan")
    if "metric_accuracy" in df and len(df) > 0:
        final_acc = float(df["metric_accuracy"].tail(final_window).mean())
    else:
        final_acc = float("nan")
    severity_series = df.loc[df["drift_flag"] == 1, "drift_severity"] if "drift_severity" in df else pd.Series(dtype=float)
    return {
        "dataset_name": dataset,
        "model_variant": model_variant,
        "seed": seed,
        "mean_acc": mean_acc,
        "final_acc": final_acc,
        "mean_drop_min_acc": drops["mean_drop"],
        "max_drop_min_acc": drops["max_drop"],
        "n_detected_events": len(drifts),
        "n_drop_windows": drops["count"],
        "mean_drift_severity": float(severity_series.mean()) if not severity_series.empty else float("nan"),
    }


def summarize_group(run_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    grouped = run_df.groupby(["dataset_name", "model_variant"])
    for (dataset, variant), group in grouped:
        stats: Dict[str, float] = {"dataset_name": dataset, "model_variant": variant, "runs": float(len(group))}
        for col in ["mean_acc", "final_acc", "mean_drop_min_acc", "max_drop_min_acc", "mean_drift_severity"]:
            values = group[col].astype(float)
            stats[f"{col}_mean"] = float(values.mean())
            stats[f"{col}_std"] = float(values.std(ddof=0))
        records.append(stats)
    return pd.DataFrame(records)


def build_pairwise_diff(
    summary_df: pd.DataFrame,
    baseline_variant: str = "ts_drift_adapt",
    severity_keyword: str = "ts_drift_adapt_severity",
) -> pd.DataFrame:
    if summary_df.empty or "model_variant" not in summary_df.columns:
        return pd.DataFrame()
    baseline_df = summary_df.loc[summary_df["model_variant"] == baseline_variant].set_index("dataset_name")
    severity_rows = summary_df.loc[summary_df["model_variant"].str.contains(severity_keyword, na=False)].copy()
    if severity_rows.empty or baseline_df.empty:
        return pd.DataFrame()
    records: List[Dict[str, float]] = []
    for _, row in severity_rows.iterrows():
        dataset = row["dataset_name"]
        if dataset not in baseline_df.index:
            continue
        base_row = baseline_df.loc[dataset]
        record = {
            "dataset_name": dataset,
            "baseline_variant": baseline_variant,
            "severity_variant": row["model_variant"],
            "baseline_mean_acc": float(base_row.get("mean_acc_mean", float("nan"))),
            "severity_mean_acc": float(row.get("mean_acc_mean", float("nan"))),
            "baseline_final_acc": float(base_row.get("final_acc_mean", float("nan"))),
            "severity_final_acc": float(row.get("final_acc_mean", float("nan"))),
            "baseline_mean_drop_min_acc": float(base_row.get("mean_drop_min_acc_mean", float("nan"))),
            "severity_mean_drop_min_acc": float(row.get("mean_drop_min_acc_mean", float("nan"))),
            "baseline_max_drop_min_acc": float(base_row.get("max_drop_min_acc_mean", float("nan"))),
            "severity_max_drop_min_acc": float(row.get("max_drop_min_acc_mean", float("nan"))),
        }
        record["delta_mean_acc"] = record["severity_mean_acc"] - record["baseline_mean_acc"]
        record["delta_final_acc"] = record["severity_final_acc"] - record["baseline_final_acc"]
        record["delta_mean_drop_min_acc"] = (
            record["severity_mean_drop_min_acc"] - record["baseline_mean_drop_min_acc"]
        )
        record["delta_max_drop_min_acc"] = (
            record["severity_max_drop_min_acc"] - record["baseline_max_drop_min_acc"]
        )
        records.append(record)
    return pd.DataFrame(records)


def write_markdown_summary(diff_df: pd.DataFrame, output_path: Path) -> None:
    if diff_df.empty:
        return
    lines = ["# Phase C3 Real-Stream Severity Sweep", "", "| dataset | severity_variant | baseline_final | severity_final | Δfinal | Δmean | Δmean_drop | Δmax_drop |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, row in diff_df.iterrows():
        lines.append(
            "| {dataset} | {variant} | {base_final:.4f} | {sev_final:.4f} | {delta_final:.4f} | {delta_mean:.4f} | {delta_mean_drop:.4f} | {delta_max_drop:.4f} |".format(
                dataset=row["dataset_name"],
                variant=row["severity_variant"],
                base_final=row.get("baseline_final_acc", float("nan")),
                sev_final=row.get("severity_final_acc", float("nan")),
                delta_final=row.get("delta_final_acc", float("nan")),
                delta_mean=row.get("delta_mean_acc", float("nan")),
                delta_mean_drop=row.get("delta_mean_drop_min_acc", float("nan")),
                delta_max_drop=row.get("delta_max_drop_min_acc", float("nan")),
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = parse_str_list(args.datasets)
    model_variants = parse_str_list(args.model_variants)
    seeds = parse_int_list(args.seeds)
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
        for variant in model_variants:
            for seed in seeds:
                log_path = resolve_log_path(
                    logs_root=args.logs_root,
                    experiment_name=args.log_experiment,
                    dataset_name=dataset,
                    model_variant=variant,
                    seed=seed,
                    run_id=args.log_run_id,
                )
                if not log_path:
                    print(f"[warn] missing log dataset={dataset} variant={variant} seed={seed}, skip")
                    continue
                df = load_log(Path(log_path))
                if df.empty:
                    print(f"[warn] empty log {log_path}, skip")
                    continue
                record = summarize_run(
                    df=df,
                    dataset=dataset,
                    model_variant=variant,
                    seed=seed,
                    window=args.window,
                    final_window=args.final_window,
                    min_separation=args.min_separation,
                )
                run_records.append(record)
                print(
                    f"[run] dataset={dataset} model={variant} seed={seed} "
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

    diff_df = build_pairwise_diff(summary_df)
    diff_path = output_dir / "summary_metrics_by_dataset_pairwise_diff.csv"
    if not diff_df.empty:
        diff_df.to_csv(diff_path, index=False)
        write_markdown_summary(diff_df, output_dir / "phaseC_scheduler_real_summary.md")

    for _, row in summary_df.iterrows():
        print(
            f"[info] dataset={row['dataset_name']} model={row['model_variant']}: "
            f"final_acc={row['final_acc_mean']:.4f} "
            f"mean_drop={row['mean_drop_min_acc_mean']:.4f}"
        )
    print(f"[done] run-level metrics saved to {run_path}")
    print(f"[done] summary metrics saved to {summary_path}")
    if not diff_df.empty:
        print(f"[done] pairwise diffs saved to {diff_path}")


if __name__ == "__main__":
    main()
