"""聚合在线实验日志，输出表格、检测指标与 SVG 图像。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.drift_metrics import compute_detection_metrics
from data.real_meta import load_insects_abrupt_balanced_meta


SYNTH_DATASETS = {"sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"}
REAL_DATASETS = {"INSECTS_abrupt_balanced"}

SYNTH_FALLBACK_META = {
    "sea_abrupt4": {
        "n_samples": 50_000,
        "drifts": [{"start": 10_000}, {"start": 20_000}, {"start": 30_000}, {"start": 40_000}],
    },
    "sine_abrupt4": {
        "n_samples": 50_000,
        "drifts": [{"start": 10_000}, {"start": 20_000}, {"start": 30_000}, {"start": 40_000}],
    },
    "stagger_abrupt3": {
        "n_samples": 60_000,
        "drifts": [{"start": 20_000}, {"start": 40_000}],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总在线实验结果（含漂移指标）")
    parser.add_argument(
        "--datasets",
        type=str,
        default="sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced",
        help="逗号分隔的数据集名称（或 all）",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="baseline_student,mean_teacher,ts_drift_adapt",
        help="逗号分隔的模型列表（或 all）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="需要统计的随机种子",
    )
    parser.add_argument("--logs_root", type=str, default="logs", help="日志根目录")
    parser.add_argument("--out_runs_csv", type=str, default="results/online_runs.csv")
    parser.add_argument("--out_summary_csv", type=str, default="results/online_summary.csv")
    parser.add_argument("--out_md", type=str, default="results/online_summary.md")
    parser.add_argument("--fig_dir", type=str, default="figures/online_accuracy")
    parser.add_argument(
        "--detection_fig_dir",
        type=str,
        default="figures/online_detections",
        help="漂移检测时间线 SVG 输出目录",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Markdown 中每个数据集保留的组合数")
    parser.add_argument(
        "--insects_meta",
        type=str,
        default="datasets/real/INSECTS_abrupt_balanced.json",
        help="INSECTS meta 路径（含真值漂移位置）",
    )
    parser.add_argument(
        "--synth_meta_root",
        type=str,
        default="data/synthetic",
        help="合成流 meta 根目录",
    )
    return parser.parse_args()


def ensure_list(spec: str) -> Optional[List[str]]:
    if not spec or spec.lower() == "all":
        return None
    return [item.strip() for item in spec.split(",") if item.strip()]


def list_datasets(logs_root: str) -> List[str]:
    root = Path(logs_root)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_models(log_dir: Path) -> List[str]:
    if not log_dir.exists():
        return []
    models = set()
    for csv in log_dir.glob("*.csv"):
        parts = csv.stem.split("__")
        if len(parts) >= 2:
            models.add(parts[1])
    return sorted(models)


def build_log_path(logs_root: str, dataset: str, model: str, seed: int) -> Path:
    return Path(logs_root) / dataset / f"{dataset}__{model}__seed{seed}.csv"


def load_log(log_path: Path) -> pd.DataFrame:
    return pd.read_csv(log_path)


def summarize_run(df: pd.DataFrame) -> Dict[str, float]:
    acc_final = float(df["metric_accuracy"].iloc[-1])
    kappa_final = float(df["metric_kappa"].iloc[-1])
    acc_max = float(df["metric_accuracy"].max())
    drift_events = int(df["drift_flag"].sum()) if "drift_flag" in df else 0
    total_steps = int(df["step"].iloc[-1])
    seen_samples = int(df["seen_samples"].iloc[-1])
    return {
        "acc_final": acc_final,
        "acc_max": acc_max,
        "kappa_final": kappa_final,
        "drift_events": drift_events,
        "total_steps": total_steps,
        "seen_samples": seen_samples,
    }


def load_synth_meta(dataset_name: str, seed: int, root: str) -> Dict[str, object]:
    meta_path = Path(root) / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
    if not meta_path.exists():
        fallback = SYNTH_FALLBACK_META.get(dataset_name)
        if fallback:
            return fallback
        raise FileNotFoundError(f"缺少合成流 meta：{meta_path}")
    return json.loads(meta_path.read_text())


def get_ground_truth_drifts(
    dataset: str,
    seed: int,
    synth_meta_root: str,
    insects_meta_cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[int], int]:
    if dataset in SYNTH_DATASETS:
        meta = load_synth_meta(dataset, seed, synth_meta_root)
        drifts = [int(d["start"]) for d in meta.get("drifts", [])]
        T = int(meta.get("n_samples", 0))
        return drifts, T
    if dataset in REAL_DATASETS:
        meta = insects_meta_cache or load_insects_abrupt_balanced_meta()
        return [int(pos) for pos in meta.get("positions", [])], int(meta.get("n_samples", 0)) or 0
    return [], 0


def compute_detection_from_log(
    df: pd.DataFrame,
    gt_drifts: Sequence[int],
    horizon: int,
) -> Dict[str, float]:
    if "drift_flag" not in df or "sample_idx" not in df:
        return {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan, "n_detected": 0}
    detections = df.loc[df["drift_flag"] == 1, "sample_idx"].astype(int).tolist()
    T = horizon if horizon > 0 else int(df["sample_idx"].iloc[-1] + 1)
    metrics = compute_detection_metrics(gt_drifts, detections, T)
    return {**metrics, "n_detected": len(detections), "detections": detections}


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value/1000:.0f}k"
    return f"{value:.0f}"


def render_accuracy_plot(
    dataset: str,
    series: Dict[str, pd.DataFrame],
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    width, height = 960, 400
    margin = 50
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    valid_series: Dict[str, List[Tuple[float, float]]] = {}
    min_x, max_x = math.inf, -math.inf
    min_y, max_y = math.inf, -math.inf
    for model, df in series.items():
        if "sample_idx" not in df or "metric_accuracy" not in df:
            continue
        xs = df["sample_idx"].tolist()
        ys = df["metric_accuracy"].tolist()
        if not xs:
            continue
        points = [(float(x), float(y)) for x, y in zip(xs, ys)]
        valid_series[model] = points
        min_x = min(min_x, min(xs))
        max_x = max(max_x, max(xs))
        min_y = min(min_y, min(ys))
        max_y = max(max_y, max(ys))
    if not valid_series or math.isinf(min_y) or math.isinf(max_y) or max_x == min_x:
        return
    min_y = min(0.0, min_y)
    max_y = max(1.0, max_y)

    def scale_x(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * plot_width

    def scale_y(y: float) -> float:
        return height - margin - (y - min_y) / (max_y - min_y) * plot_height

    lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#dddddd"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#444444"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#444444"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="18">{dataset} Accuracy</text>',
        f'<text x="{width/2:.1f}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#333333">sample_idx</text>',
        f'<text x="20" y="{height/2:.1f}" transform="rotate(-90,20,{height/2:.1f})" text-anchor="middle" font-size="12" fill="#333333">metric_accuracy</text>',
    ]
    # X ticks
    tick_count = 5
    if tick_count > 0:
        for i in range(tick_count + 1):
            x_val = min_x + (max_x - min_x) * i / tick_count
            px = scale_x(x_val)
            lines.append(f'<line x1="{px:.2f}" y1="{height - margin}" x2="{px:.2f}" y2="{height - margin + 5}" stroke="#666666"/>')
            lines.append(
                f'<text x="{px:.2f}" y="{height - margin + 18}" text-anchor="middle" font-size="11" fill="#444444">{_format_tick(x_val)}</text>'
            )
        for i in range(tick_count + 1):
            y_val = min_y + (max_y - min_y) * i / tick_count
            py = scale_y(y_val)
            lines.append(f'<line x1="{margin - 5}" y1="{py:.2f}" x2="{margin}" y2="{py:.2f}" stroke="#666666"/>')
            lines.append(
                f'<text x="{margin - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#444444">{y_val:.2f}</text>'
            )
    legend_y = margin - 25
    legend_x = width - margin - 120
    lines.append(f'<rect x="{legend_x}" y="{legend_y - 15}" width="120" height="{25 + 20 * len(valid_series)}" fill="#ffffff" stroke="#cccccc"/>')
    lines.append(
        f'<text x="{legend_x + 60}" y="{legend_y - 2}" text-anchor="middle" font-size="12" fill="#333333">legend</text>'
    )
    for idx, (model, points) in enumerate(valid_series.items()):
        color = colors[idx % len(colors)]
        path_cmds = []
        for i, (x, y) in enumerate(points):
            px, py = scale_x(x), scale_y(y)
            cmd = "M" if i == 0 else "L"
            path_cmds.append(f"{cmd}{px:.2f},{py:.2f}")
        lines.append(f'<path d="{" ".join(path_cmds)}" fill="none" stroke="{color}" stroke-width="2"/>')
        ly = legend_y + 15 + 20 * idx
        lines.append(f'<line x1="{legend_x + 10}" y1="{ly - 4}" x2="{legend_x + 30}" y2="{ly - 4}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{legend_x + 35}" y="{ly}" font-size="12" fill="{color}">{model}</text>')
    lines.append("</svg>")
    svg_path = fig_dir / f"{dataset}_accuracy.svg"
    svg_path.write_text("\n".join(lines), encoding="utf-8")


def render_detection_plot(
    dataset: str,
    df: pd.DataFrame,
    detections: Sequence[int],
    gt_drifts: Sequence[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "sample_idx" not in df:
        return
    xs = df["sample_idx"].tolist()
    if not xs:
        return
    width, height = 960, 200
    margin = 50
    plot_width = width - 2 * margin
    colors = {"gt": "#999999", "det": "#d62728"}
    min_x, max_x = min(xs), max(xs)
    if max_x == min_x:
        return

    def scale_x(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * plot_width

    lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#dddddd"/>',
        f'<text x="{width/2:.1f}" y="25" text-anchor="middle" font-size="16">{dataset} Drift Timeline</text>',
        f'<text x="{width/2:.1f}" y="{height - 5}" text-anchor="middle" font-size="12" fill="#333333">sample_idx</text>',
        f'<line x1="{margin}" y1="{height/2}" x2="{width - margin}" y2="{height/2}" stroke="#555555"/>',
    ]
    for i in range(5):
        x_val = min_x + (max_x - min_x) * i / 4
        px = scale_x(x_val)
        lines.append(f'<line x1="{px:.2f}" y1="{height/2 - 5}" x2="{px:.2f}" y2="{height/2 + 5}" stroke="#888888"/>')
        lines.append(
            f'<text x="{px:.2f}" y="{height/2 + 18}" text-anchor="middle" font-size="11" fill="#444444">{_format_tick(x_val)}</text>'
        )
    legend_x = width - margin - 140
    legend_y = 35
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="140" height="50" fill="#ffffff" stroke="#cccccc"/>')
    lines.append(f'<text x="{legend_x + 70}" y="{legend_y + 15}" text-anchor="middle" font-size="12" fill="#333333">legend</text>')
    lines.append(f'<line x1="{legend_x + 15}" y1="{legend_y + 30}" x2="{legend_x + 30}" y2="{legend_y + 30}" stroke="#999999" stroke-width="2" stroke-dasharray="4,4"/>')
    lines.append(f'<text x="{legend_x + 35}" y="{legend_y + 34}" font-size="12" fill="#555555">ground truth</text>')
    lines.append(f'<line x1="{legend_x + 15}" y1="{legend_y + 45}" x2="{legend_x + 30}" y2="{legend_y + 45}" stroke="#d62728" stroke-width="2"/>')
    lines.append(f'<text x="{legend_x + 35}" y="{legend_y + 49}" font-size="12" fill="#d62728">detections</text>')
    for gt in gt_drifts:
        px = scale_x(gt)
        lines.append(f'<line x1="{px:.2f}" y1="60" x2="{px:.2f}" y2="{height - 20}" stroke="{colors["gt"]}" stroke-width="2" stroke-dasharray="4,4"/>')
    for det in detections:
        px = scale_x(det)
        lines.append(f'<line x1="{px:.2f}" y1="60" x2="{px:.2f}" y2="{height - 20}" stroke="{colors["det"]}" stroke-width="2"/>')
    lines.append("</svg>")
    (out_dir / f"{dataset}_detections.svg").write_text("\n".join(lines), encoding="utf-8")


def save_markdown(summary_df: pd.DataFrame, out_md: Path, top_k: int) -> None:
    lines = ["# 在线实验汇总", ""]
    for dataset, group in summary_df.groupby("dataset"):
        top = (
            group.sort_values(by=["acc_final", "acc_max"], ascending=[False, False])
            .head(top_k)
            .copy()
        )
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| model | acc_final | acc_max | kappa_final | drift_events | seen_samples | MDR | MTD | MTFA | MTR |")
        lines.append("|-------|-----------|---------|-------------|--------------|--------------|-----|-----|------|-----|")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['model']} | {row['acc_final']:.4f} | {row['acc_max']:.4f} | {row['kappa_final']:.4f} | "
                f"{row['drift_events']:.0f} | {row['seen_samples']:.0f} | "
                f"{row['MDR']:.3f} | {row['MTD']:.1f} | {row['MTFA']:.1f} | {row['MTR']:.3f} |"
            )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_filters = ensure_list(args.datasets)
    model_filters = ensure_list(args.models)
    available_datasets = dataset_filters or list_datasets(args.logs_root)
    if not available_datasets:
        raise RuntimeError("未找到任何日志数据集，请先运行在线实验。")
    insects_meta = load_insects_abrupt_balanced_meta(args.insects_meta)
    runs: List[Dict[str, object]] = []
    plot_sources: Dict[str, Dict[str, pd.DataFrame]] = {}
    detection_plots: Dict[str, Tuple[pd.DataFrame, List[int], List[int]]] = {}
    for dataset in available_datasets:
        models = model_filters or list_models(Path(args.logs_root) / dataset)
        if not models:
            print(f"[skip] 数据集 {dataset} 没有可用模型日志")
            continue
        for model in models:
            for seed in args.seeds:
                log_path = build_log_path(args.logs_root, dataset, model, seed)
                if not log_path.exists():
                    continue
                df = load_log(log_path)
                stats = summarize_run(df)
                gt_drifts, T = get_ground_truth_drifts(dataset, seed, args.synth_meta_root, insects_meta)
                detection_metrics = compute_detection_from_log(df, gt_drifts, T)
                runs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        **stats,
                        "MDR": detection_metrics["MDR"],
                        "MTD": detection_metrics["MTD"],
                        "MTFA": detection_metrics["MTFA"],
                        "MTR": detection_metrics["MTR"],
                        "n_detected": detection_metrics["n_detected"],
                    }
                )
                plot_sources.setdefault(dataset, {})
                if model not in plot_sources[dataset]:
                    plot_sources[dataset][model] = df[["sample_idx", "metric_accuracy"]].copy()
                detection_plots[f"{dataset}__{model}__seed{seed}"] = (
                    df,
                    detection_metrics.get("detections", []),
                    list(gt_drifts),
                )
    if not runs:
        raise RuntimeError("没有任何日志文件被读取，请检查 --datasets/--models/--seeds 是否正确。")
    runs_df = pd.DataFrame(runs)
    out_runs = Path(args.out_runs_csv)
    ensure_dir(out_runs)
    runs_df.to_csv(out_runs, index=False)
    summary_df = (
        runs_df.groupby(["dataset", "model"])
        .mean(numeric_only=True)
        .reset_index()
    )
    out_summary = Path(args.out_summary_csv)
    ensure_dir(out_summary)
    summary_df.to_csv(out_summary, index=False)
    save_markdown(summary_df, Path(args.out_md), args.top_k)
    fig_dir = Path(args.fig_dir)
    for dataset, series in plot_sources.items():
        render_accuracy_plot(dataset, series, fig_dir)
    detection_dir = Path(args.detection_fig_dir)
    for key, (df, detections, gt_drifts) in detection_plots.items():
        render_detection_plot(key, df, detections, gt_drifts, detection_dir)
    print(f"共生成 {len(runs_df)} 条运行记录，结果已写入 {out_runs} / {out_summary}")
    print(f"Markdown 汇总：{args.out_md}")
    print(f"Accuracy SVG 目录：{fig_dir}")
    print(f"检测时间线 SVG 目录：{detection_dir}")


if __name__ == "__main__":
    main()
