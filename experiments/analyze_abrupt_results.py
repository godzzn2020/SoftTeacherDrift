"""汇总突变漂移实验结果并绘制图像。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.drift_metrics import compute_detection_metrics
from data.real_meta import load_insects_abrupt_balanced_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析突变漂移实验结果")
    parser.add_argument("--datasets", nargs="*", required=True)
    parser.add_argument("--models", nargs="*", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, required=True)
    parser.add_argument("--logs_root", default="logs")
    parser.add_argument("--synth_root", default="data/synthetic")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--fig_root", default="figures/abrupt")
    parser.add_argument("--insects_meta", default="datasets/real/INSECTS_abrupt_balanced.json")
    return parser.parse_args()


def load_synth_meta(dataset_name: str, seed: int, synth_root: str) -> Dict[str, any]:
    meta_path = Path(synth_root) / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少合成流 meta: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def gather_records(args: argparse.Namespace) -> pd.DataFrame:
    records: List[Dict[str, any]] = []
    insects_meta = load_insects_abrupt_balanced_meta(args.insects_meta)
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                log_path = Path(args.logs_root) / dataset / f"{dataset}__{model}__seed{seed}.csv"
                if not log_path.exists():
                    continue
                df = pd.read_csv(log_path)
                detections = df.loc[df["drift_flag"] == 1, "sample_idx"].astype(int).tolist()
                acc_final = float(df["metric_accuracy"].iloc[-1])
                if dataset == "INSECTS_abrupt_balanced":
                    gt_meta = insects_meta
                    gt_drifts = [int(x) for x in gt_meta["positions"]]
                    T = int(df["sample_idx"].iloc[-1]) + 1
                else:
                    gt_meta = load_synth_meta(dataset, seed, args.synth_root)
                    gt_drifts = [int(d["start"]) for d in gt_meta.get("drifts", [])]
                    T = int(gt_meta.get("n_samples", df["sample_idx"].iloc[-1] + 1))
                metrics = compute_detection_metrics(gt_drifts, detections, T)
                record = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "MDR": metrics["MDR"],
                    "MTD": metrics["MTD"],
                    "MTFA": metrics["MTFA"],
                    "MTR": metrics["MTR"],
                    "acc_final": acc_final,
                    "n_gt_drifts": len(gt_drifts),
                    "n_detected": len(detections),
                    "sample_count": T,
                }
                records.append(record)
    if not records:
        raise RuntimeError("没有找到任何日志记录，请先运行实验。")
    return pd.DataFrame(records)


def save_tables(df: pd.DataFrame, results_root: str) -> None:
    out_dir = Path(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "abrupt_drift_runs.csv"
    summary_path = out_dir / "abrupt_drift_summary.csv"
    md_path = out_dir / "abrupt_drift_summary.md"
    df.to_csv(runs_path, index=False)
    summary = df.groupby(["dataset", "model"]).mean(numeric_only=True).reset_index()
    summary.to_csv(summary_path, index=False)
    md_lines = ["| dataset | model | MDR | MTD | MTFA | MTR | acc_final |", "|--------|-------|-----|-----|------|-----|-----------|"]
    for _, row in summary.iterrows():
        md_lines.append(
            f"| {row['dataset']} | {row['model']} | {row['MDR']:.3f} | {row['MTD']:.2f} | {row['MTFA']:.2f} | {row['MTR']:.2f} | {row['acc_final']:.4f} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def plot_accuracy(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    logs_root: str,
    fig_root: str,
) -> None:
    fig_dir = Path(fig_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        plt.figure()
        plotted = False
        for model in models:
            df = None
            for seed in seeds:
                path = Path(logs_root) / dataset / f"{dataset}__{model}__seed{seed}.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    break
            if df is None:
                continue
            plt.plot(df["sample_idx"], df["metric_accuracy"], label=model)
            plotted = True
        if plotted:
            plt.xlabel("sample_idx")
            plt.ylabel("metric_accuracy")
            plt.title(f"{dataset} Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{dataset}_accuracy.png")
        plt.close()


def plot_drift_timelines(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    logs_root: str,
    synth_root: str,
    fig_root: str,
    insects_meta_path: str,
) -> None:
    fig_dir = Path(fig_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    insects_meta = load_insects_abrupt_balanced_meta(insects_meta_path)
    for dataset in datasets:
        plt.figure()
        if dataset == "INSECTS_abrupt_balanced":
            gt_drifts = insects_meta["positions"]
        else:
            seed = seeds[0]
            meta = load_synth_meta(dataset, seed, synth_root)
            gt_drifts = [int(d["start"]) for d in meta.get("drifts", [])]
        for g in gt_drifts:
            plt.axvline(g, color="gray", linestyle="--", alpha=0.5)
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for idx, model in enumerate(models):
            df = None
            for seed in seeds:
                path = Path(logs_root) / dataset / f"{dataset}__{model}__seed{seed}.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    break
            if df is None:
                continue
            detections = df.loc[df["drift_flag"] == 1, "sample_idx"].astype(int).tolist()
            plt.vlines(
                detections,
                ymin=0,
                ymax=1,
                colors=colors[idx % len(colors)],
                label=f"{model} detections",
                linewidth=1,
            )
        plt.xlabel("sample_idx")
        plt.yticks([])
        plt.title(f"{dataset} Drift Timeline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{dataset}_drifts.png")
        plt.close()


def main() -> None:
    args = parse_args()
    df = gather_records(args)
    save_tables(df, args.results_root)
    plot_accuracy(args.datasets, args.models, args.seeds, args.logs_root, args.fig_root)
    plot_drift_timelines(
        args.datasets,
        args.models,
        args.seeds,
        args.logs_root,
        args.synth_root,
        args.fig_root,
        args.insects_meta,
    )


if __name__ == "__main__":
    main()
