"""批量运行 Stage-1 多 seed 实验并汇总结果。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def parse_gpus(spec: str) -> List[str]:
    if not spec or spec.lower() in {"none", "cpu"}:
        return []
    return [token.strip() for token in spec.split(",") if token.strip()]

from data.real_meta import load_insects_abrupt_balanced_meta
from data.streams import generate_default_abrupt_synth_datasets
from experiments import summarize_online_results as sor
from experiments.first_stage_experiments import (
    ExperimentConfig,
    _default_experiment_configs,
    _default_log_path,
)


DEFAULT_DATASETS = ["sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"]
DEFAULT_MODELS = ["baseline_student", "mean_teacher", "ts_drift_adapt"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


@dataclass
class Args:
    datasets: List[str]
    models: List[str]
    seeds: List[int]
    monitor_preset: str
    device: str
    gpus: List[str]
    max_jobs_per_gpu: int
    sleep_interval: float
    logs_root: Path
    synth_meta_root: str
    insects_meta: str
    out_csv_raw: Path
    out_csv_summary: Path
    out_md_dir: Path


@dataclass
class Task:
    label: str
    cmd: List[str]
    dataset: str
    model: str
    seed: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Stage-1 多 seed 实验调度与汇总")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="逗号分隔的数据集名称列表",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="逗号分隔的模型变体列表",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="需要运行的随机种子",
    )
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="error_ph_meta",
        choices=["none", "error_ph_meta", "divergence_ph_meta", "error_divergence_ph_meta"],
        help="漂移检测器预设（默认 error_ph_meta）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="运行设备（传递给 run_experiment）")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="并行运行时使用的 GPU（逗号分隔，填 none 表示仅 CPU 顺序运行）",
    )
    parser.add_argument(
        "--max_jobs_per_gpu",
        type=int,
        default=2,
        help="单张 GPU 同时运行的任务数量",
    )
    parser.add_argument(
        "--sleep_interval",
        type=float,
        default=2.0,
        help="轮询子任务状态的间隔（秒）",
    )
    parser.add_argument("--logs_root", type=str, default="logs", help="日志根目录")
    parser.add_argument("--synth_meta_root", type=str, default="data/synthetic")
    parser.add_argument("--insects_meta", type=str, default="datasets/real/INSECTS_abrupt_balanced.json")
    parser.add_argument("--out_csv_raw", type=str, default="results/stage1_multi_seed_raw.csv")
    parser.add_argument("--out_csv_summary", type=str, default="results/stage1_multi_seed_summary.csv")
    parser.add_argument("--out_md_dir", type=str, default="results/stage1_multi_seed_md")
    ns = parser.parse_args()
    datasets = [item.strip() for item in ns.datasets.split(",") if item.strip()]
    models = [item.strip() for item in ns.models.split(",") if item.strip()]
    if not datasets:
        raise ValueError("至少需要一个数据集")
    if not models:
        raise ValueError("至少需要一个模型")
    if not ns.seeds:
        raise ValueError("至少需要一个 seed")
    return Args(
        datasets=datasets,
        models=models,
        seeds=list(ns.seeds),
        monitor_preset=ns.monitor_preset,
        device=ns.device,
        gpus=parse_gpus(ns.gpus),
        max_jobs_per_gpu=ns.max_jobs_per_gpu,
        sleep_interval=ns.sleep_interval,
        logs_root=Path(ns.logs_root),
        synth_meta_root=ns.synth_meta_root,
        insects_meta=ns.insects_meta,
        out_csv_raw=Path(ns.out_csv_raw),
        out_csv_summary=Path(ns.out_csv_summary),
        out_md_dir=Path(ns.out_md_dir),
    )


def build_dataset_config_map(seed: int) -> Dict[str, ExperimentConfig]:
    base_configs = _default_experiment_configs(seed)
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in base_configs:
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def build_command(
    cfg: ExperimentConfig,
    model: str,
    seed: int,
    monitor_preset: str,
    device: str,
    logs_root: Path,
) -> Tuple[str, List[str]]:
    default_log = Path(_default_log_path(cfg.dataset_name, model, seed))
    if logs_root.resolve() != default_log.parent.parent.resolve():
        log_dir = logs_root / cfg.dataset_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / default_log.name)
    else:
        log_path = str(default_log)
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--dataset_type",
        cfg.dataset_type,
        "--dataset_name",
        cfg.dataset_name,
        "--model_variant",
        model,
        "--seed",
        str(seed),
        "--monitor_preset",
        monitor_preset,
        "--device",
        device,
        "--log_path",
        log_path,
        "--batch_size",
        str(cfg.batch_size),
        "--labeled_ratio",
        str(cfg.labeled_ratio),
        "--n_steps",
        str(cfg.n_steps),
        "--initial_alpha",
        str(cfg.initial_alpha),
        "--initial_lr",
        str(cfg.initial_lr),
        "--lambda_u",
        str(cfg.lambda_u),
        "--tau",
        str(cfg.tau),
        "--hidden_dims",
        ",".join(str(x) for x in cfg.hidden_dims),
        "--dropout",
        str(cfg.dropout),
    ]
    if cfg.csv_path:
        cmd.extend(["--csv_path", cfg.csv_path])
    if cfg.label_col:
        cmd.extend(["--label_col", cfg.label_col])
    return log_path, cmd


def collect_run_metrics(
    datasets: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    logs_root: Path,
    synth_meta_root: str,
    insects_meta_path: str,
) -> pd.DataFrame:
    insects_meta = load_insects_abrupt_balanced_meta(insects_meta_path)
    records: List[Dict[str, object]] = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                log_path = logs_root / dataset / f"{dataset}__{model}__seed{seed}.csv"
                if not log_path.exists():
                    continue
                df = sor.load_log(log_path)
                stats = sor.summarize_run(df)
                gt_drifts, horizon = sor.get_ground_truth_drifts(dataset, seed, synth_meta_root, insects_meta)
                detection = sor.compute_detection_from_log(df, gt_drifts, horizon)
                records.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "acc_final": stats["acc_final"],
                        "acc_max": stats["acc_max"],
                        "kappa_final": stats["kappa_final"],
                        "drift_events": stats["drift_events"],
                        "total_steps": stats["total_steps"],
                        "seen_samples": stats["seen_samples"],
                        "MDR": detection["MDR"],
                        "MTD": detection["MTD"],
                        "MTFA": detection["MTFA"],
                        "MTR": detection["MTR"],
                        "n_detected": detection["n_detected"],
                    }
                )
    if not records:
        raise RuntimeError("未找到任何日志，请先运行实验。")
    return pd.DataFrame(records)


def compute_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "acc_final",
        "acc_max",
        "kappa_final",
        "MDR",
        "MTD",
        "MTFA",
        "MTR",
        "n_detected",
    ]
    agg_dict = {}
    for m in metrics:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    summary = df_raw.groupby(["dataset", "model"]).agg(**agg_dict).reset_index()
    return summary


def save_md_per_dataset(df_summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("acc_final", 3),
        ("acc_max", 3),
        ("kappa_final", 3),
        ("MDR", 3),
        ("MTD", 1),
        ("MTFA", 1),
        ("MTR", 3),
        ("n_detected", 1),
    ]
    for dataset, group in df_summary.groupby("dataset"):
        lines = [
            f"# {dataset} Multi-seed Summary",
            "",
            "| model | acc_final (mean±std) | acc_max (mean±std) | kappa_final (mean±std) | MDR (mean±std) | MTD (mean±std) | MTFA (mean±std) | MTR (mean±std) | n_detected (mean±std) |",
            "|-------|----------------------|---------------------|------------------------|----------------|----------------|-----------------|----------------|------------------------|",
        ]

        def fmt(mu: float, sigma: float, ndigits: int) -> str:
            if pd.isna(mu):
                return "NaN"
            if pd.isna(sigma):
                return f"{mu:.{ndigits}f}±NaN"
            return f"{mu:.{ndigits}f}±{sigma:.{ndigits}f}"

        for _, row in group.iterrows():
            entries = []
            for name, nd in metrics:
                entries.append(fmt(row[f"{name}_mean"], row[f"{name}_std"], nd))
            lines.append(
                "| {model} | {acc_f} | {acc_m} | {kappa} | {mdr} | {mtd} | {mtfa} | {mtr} | {nd} |".format(
                    model=row["model"], acc_f=entries[0], acc_m=entries[1], kappa=entries[2], mdr=entries[3],
                    mtd=entries[4], mtfa=entries[5], mtr=entries[6], nd=entries[7]
                )
            )
        (out_dir / f"{dataset}_multi_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")


def print_best_models(df_summary: pd.DataFrame) -> None:
    for dataset, group in df_summary.groupby("dataset"):
        best = group.sort_values(by="acc_final_mean", ascending=False).iloc[0]
        print(
            f"[best] {dataset}: {best['model']} acc_final_mean={best['acc_final_mean']:.4f} "
            f"(MDR_mean={best['MDR_mean']:.3f})"
        )


def run_task_queue(tasks: List[Task], gpus: List[str], max_jobs_per_gpu: int, sleep_interval: float) -> None:
    if not tasks:
        return
    slots: Deque[Tuple[Optional[str], int]] = deque()
    if gpus:
        for gpu in gpus:
            for slot in range(max_jobs_per_gpu):
                slots.append((gpu, slot))
    else:
        slots.append((None, 0))
    running: List[Tuple[subprocess.Popen, Optional[str], int, Task]] = []
    idx = 0
    failures: List[str] = []
    while idx < len(tasks) or running:
        # launch
        while slots and idx < len(tasks):
            gpu, slot_id = slots.popleft()
            task = tasks[idx]
            idx += 1
            env = os.environ.copy()
            gpu_label = gpu if gpu is not None else "cpu"
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"[launch][{gpu_label}] {task.label}")
            proc = subprocess.Popen(task.cmd, env=env)
            running.append((proc, gpu, slot_id, task))
        # poll
        i = 0
        while i < len(running):
            proc, gpu, slot_id, task = running[i]
            ret = proc.poll()
            if ret is None:
                i += 1
                continue
            gpu_label = gpu if gpu is not None else "cpu"
            status = "success" if ret == 0 else f"failed({ret})"
            print(f"[done][{gpu_label}] {task.label} -> {status}")
            if ret != 0:
                failures.append(task.label)
            slots.append((gpu, slot_id))
            running.pop(i)
        if idx >= len(tasks) and not running:
            break
        time.sleep(max(sleep_interval, 0.1))
    if failures:
        raise RuntimeError(f"以下任务运行失败：{failures}")


def main() -> None:
    args = parse_args()
    # 确保合成流 seeds 的 parquet + meta 已生成
    generate_default_abrupt_synth_datasets(seeds=args.seeds, out_root=args.synth_meta_root)
    cfg_map = build_dataset_config_map(seed=args.seeds[0])
    for dataset in args.datasets:
        if dataset.lower() not in cfg_map:
            raise ValueError(f"未知的数据集：{dataset}，请在 first_stage_experiments 中配置后再运行。")
    for dataset in args.datasets:
        template = cfg_map[dataset.lower()]
    tasks: List[Task] = []
    for dataset in args.datasets:
        template = cfg_map[dataset.lower()]
        for model in args.models:
            for seed in args.seeds:
                log_path, cmd = build_command(
                    template,
                    model,
                    seed,
                    args.monitor_preset,
                    args.device,
                    args.logs_root,
                )
                tasks.append(Task(label=f"{dataset}__{model}__seed{seed}", cmd=cmd, dataset=dataset, model=model, seed=seed))
    run_task_queue(tasks, args.gpus, args.max_jobs_per_gpu, args.sleep_interval)
    df_raw = collect_run_metrics(
        args.datasets,
        args.models,
        args.seeds,
        args.logs_root,
        args.synth_meta_root,
        args.insects_meta,
    )
    args.out_csv_raw.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(args.out_csv_raw, index=False)
    df_summary = compute_summary(df_raw)
    args.out_csv_summary.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(args.out_csv_summary, index=False)
    save_md_per_dataset(df_summary, args.out_md_dir)
    print_best_models(df_summary)


if __name__ == "__main__":
    main()
def parse_gpus(spec: str) -> List[str]:
    if not spec or spec.lower() in {"none", "cpu"}:
        return []
    return [token.strip() for token in spec.split(",") if token.strip()]
@dataclass
class Task:
    label: str
    cmd: List[str]
    dataset: str
    model: str
    seed: int
