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

from soft_drift.utils.run_paths import DatasetRunPaths, ExperimentRun, create_experiment_run

def parse_gpus(spec: str) -> List[str]:
    if not spec or spec.lower() in {"none", "cpu"}:
        return []
    return [token.strip() for token in spec.split(",") if token.strip()]

from data.real_meta import load_insects_abrupt_balanced_meta
from data.streams import generate_default_abrupt_synth_datasets
from experiments import summarize_online_results as sor
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs


DEFAULT_DATASETS = ["sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"]
DEFAULT_MODELS = ["baseline_student", "mean_teacher", "ts_drift_adapt", "ts_drift_adapt_severity"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


@dataclass
class Args:
    datasets: List[str]
    models: List[str]
    seeds: List[int]
    monitor_preset: str
    trigger_mode: str
    trigger_k: int
    trigger_threshold: float
    trigger_weights: str
    confirm_window: int
    use_severity_v2: bool
    severity_gate: str
    severity_gate_min_streak: int
    entropy_mode: str
    severity_decay: float
    freeze_baseline_steps: int
    device: str
    gpus: List[str]
    max_jobs_per_gpu: int
    sleep_interval: float
    logs_root: Path
    synth_meta_root: str
    insects_meta: str
    out_csv_raw: Optional[Path]
    out_csv_summary: Optional[Path]
    out_md_dir: Optional[Path]
    results_root: Path
    run_name: Optional[str]
    run_id: Optional[str]
    severity_scheduler_scale: float


@dataclass
class Task:
    label: str
    cmd: List[str]
    dataset: str
    model: str
    seed: int
    run_paths: DatasetRunPaths


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
        type=str,
        default=[",".join(str(x) for x in DEFAULT_SEEDS)],
        help="随机种子（逗号或空格分隔）",
    )
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="error_ph_meta",
        choices=["none", "error_ph_meta", "divergence_ph_meta", "error_divergence_ph_meta"],
        help="漂移检测器预设（默认 error_ph_meta）",
    )
    parser.add_argument(
        "--trigger_mode",
        type=str,
        default="or",
        choices=["or", "k_of_n", "weighted", "two_stage"],
        help="多 detector 融合触发策略（默认 or）",
    )
    parser.add_argument("--trigger_k", type=int, default=2, help="trigger_mode=k_of_n 时的 k")
    parser.add_argument("--trigger_threshold", type=float, default=0.5, help="trigger_mode=weighted 时阈值")
    parser.add_argument(
        "--trigger_weights",
        type=str,
        default="",
        help="trigger_mode=weighted 时权重（key=value 逗号分隔，空表示默认）",
    )
    parser.add_argument("--confirm_window", type=int, default=200, help="trigger_mode=two_stage 的 confirm_window")
    parser.add_argument("--use_severity_v2", action="store_true", help="启用 Severity-Aware v2（carry+decay）")
    parser.add_argument(
        "--severity_gate",
        type=str,
        default="none",
        choices=["none", "confirmed_only", "confirmed_streak"],
        help="severity v2 gating（confirmed_only 仅高置信 drift 更新 carry/freeze；confirmed_streak 需要连续满足条件）",
    )
    parser.add_argument(
        "--severity_gate_min_streak",
        type=int,
        default=1,
        help="severity_gate=confirmed_streak 时的最小连续确认步数（>=1）",
    )
    parser.add_argument(
        "--entropy_mode",
        type=str,
        default="overconfident",
        choices=["overconfident", "uncertain", "abs"],
        help="SeverityCalibrator 的 entropy 正向增量定义",
    )
    parser.add_argument("--severity_decay", type=float, default=0.95, help="Severity-Aware v2 carry 衰减系数")
    parser.add_argument("--freeze_baseline_steps", type=int, default=0, help="漂移后冻结 baseline 的步数")
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
    parser.add_argument("--out_csv_raw", type=str, default=None)
    parser.add_argument("--out_csv_summary", type=str, default=None)
    parser.add_argument("--out_md_dir", type=str, default=None)
    parser.add_argument("--results_root", type=str, default="results", help="结果输出根目录")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="自定义 run 名称（将附加在自动 run_id 后）",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="覆盖自动生成的 run_id（谨慎使用，可能造成覆盖）",
    )
    parser.add_argument(
        "--severity_scheduler_scale",
        type=float,
        default=1.0,
        help="severity-aware 调度缩放（仅在 *_severity 变体生效）",
    )
    ns = parser.parse_args()
    datasets = [item.strip() for item in ns.datasets.split(",") if item.strip()]
    models = [item.strip() for item in ns.models.split(",") if item.strip()]
    if not datasets:
        raise ValueError("至少需要一个数据集")
    if not models:
        raise ValueError("至少需要一个模型")
    if not ns.seeds:
        raise ValueError("至少需要一个 seed")
    seed_values: List[int] = []
    for token in ns.seeds:
        for part in token.replace(",", " ").split():
            if not part:
                continue
            seed_values.append(int(part))
    if not seed_values:
        raise ValueError("至少需要一个 seed")
    return Args(
        datasets=datasets,
        models=models,
        seeds=seed_values,
        monitor_preset=ns.monitor_preset,
        trigger_mode=ns.trigger_mode,
        trigger_k=int(ns.trigger_k),
        trigger_threshold=float(ns.trigger_threshold),
        trigger_weights=str(ns.trigger_weights),
        confirm_window=int(ns.confirm_window),
        use_severity_v2=bool(ns.use_severity_v2),
        severity_gate=str(ns.severity_gate),
        severity_gate_min_streak=int(getattr(ns, "severity_gate_min_streak", 1) or 1),
        entropy_mode=str(ns.entropy_mode),
        severity_decay=float(ns.severity_decay),
        freeze_baseline_steps=int(ns.freeze_baseline_steps),
        device=ns.device,
        gpus=parse_gpus(ns.gpus),
        max_jobs_per_gpu=ns.max_jobs_per_gpu,
        sleep_interval=ns.sleep_interval,
        logs_root=Path(ns.logs_root),
        synth_meta_root=ns.synth_meta_root,
        insects_meta=ns.insects_meta,
        out_csv_raw=Path(ns.out_csv_raw) if ns.out_csv_raw else None,
        out_csv_summary=Path(ns.out_csv_summary) if ns.out_csv_summary else None,
        out_md_dir=Path(ns.out_md_dir) if ns.out_md_dir else None,
        results_root=Path(ns.results_root),
        run_name=ns.run_name,
        run_id=ns.run_id,
        severity_scheduler_scale=ns.severity_scheduler_scale,
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
    trigger_mode: str,
    trigger_k: int,
    trigger_threshold: float,
    trigger_weights: str,
    confirm_window: int,
    use_severity_v2: bool,
    severity_gate: str,
    severity_gate_min_streak: int,
    entropy_mode: str,
    severity_decay: float,
    freeze_baseline_steps: int,
    device: str,
    log_path: Path,
    severity_scheduler_scale: float,
    experiment_run: ExperimentRun,
) -> Tuple[str, List[str]]:
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
        "--trigger_mode",
        trigger_mode,
        "--trigger_k",
        str(trigger_k),
        "--trigger_threshold",
        str(trigger_threshold),
        "--confirm_window",
        str(confirm_window),
        "--device",
        device,
        "--log_path",
        str(log_path),
        "--results_root",
        str(experiment_run.results_root),
        "--logs_root",
        str(experiment_run.logs_root),
        "--experiment_name",
        experiment_run.experiment_name,
        "--run_id",
        experiment_run.run_id,
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
    cmd.extend(["--severity_scheduler_scale", str(severity_scheduler_scale)])
    if trigger_weights:
        cmd.extend(["--trigger_weights", trigger_weights])
    if use_severity_v2:
        cmd.append("--use_severity_v2")
        cmd.extend(["--severity_gate", str(severity_gate)])
        cmd.extend(["--severity_gate_min_streak", str(int(severity_gate_min_streak))])
        cmd.extend(["--entropy_mode", str(entropy_mode)])
        cmd.extend(["--severity_decay", str(severity_decay)])
        cmd.extend(["--freeze_baseline_steps", str(freeze_baseline_steps)])
    return str(log_path), cmd


def collect_run_metrics(
    tasks: Sequence["Task"],
    synth_meta_root: str,
    insects_meta_path: str,
) -> pd.DataFrame:
    insects_meta = load_insects_abrupt_balanced_meta(insects_meta_path)
    records: List[Dict[str, object]] = []
    for task in tasks:
        log_path = task.run_paths.log_csv_path()
        if not log_path.exists():
            continue
        df = sor.load_log(log_path)
        stats = sor.summarize_run(df)
        gt_drifts, horizon = sor.get_ground_truth_drifts(task.dataset, task.seed, synth_meta_root, insects_meta)
        detection = sor.compute_detection_from_log(df, gt_drifts, horizon)
        records.append(
            {
                "dataset": task.dataset,
                "model": task.model,
                "seed": task.seed,
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


def create_tasks(
    datasets: List[ExperimentConfig],
    models: List[str],
    seeds: Sequence[int],
    monitor_preset: str,
    trigger_mode: str,
    trigger_k: int,
    trigger_threshold: float,
    trigger_weights: str,
    confirm_window: int,
    use_severity_v2: bool,
    severity_gate: str,
    severity_gate_min_streak: int,
    entropy_mode: str,
    severity_decay: float,
    freeze_baseline_steps: int,
    python_bin: str,
    device: str,
    experiment_run: ExperimentRun,
    severity_scheduler_scale: float,
) -> List[Task]:
    tasks: List[Task] = []
    for cfg in datasets:
        for model_variant in models:
            for seed in seeds:
                run_paths = experiment_run.prepare_dataset_run(cfg.dataset_name, model_variant, seed)
                log_path_str, cmd = build_command(
                    cfg,
                    model_variant,
                    seed,
                    monitor_preset,
                    trigger_mode,
                    trigger_k,
                    trigger_threshold,
                    trigger_weights,
                    confirm_window,
                    use_severity_v2,
                    severity_gate,
                    severity_gate_min_streak,
                    entropy_mode,
                    severity_decay,
                    freeze_baseline_steps,
                    device,
                    run_paths.log_csv_path(),
                    severity_scheduler_scale,
                    experiment_run,
                )
                cmd[0] = python_bin
                tasks.append(
                    Task(
                        label=f"{cfg.dataset_name}__{model_variant}__seed{seed}",
                        cmd=cmd,
                        dataset=cfg.dataset_name,
                        model=model_variant,
                        seed=seed,
                        run_paths=run_paths,
                    )
                )
    return tasks


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
    experiment_run = create_experiment_run(
        experiment_name="stage1_multi_seed",
        results_root=args.results_root,
        logs_root=args.logs_root,
        run_name=args.run_name,
        run_id=args.run_id,
    )
    print(f"[run] stage1_multi_seed run_id={experiment_run.run_id}")
    # 确保合成流 seeds 的 parquet + meta 已生成
    generate_default_abrupt_synth_datasets(seeds=args.seeds, out_root=args.synth_meta_root)
    cfg_map = build_dataset_config_map(seed=args.seeds[0])
    for dataset in args.datasets:
        if dataset.lower() not in cfg_map:
            raise ValueError(f"未知的数据集：{dataset}，请在 first_stage_experiments 中配置后再运行。")
    selected_cfgs = [cfg_map[d.lower()] for d in args.datasets]
    tasks = create_tasks(
        datasets=selected_cfgs,
        models=args.models,
        seeds=args.seeds,
        monitor_preset=args.monitor_preset,
        trigger_mode=args.trigger_mode,
        trigger_k=args.trigger_k,
        trigger_threshold=args.trigger_threshold,
        trigger_weights=args.trigger_weights,
        confirm_window=args.confirm_window,
        use_severity_v2=args.use_severity_v2,
        severity_gate=args.severity_gate,
        severity_gate_min_streak=args.severity_gate_min_streak,
        entropy_mode=args.entropy_mode,
        severity_decay=args.severity_decay,
        freeze_baseline_steps=args.freeze_baseline_steps,
        python_bin=sys.executable,
        device=args.device,
        experiment_run=experiment_run,
        severity_scheduler_scale=args.severity_scheduler_scale,
    )
    run_task_queue(tasks, args.gpus, args.max_jobs_per_gpu, args.sleep_interval)
    for task in tasks:
        task.run_paths.update_legacy_pointer()
    df_raw = collect_run_metrics(tasks, args.synth_meta_root, args.insects_meta)
    summary_dir = experiment_run.summary_dir()
    raw_path = args.out_csv_raw or (summary_dir / "run_level_metrics.csv")
    summary_path = args.out_csv_summary or (summary_dir / "summary_metrics_by_dataset_model.csv")
    md_dir = args.out_md_dir or (summary_dir / "markdown")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_path, index=False)
    df_summary = compute_summary(df_raw)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    save_md_per_dataset(df_summary, md_dir)
    print_best_models(df_summary)
    print(f"[done] raw metrics -> {raw_path}")
    print(f"[done] summary metrics -> {summary_path}")


if __name__ == "__main__":
    main()
