"""多 GPU 并行运行 Stage-1 实验组合。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.first_stage_experiments import (
    ExperimentConfig,
    _default_experiment_configs,
    _default_log_path,
)


@dataclass
class Task:
    """单个运行任务，包含命令与标识。"""

    label: str
    cmd: List[str]
    env: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 GPU 并行运行 Stage-1 实验组合")
    parser.add_argument(
        "--datasets",
        type=str,
        default="sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced",
        help="逗号分隔数据集（或 all）",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="baseline_student,mean_teacher,ts_drift_adapt",
        help="逗号分隔模型（或 all）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1],
        help="需要运行的随机种子",
    )
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta",
        choices=["none", "error_ph_meta", "divergence_ph_meta", "error_divergence_ph_meta"],
        help="漂移检测预设",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="逗号分隔 GPU 列表（例如 0,1），若只用 CPU 可填 none",
    )
    parser.add_argument(
        "--max_jobs_per_gpu",
        type=int,
        default=1,
        help="单个 GPU 同时运行的任务数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="传给 run_experiment.py 的 --device（默认 cuda）",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="运行子任务的 Python 解释器",
    )
    parser.add_argument(
        "--sleep_interval",
        type=float,
        default=2.0,
        help="轮询任务状态的间隔（秒）",
    )
    return parser.parse_args()


def ensure_list(spec: str) -> Optional[List[str]]:
    if not spec or spec.lower() == "all":
        return None
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_gpus(spec: str) -> List[str]:
    if not spec or spec.lower() in {"none", "cpu"}:
        return []
    return [gpu.strip() for gpu in spec.split(",") if gpu.strip()]


def select_configs(
    dataset_filters: Optional[List[str]],
    base_configs: Sequence[ExperimentConfig],
) -> List[ExperimentConfig]:
    if not dataset_filters:
        return list(base_configs)
    target = {name.lower() for name in dataset_filters}
    selected = [cfg for cfg in base_configs if cfg.dataset_name.lower() in target]
    if not selected:
        raise ValueError(f"未匹配到数据集：{dataset_filters}")
    return selected


def select_models(model_filters: Optional[List[str]]) -> List[str]:
    available_models = ["baseline_student", "mean_teacher", "ts_drift_adapt"]
    if not model_filters:
        return available_models
    target = [m.strip() for m in model_filters if m.strip()]
    invalid = [m for m in target if m not in available_models]
    if invalid:
        raise ValueError(f"不支持的模型：{', '.join(invalid)}")
    return target


def build_command(
    cfg: ExperimentConfig,
    model_variant: str,
    seed: int,
    log_path: str,
    monitor_preset: str,
    python_bin: str,
    device: str,
) -> List[str]:
    hidden_dims = ",".join(str(d) for d in cfg.hidden_dims)
    cmd = [
        python_bin,
        "run_experiment.py",
        "--dataset_type",
        cfg.dataset_type,
        "--dataset_name",
        cfg.dataset_name,
        "--model_variant",
        model_variant,
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
        hidden_dims,
        "--dropout",
        str(cfg.dropout),
        "--monitor_preset",
        monitor_preset,
        "--log_path",
        log_path,
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if cfg.csv_path:
        cmd.extend(["--csv_path", cfg.csv_path])
    if cfg.label_col:
        cmd.extend(["--label_col", cfg.label_col])
    return cmd


def create_tasks(
    datasets: List[ExperimentConfig],
    models: List[str],
    seeds: Sequence[int],
    monitor_preset: str,
    python_bin: str,
    device: str,
) -> List[Task]:
    tasks: List[Task] = []
    for cfg in datasets:
        for seed in seeds:
            for model_variant in models:
                log_path = _default_log_path(cfg.dataset_name, model_variant, seed)
                cmd = build_command(
                    cfg,
                    model_variant,
                    seed,
                    log_path,
                    monitor_preset,
                    python_bin,
                    device,
                )
                label = f"{cfg.dataset_name}__{model_variant}__seed{seed}"
                tasks.append(Task(label=label, cmd=cmd, env={}))
    return tasks


def assign_env(task: Task, gpu: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    task.env = env
    return env


@dataclass
class RunningTask:
    process: subprocess.Popen
    gpu: Optional[str]
    slot_id: int
    task: Task


def run_tasks(tasks: List[Task], gpus: List[str], max_jobs_per_gpu: int, sleep_interval: float) -> None:
    if not tasks:
        print("没有任何任务需要运行。")
        return
    if gpus:
        slots: Deque[Tuple[Optional[str], int]] = deque()
        for gpu in gpus:
            for slot in range(max_jobs_per_gpu):
                slots.append((gpu, slot))
    else:
        # CPU 模式，仅一个 slot
        slots = deque([(None, 0)])
    running: List[RunningTask] = []
    next_task_idx = 0
    total = len(tasks)
    try:
        while next_task_idx < total or running:
            # 回收已完成任务
            i = 0
            while i < len(running):
                rt = running[i]
                ret = rt.process.poll()
                if ret is not None:
                    status = "成功" if ret == 0 else f"失败({ret})"
                    gpu_label = rt.gpu if rt.gpu is not None else "cpu"
                    print(f"[完成][{gpu_label}] {rt.task.label} -> {status}")
                    slots.append((rt.gpu, rt.slot_id))
                    running.pop(i)
                else:
                    i += 1
            # 启动新任务
            while slots and next_task_idx < total:
                gpu, slot_id = slots.popleft()
                task = tasks[next_task_idx]
                next_task_idx += 1
                env = assign_env(task, gpu)
                gpu_label = gpu if gpu is not None else "cpu"
                print(f"[启动][{gpu_label}] {task.label}")
                proc = subprocess.Popen(task.cmd, env=env)
                running.append(RunningTask(process=proc, gpu=gpu, slot_id=slot_id, task=task))
            if next_task_idx >= total and not running:
                break
            time.sleep(sleep_interval)
    except KeyboardInterrupt:
        print("收到中断信号，正在终止子任务...")
        for rt in running:
            rt.process.terminate()
        raise


def main() -> None:
    args = parse_args()
    dataset_filters = ensure_list(args.datasets)
    model_filters = ensure_list(args.models)
    gpus = parse_gpus(args.gpus)
    base_configs = _default_experiment_configs(args.seeds[0] if args.seeds else 1)
    selected_configs = select_configs(dataset_filters, base_configs)
    models = select_models(model_filters)
    tasks = create_tasks(
        datasets=selected_configs,
        models=models,
        seeds=args.seeds,
        monitor_preset=args.monitor_preset,
        python_bin=args.python_bin,
        device=args.device,
    )
    print(f"共生成 {len(tasks)} 个任务，将使用 GPU 列表：{gpus or ['cpu']}（每卡 {args.max_jobs_per_gpu} 个并发）")
    run_tasks(tasks, gpus, args.max_jobs_per_gpu, args.sleep_interval)
    print("全部任务完成。")


if __name__ == "__main__":
    main()
