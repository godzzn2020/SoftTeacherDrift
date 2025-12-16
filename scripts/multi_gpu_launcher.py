#!/usr/bin/env python
"""通用多 GPU 实验调度器。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_PHASE0_DATASETS = ["Airlines", "Electricity", "NOAA", "INSECTS_abrupt_balanced"]
DEFAULT_STAGE1_SYNTH_DATASETS = ["sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"]
DEFAULT_STAGE1_MODELS = ["ts_drift_adapt", "ts_drift_adapt_severity"]

DATASET_GPU_COST: Dict[str, float] = {
    # 真实流
    "airlines": 1.5,
    "insects_abrupt_balanced": 1.5,
    "electricity": 1.0,
    "noaa": 1.0,
    # 合成流
    "sea_abrupt4": 1.0,
    "sine_abrupt4": 1.0,
    "stagger_abrupt3": 1.0,
}
GPU_CAPACITY = 3.0
DEFAULT_COST = 1.0


@dataclass
class Job:
    name: str
    cmd: List[str]
    dataset_name: str
    gpu_cost: float


def parse_seed_spec(spec: str) -> List[int]:
    tokens = [tok for tok in spec.replace(",", " ").split() if tok.strip()]
    if not tokens:
        raise ValueError("至少需要提供一个 seed")
    seeds = [int(tok) for tok in tokens]
    return seeds


def parse_gpu_ids(spec: Optional[str]) -> List[int]:
    if spec:
        tokens = [token.strip() for token in spec.split(",") if token.strip()]
        if tokens:
            return [int(token) for token in tokens]
    env_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_spec:
        tokens = [token.strip() for token in env_spec.split(",") if token.strip()]
        if tokens:
            return [int(token) for token in tokens]
    # 默认假设 GPU 0/1 可用（与机器配置一致），若不存在也不会报错，只是会重复使用。
    return [0, 1]


def estimate_gpu_cost(dataset_name: str) -> float:
    return DATASET_GPU_COST.get(dataset_name.lower(), DEFAULT_COST)


def build_phase0_mlp_jobs(seeds: Sequence[int]) -> List[Job]:
    seed_str = ",".join(str(s) for s in seeds)
    jobs: List[Job] = []
    for dataset in DEFAULT_PHASE0_DATASETS:
        cmd = [
            sys.executable,
            "experiments/phase0_mlp_full_supervised.py",
            "--datasets",
            dataset,
            "--seeds",
            seed_str,
        ]
        jobs.append(
            Job(
                name=f"{dataset}_phase0_mlp",
                cmd=cmd,
                dataset_name=dataset,
                gpu_cost=estimate_gpu_cost(dataset),
            )
        )
    return jobs


def build_stage1_synth_default_jobs(seeds: Sequence[int]) -> List[Job]:
    seed_str = ",".join(str(s) for s in seeds)
    models_str = ",".join(DEFAULT_STAGE1_MODELS)
    jobs: List[Job] = []
    for dataset in DEFAULT_STAGE1_SYNTH_DATASETS:
        cmd = [
            sys.executable,
            "experiments/stage1_multi_seed.py",
            "--datasets",
            dataset,
            "--models",
            models_str,
            "--seeds",
            seed_str,
            "--gpus",
            "none",
            "--device",
            "cuda",
        ]
        jobs.append(
            Job(
                name=f"{dataset}_stage1",
                cmd=cmd,
                dataset_name=dataset,
                gpu_cost=estimate_gpu_cost(dataset),
            )
        )
    return jobs


def _select_gpu(
    job: Job,
    gpu_ids: List[int],
    gpu_states: Dict[int, Dict[str, float]],
    max_jobs_per_gpu: int,
) -> Optional[int]:
    chosen_gpu: Optional[int] = None
    chosen_load: float = 0.0
    for gpu in gpu_ids:
        state = gpu_states[gpu]
        running = int(state["running"])
        cost = float(state["cost"])
        if running >= max_jobs_per_gpu:
            continue
        if cost + job.gpu_cost > GPU_CAPACITY and running > 0:
            continue
        if chosen_gpu is None or cost < chosen_load:
            chosen_gpu = gpu
            chosen_load = cost
    return chosen_gpu


def _print_job(prefix: str, gpu: int, job: Job, message: str) -> None:
    print(f"[GPU{gpu}][{prefix}][{job.name}] {message}")


def run_jobs_multi_gpu(
    jobs: Sequence[Job],
    gpu_ids: Optional[List[int]] = None,
    max_jobs_per_gpu: int = 2,
    dry_run: bool = False,
) -> None:
    if not jobs:
        print("[info] 没有需要运行的任务。")
        return
    gpu_ids = gpu_ids or parse_gpu_ids(None)
    if not gpu_ids:
        raise ValueError("至少需要一张 GPU 可用")
    gpu_states: Dict[int, Dict[str, float]] = {
        gpu: {"running": 0, "cost": 0.0} for gpu in gpu_ids
    }
    pending: List[Job] = list(jobs)
    running: List[Dict[str, object]] = []

    if dry_run:
        simulated_states = {gpu: {"running": 0, "cost": 0.0} for gpu in gpu_ids}
        for job in pending:
            gpu = _select_gpu(job, gpu_ids, simulated_states, max_jobs_per_gpu)
            if gpu is None:
                # 简化：若找不到合适的 GPU，就重置状态以模拟“等待直至 GPU 空闲”。
                simulated_states = {gpu_id: {"running": 0, "cost": 0.0} for gpu_id in gpu_ids}
                gpu = _select_gpu(job, gpu_ids, simulated_states, max_jobs_per_gpu)
            simulated_states[gpu]["running"] += 1
            simulated_states[gpu]["cost"] += job.gpu_cost
            _print_job("PLAN", gpu, job, " ".join(job.cmd))
            simulated_states[gpu]["running"] -= 1
            simulated_states[gpu]["cost"] = max(0.0, simulated_states[gpu]["cost"] - job.gpu_cost)
        print("[dry-run] 调度计划打印完成，未启动任何子进程。")
        return

    pending_idx = 0
    while pending or running:
        if pending_idx >= len(pending):
            pending_idx = 0
        scheduled = False
        while pending_idx < len(pending):
            job = pending[pending_idx]
            gpu = _select_gpu(job, gpu_ids, gpu_states, max_jobs_per_gpu)
            if gpu is None:
                pending_idx += 1
                continue
            scheduled = True
            pending.pop(pending_idx)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            start_time = time.time()
            _print_job("START", gpu, job, " ".join(job.cmd))
            proc = subprocess.Popen(job.cmd, env=env)
            gpu_states[gpu]["running"] += 1
            gpu_states[gpu]["cost"] += job.gpu_cost
            running.append({"proc": proc, "gpu": gpu, "job": job, "start": start_time})
        if running:
            time.sleep(1.0)
            finished = [entry for entry in running if entry["proc"].poll() is not None]
            for entry in finished:
                running.remove(entry)
                gpu = entry["gpu"]
                job = entry["job"]
                exit_code = entry["proc"].returncode
                elapsed = time.time() - entry["start"]
                gpu_states[gpu]["running"] -= 1
                gpu_states[gpu]["cost"] = max(0.0, gpu_states[gpu]["cost"] - job.gpu_cost)
                _print_job("DONE ", gpu, job, f"exit_code={exit_code} elapsed={elapsed:.1f}s")
                if exit_code != 0:
                    raise RuntimeError(f"任务 {job.name} 失败，退出码 {exit_code}")
        elif not scheduled and pending:
            # 没有任务可调度但也没有进程运行，说明 cost/限制过紧，强制等待片刻后重试。
            time.sleep(1.0)
            pending_idx = 0


def build_jobs(plan: str, seeds: Sequence[int]) -> List[Job]:
    plan = plan.lower()
    if plan == "phase0_mlp":
        return build_phase0_mlp_jobs(seeds)
    if plan == "stage1_synth_default":
        return build_stage1_synth_default_jobs(seeds)
    raise ValueError(f"未知的 plan: {plan}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多 GPU 实验调度器")
    parser.add_argument(
        "--plan",
        required=True,
        choices=["phase0_mlp", "stage1_synth_default"],
        help="预设任务计划",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="逗号或空格分隔的随机种子列表",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU 编号（缺省时使用 0,1）",
    )
    parser.add_argument(
        "--max-jobs-per-gpu",
        type=int,
        default=2,
        help="单卡最大并发任务数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印调度计划，不真正执行",
    )
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    jobs = build_jobs(args.plan, seeds)
    run_jobs_multi_gpu(
        jobs=jobs,
        gpu_ids=gpu_ids,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
