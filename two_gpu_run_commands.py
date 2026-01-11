#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CommandJob:
    index: int
    command: str
    job_name: str
    dataset_name: str
    gpu_cost: float


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


def _sanitize_token(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return "unknown"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", ".", "+"}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._-")
    return cleaned or "unknown"


def estimate_gpu_cost(dataset_name: str) -> float:
    return float(DATASET_GPU_COST.get(str(dataset_name).lower(), DEFAULT_COST))


def _parse_arg_tokens(command: str) -> List[str]:
    try:
        return shlex.split(command)
    except Exception:
        return command.split()


def _extract_arg_values(tokens: Sequence[str], keys: Sequence[str]) -> List[str]:
    values: List[str] = []
    key_set = set(keys)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            k, v = tok.split("=", 1)
            if k in key_set and v.strip():
                values.append(v.strip())
            i += 1
            continue
        if tok in key_set:
            if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("--"):
                v = str(tokens[i + 1]).strip()
                if v:
                    values.append(v)
                i += 2
                continue
        i += 1
    return values


def _infer_dataset_and_seed(command: str) -> Tuple[str, Optional[str]]:
    tokens = _parse_arg_tokens(command)
    ds_vals = _extract_arg_values(tokens, ["--dataset_name", "--datasets"])
    seed_vals = _extract_arg_values(tokens, ["--seed", "--seeds"])

    dataset_name = ""
    if ds_vals:
        # --datasets 可能是逗号分隔；优先取第一个 token 作为“主 dataset”
        first = str(ds_vals[0]).strip()
        dataset_name = first.split(",")[0].strip()

    seed = str(seed_vals[0]).strip() if seed_vals else None
    return dataset_name, seed


def _short_hash(text: str, n: int = 8) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return h[: max(4, int(n))]


def _build_job_name(dataset_name: str, seed: Optional[str], command: str) -> str:
    parts: List[str] = []
    if dataset_name:
        parts.append(_sanitize_token(dataset_name))
    if seed:
        parts.append(_sanitize_token(f"seed{seed}"))
    parts.append(_short_hash(command, 8))
    return "_".join(parts)


def _read_commands(path: Path) -> List[CommandJob]:
    raw = path.read_text(encoding="utf-8").splitlines()
    jobs: List[CommandJob] = []
    for i, line in enumerate(raw, start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cmd = line.rstrip("\n")
        dataset_name, seed = _infer_dataset_and_seed(cmd)
        cost = estimate_gpu_cost(dataset_name) if dataset_name else DEFAULT_COST
        job_name = _build_job_name(dataset_name, seed, cmd)
        jobs.append(
            CommandJob(
                index=i,
                command=cmd,
                job_name=job_name,
                dataset_name=dataset_name or "unknown",
                gpu_cost=float(cost),
            )
        )
    return jobs


def parse_gpu_ids(spec: Optional[str]) -> List[int]:
    if spec:
        tokens = [token.strip() for token in str(spec).split(",") if token.strip()]
        if tokens:
            return [int(token) for token in tokens]
    env_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_spec:
        tokens = [token.strip() for token in env_spec.split(",") if token.strip()]
        if tokens:
            return [int(token) for token in tokens]
    return [0, 1]


def _open_log_files(log_dir: Path, job: CommandJob, gpu: int) -> tuple[object, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"job_{job.index:04d}_gpu{gpu}_{_sanitize_token(job.job_name)[:80]}"
    out_path = log_dir / f"{stem}.out.log"
    err_path = log_dir / f"{stem}.err.log"
    out_f = out_path.open("w", encoding="utf-8")
    err_f = err_path.open("w", encoding="utf-8")
    return out_f, err_f


def _start_job(job: CommandJob, gpu: int, log_dir: Optional[Path]) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    stdout = None
    stderr = None
    out_f = None
    err_f = None
    if log_dir is not None:
        out_f, err_f = _open_log_files(log_dir, job, gpu)
        stdout = out_f
        stderr = err_f

    cmd = ["bash", "-lc", job.command]
    print(f"[GPU{gpu}][START][{job.job_name}][line={job.index}] {job.command}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
    return {
        "proc": proc,
        "gpu": gpu,
        "job": job,
        "start": time.time(),
        "out_f": out_f,
        "err_f": err_f,
    }


def _close_logs(entry: dict) -> None:
    for k in ("out_f", "err_f"):
        f = entry.get(k)
        try:
            if f is not None:
                f.close()
        except Exception:
            pass


def _select_gpu(
    job: CommandJob,
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
        if cost + float(job.gpu_cost) > GPU_CAPACITY and running > 0:
            continue
        if chosen_gpu is None or cost < chosen_load:
            chosen_gpu = gpu
            chosen_load = cost
    return chosen_gpu


def _print_job(prefix: str, gpu: int, job: CommandJob, message: str) -> None:
    print(f"[GPU{gpu}][{prefix}][{job.job_name}] {message}", flush=True)


def _extract_output_paths(command: str) -> Dict[str, str]:
    tokens = _parse_arg_tokens(command)
    out: Dict[str, str] = {}
    for k in ("--log_path", "--out_csv", "--output_dir", "--log_dir"):
        vals = _extract_arg_values(tokens, [k])
        if vals:
            out[k] = str(vals[0])
    return out


def _check_output_conflicts(jobs: Sequence[CommandJob]) -> None:
    seen: Dict[str, List[int]] = {}
    for job in jobs:
        outs = _extract_output_paths(job.command)
        if not outs:
            print(
                f"[warn] line={job.index} job={job.job_name} 未解析到输出参数（--log_path/--out_csv/--output_dir/--log_dir）；强烈建议显式指定输出路径以避免并行写冲突。",
                file=sys.stderr,
            )
            continue
        for _k, v in outs.items():
            key = str(v)
            seen.setdefault(key, []).append(int(job.index))
    conflicts = {path: lines for path, lines in seen.items() if len(lines) >= 2}
    if conflicts:
        print("[error] 检测到输出路径冲突（同一路径被多条命令使用），已 fail-fast 退出：", file=sys.stderr)
        for path, lines in sorted(conflicts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines_str = ",".join(str(x) for x in sorted(lines))
            print(f"  path={path} lines={lines_str}", file=sys.stderr)
        raise SystemExit(2)


def run_jobs_multi_gpu(
    jobs: Sequence[CommandJob],
    gpu_ids: List[int],
    *,
    max_jobs_per_gpu: int = 1,
    poll_seconds: float = 0.5,
    fail_fast: bool = True,
    dry_run: bool = False,
    log_dir: Optional[Path] = None,
) -> None:
    if not jobs:
        print("[info] 没有需要运行的任务。")
        return
    if not gpu_ids:
        raise ValueError("至少需要一张 GPU 可用")
    if max_jobs_per_gpu <= 0:
        raise ValueError("--max-jobs-per-gpu 必须 >= 1")

    print(f"[info] 可用 GPU: {gpu_ids}")
    print(f"[info] GPU_CAPACITY={GPU_CAPACITY:g} DEFAULT_COST={DEFAULT_COST:g} max_jobs_per_gpu={max_jobs_per_gpu}")
    for gpu in gpu_ids:
        print(f"[info] GPU{gpu} state: running=0 cost=0.0 capacity={GPU_CAPACITY:g}")

    _check_output_conflicts(jobs)

    gpu_states: Dict[int, Dict[str, float]] = {gpu: {"running": 0, "cost": 0.0} for gpu in gpu_ids}

    if dry_run:
        simulated_states: Dict[int, Dict[str, float]] = {gpu: {"running": 0, "cost": 0.0} for gpu in gpu_ids}
        for job in jobs:
            gpu = _select_gpu(job, gpu_ids, simulated_states, max_jobs_per_gpu)
            if gpu is None:
                simulated_states = {gpu_id: {"running": 0, "cost": 0.0} for gpu_id in gpu_ids}
                gpu = _select_gpu(job, gpu_ids, simulated_states, max_jobs_per_gpu)
            assert gpu is not None
            simulated_states[gpu]["running"] += 1
            simulated_states[gpu]["cost"] += float(job.gpu_cost)
            _print_job("PLAN", gpu, job, f"cost={job.gpu_cost:g} dataset={job.dataset_name} :: {job.command}")
        print("[dry-run] 调度计划打印完成，未启动任何子进程。")
        return

    pending: List[CommandJob] = list(jobs)
    running: List[Dict[str, object]] = []
    pending_idx = 0

    try:
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
                entry = _start_job(job, gpu, log_dir)
                gpu_states[gpu]["running"] += 1
                gpu_states[gpu]["cost"] += float(job.gpu_cost)
                running.append(entry)

            if running:
                time.sleep(max(0.1, float(poll_seconds)))
                finished = [entry for entry in running if entry["proc"].poll() is not None]
                for entry in finished:
                    running.remove(entry)
                    gpu = int(entry["gpu"])
                    job = entry["job"]
                    exit_code = int(entry["proc"].returncode or 0)
                    elapsed = float(time.time() - float(entry["start"]))
                    gpu_states[gpu]["running"] -= 1
                    gpu_states[gpu]["cost"] = max(0.0, float(gpu_states[gpu]["cost"]) - float(job.gpu_cost))
                    _close_logs(entry)
                    _print_job("DONE ", gpu, job, f"exit_code={exit_code} elapsed={elapsed:.1f}s")
                    if exit_code != 0:
                        if fail_fast:
                            raise RuntimeError(f"任务失败：job={job.job_name} line={job.index} exit_code={exit_code}")
                        raise SystemExit(exit_code)
            elif not scheduled and pending:
                time.sleep(max(0.1, float(poll_seconds)))
                pending_idx = 0
    except Exception as e:
        if fail_fast:
            for entry in list(running):
                try:
                    entry["proc"].terminate()
                except Exception:
                    pass
            time.sleep(0.2)
            for entry in list(running):
                try:
                    if entry["proc"].poll() is None:
                        entry["proc"].kill()
                except Exception:
                    pass
                _close_logs(entry)
        raise e


def main() -> None:
    p = argparse.ArgumentParser(description="两卡并行命令调度器（按 CUDA_VISIBLE_DEVICES 绑定）")
    p.add_argument("--cmd-file", type=str, required=True, help="命令列表文件：每行一条命令（支持 # 注释）")
    p.add_argument("--gpu-ids", type=str, default=None, help="逗号分隔 GPU 编号（缺省时使用 CUDA_VISIBLE_DEVICES 或 0,1）")
    p.add_argument("--max-jobs-per-gpu", type=int, default=1, help="单卡最大并发（默认 1，更稳）")
    p.add_argument("--poll-seconds", type=float, default=0.5, help="轮询间隔秒数")
    p.add_argument("--no-fail-fast", action="store_true", help="遇到失败不立即终止其它任务（不推荐）")
    p.add_argument("--dry-run", action="store_true", help="只打印分配，不执行")
    p.add_argument("--log-dir", type=str, default="", help="保存 stdout/stderr 日志的目录（空表示不落盘）")
    args = p.parse_args()

    cmd_file = Path(args.cmd_file)
    if not cmd_file.exists():
        raise FileNotFoundError(str(cmd_file))

    jobs = _read_commands(cmd_file)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    log_dir = Path(args.log_dir) if str(args.log_dir).strip() else None

    run_jobs_multi_gpu(
        jobs=jobs,
        gpu_ids=list(gpu_ids),
        max_jobs_per_gpu=int(args.max_jobs_per_gpu),
        poll_seconds=float(args.poll_seconds),
        fail_fast=not bool(args.no_fail_fast),
        dry_run=bool(args.dry_run),
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
