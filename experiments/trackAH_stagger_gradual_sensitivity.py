#!/usr/bin/env python
"""
NEXT_STAGE V12 - Track AH（必做）：stagger_gradual_frequent 补救（专项小扫）

固定 confirm-side（默认从 Track AG winner 自动读取，也可手动传参）：
- confirm_theta / confirm_window / confirm_cooldown

只扫 detector 敏感度：
- error.threshold ∈ {0.10,0.08,0.05,0.03,0.02}
- error.min_instances 固定 5

评估输出三种 tol 口径（start/mid/end）与 delay 分位数（P90/P99）。

实现说明：
- 训练实际跑 base_dataset=stagger_abrupt3，通过 cfg.stream_kwargs 指定 drift_type=gradual + drift_positions + transition_length
- tol 口径使用“区间 drift GT”推断 start/end/mid（与 V11 Track AD 一致）

输出：scripts/TRACKAH_STAGGER_GRADUAL_SENSITIVITY.csv（逐 seed）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track AH: stagger gradual sensitivity sweep (error.threshold)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAH_STAGGER_GRADUAL_SENSITIVITY.csv")

    p.add_argument("--trackag_csv", type=str, default="scripts/TRACKAG_CONFIRM_SIDE_NODRIFT.csv")
    p.add_argument("--confirm_theta", type=float, default=-1.0)
    p.add_argument("--confirm_window", type=int, default=-1)
    p.add_argument("--confirm_cooldown", type=int, default=-1)

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--n_samples", type=int, default=51200)
    p.add_argument("--concept_length", type=int, default=5000)
    p.add_argument("--transition_length", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)

    p.add_argument("--error_thresholds", type=str, default="0.10,0.08,0.05,0.03,0.02")
    p.add_argument("--error_min_instances", type=int, default=5)
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    return p.parse_args()


def parse_weights(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--weights 格式错误：{token}（期望 key=value）")
        k, v = token.split("=", 1)
        weights[k.strip()] = float(v.strip())
    if not weights:
        raise ValueError("--weights 不能为空")
    return weights


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(v: Any) -> Optional[int]:
    x = _safe_float(v)
    return None if x is None else int(x)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))  # type: ignore[arg-type]
    if not vals:
        return None
    if q <= 0:
        return float(vals[0])
    if q >= 1:
        return float(vals[-1])
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    w = pos - lo
    return float(vals[lo] * (1 - w) + vals[hi] * w)


def merge_events(events: Sequence[int], min_sep: int) -> List[int]:
    if not events:
        return []
    ev = sorted(int(x) for x in events)
    merged = [ev[0]]
    for x in ev[1:]:
        if x - merged[-1] < int(min_sep):
            continue
        merged.append(x)
    return merged


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def build_stagger_intervals(*, n_samples: int, concept_length: int, transition_length: int) -> Tuple[List[int], List[Dict[str, float]]]:
    step = int(concept_length) * 2
    starts = [int(x) for x in range(step, int(n_samples), step)]
    intervals: List[Dict[str, float]] = []
    for i, s in enumerate(starts):
        next_s = starts[i + 1] if i + 1 < len(starts) else int(n_samples)
        e = min(int(next_s), int(s + int(transition_length)))
        mid = 0.5 * (float(s) + float(e))
        intervals.append({"start": float(s), "end": float(e), "mid": float(mid)})
    return starts, intervals


def compute_interval_metrics(
    intervals: Sequence[Dict[str, float]],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Dict[str, Any]:
    starts = [int(d["start"]) for d in intervals]
    ends = [int(d["end"]) for d in intervals]
    mids = [float(d["mid"]) for d in intervals]

    confs = sorted(int(x) for x in confirmed)
    first_conf: List[Optional[int]] = []
    for i, s in enumerate(starts):
        end_window = starts[i + 1] if i + 1 < len(starts) else int(horizon)
        first_conf.append(first_event_in_range(confs, int(s), int(end_window)))

    miss_start = [0 if (fc is not None and fc <= s + tol) else 1 for fc, s in zip(first_conf, starts)]
    miss_mid = [0 if (fc is not None and float(fc) <= m + tol) else 1 for fc, m in zip(first_conf, mids)]
    miss_end = [0 if (fc is not None and fc <= e + tol) else 1 for fc, e in zip(first_conf, ends)]

    def _delays(anchor: Sequence[float]) -> List[float]:
        out: List[float] = []
        for i, fc in enumerate(first_conf):
            if fc is None:
                continue
            out.append(float(fc) - float(anchor[i]))
        return out

    d_start = _delays([float(x) for x in starts])
    d_mid = _delays([float(x) for x in mids])
    d_end = _delays([float(x) for x in ends])

    return {
        "n_drifts": int(len(intervals)),
        "miss_tol500_start": (float(sum(miss_start)) / len(miss_start)) if miss_start else None,
        "miss_tol500_mid": (float(sum(miss_mid)) / len(miss_mid)) if miss_mid else None,
        "miss_tol500_end": (float(sum(miss_end)) / len(miss_end)) if miss_end else None,
        "delay_start_P90": percentile(d_start, 0.90),
        "delay_start_P99": percentile(d_start, 0.99),
        "delay_mid_P90": percentile(d_mid, 0.90),
        "delay_mid_P99": percentile(d_mid, 0.99),
        "delay_end_P90": percentile(d_end, 0.90),
        "delay_end_P99": percentile(d_end, 0.99),
    }


def choose_confirm_from_trackag(trackag_csv: Path) -> Optional[Tuple[float, int, int]]:
    if not trackag_csv.exists():
        return None
    rows = list(csv.DictReader(trackag_csv.open("r", encoding="utf-8")))
    # 复制 V12 规则：先约束 drift（sea+sine miss==0 且 conf_P90<500），再最小化 no-drift rate（次选最大化 no-drift MTFA）
    eligible: List[Dict[str, Any]] = []
    for r in rows:
        sea_miss = _safe_float(r.get("sea_miss_tol500_mean"))
        sine_miss = _safe_float(r.get("sine_miss_tol500_mean"))
        sea_conf = _safe_float(r.get("sea_conf_P90_mean"))
        sine_conf = _safe_float(r.get("sine_conf_P90_mean"))
        if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
            continue
        if not (sea_miss <= 0.0 + 1e-12 and sine_miss <= 0.0 + 1e-12 and sea_conf < 500 and sine_conf < 500):
            continue
        eligible.append(r)
    if not eligible:
        return None

    def nd_rate(r: Dict[str, Any]) -> float:
        v = _safe_float(r.get("no_drift_confirm_rate_per_10k_mean"))
        return float(v) if v is not None else float("inf")

    def nd_mtfa(r: Dict[str, Any]) -> float:
        v = _safe_float(r.get("no_drift_MTFA_win_mean"))
        return float(v) if v is not None else float("-inf")

    eligible.sort(
        key=lambda r: (
            nd_rate(r),
            -nd_mtfa(r),
            float(_safe_float(r.get("confirm_theta")) or float("inf")),
            int(_safe_int(r.get("confirm_window")) or 10**9),
            int(_safe_int(r.get("confirm_cooldown")) or 10**9),
        )
    )
    w = eligible[0]
    return float(w["confirm_theta"]), int(float(w["confirm_window"])), int(float(w["confirm_cooldown"]))


def ensure_log(
    exp_run: ExperimentRun,
    dataset_name: str,
    seed: int,
    base_cfg: ExperimentConfig,
    *,
    monitor_preset: str,
    confirm_theta: float,
    confirm_window: int,
    confirm_cooldown: int,
    trigger_weights: Dict[str, float],
    stream_kwargs: Dict[str, Any],
    device: str,
) -> Path:
    run_paths = exp_run.prepare_dataset_run(dataset_name, "ts_drift_adapt", seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    weights = dict(trigger_weights)
    weights["confirm_cooldown"] = float(int(confirm_cooldown))
    cfg = replace(
        base_cfg,
        dataset_name=str(dataset_name),
        dataset_type=str(base_cfg.dataset_type),
        stream_kwargs=dict(stream_kwargs),
        model_variant="ts_drift_adapt",
        seed=int(seed),
        log_path=str(log_path),
        monitor_preset=str(monitor_preset),
        trigger_mode="two_stage",
        trigger_weights=weights,
        trigger_threshold=float(confirm_theta),
        confirm_window=int(confirm_window),
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    trackag_csv = Path(args.trackag_csv)

    winner = None
    if float(args.confirm_theta) > 0 and int(args.confirm_window) > 0 and int(args.confirm_cooldown) > 0:
        winner = (float(args.confirm_theta), int(args.confirm_window), int(args.confirm_cooldown))
    else:
        winner = choose_confirm_from_trackag(trackag_csv)
    if winner is None:
        raise RuntimeError("无法自动从 Track AG 选择 winner（请显式传 --confirm_theta/--confirm_window/--confirm_cooldown）")
    confirm_theta, confirm_window, confirm_cooldown = winner

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    thresholds = [float(x) for x in str(args.error_thresholds).split(",") if str(x).strip()]
    mi = int(args.error_min_instances)
    weights_base = parse_weights(str(args.weights))

    cfg_map = build_cfg_map(1)
    base_cfg = cfg_map["stagger_abrupt3"]

    horizon = int(args.n_samples)
    drift_positions, intervals = build_stagger_intervals(n_samples=horizon, concept_length=int(args.concept_length), transition_length=int(args.transition_length))
    stream_kwargs_base = {
        "drift_type": "gradual",
        "drift_positions": drift_positions,
        "transition_length": int(args.transition_length),
    }

    rows_out: List[Dict[str, Any]] = []
    for thr in thresholds:
        preset = f"error_divergence_ph_meta@error.threshold={thr:.2f},error.min_instances={mi}"
        exp_run = create_experiment_run(
            experiment_name="trackAH_stagger_gradual_sensitivity",
            results_root=Path(args.results_root),
            logs_root=Path(args.logs_root),
            run_name=f"thr{thr:.2f}_mi{mi}_theta{confirm_theta:.2f}_w{confirm_window}_cd{confirm_cooldown}",
        )
        for seed in seeds:
            log_path = ensure_log(
                exp_run,
                dataset_name="stagger_abrupt3",
                seed=int(seed),
                base_cfg=base_cfg,
                monitor_preset=preset,
                confirm_theta=float(confirm_theta),
                confirm_window=int(confirm_window),
                confirm_cooldown=int(confirm_cooldown),
                trigger_weights=weights_base,
                stream_kwargs=stream_kwargs_base,
                device=str(args.device),
            )
            summ = read_run_summary(log_path)
            confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
            confirmed = merge_events(confirmed_raw, int(args.min_separation))
            m = compute_interval_metrics(intervals, confirmed, horizon=horizon, tol=int(args.tol))
            rows_out.append(
                {
                    "track": "AH",
                    "profile": "stagger_gradual_frequent",
                    "base_dataset_name": "stagger_abrupt3",
                    "unit": "sample_idx",
                    "seed": int(seed),
                    "run_id": exp_run.run_id,
                    "log_path": str(log_path),
                    "monitor_preset": preset,
                    "error.threshold": float(thr),
                    "error.min_instances": int(mi),
                    "confirm_theta": float(confirm_theta),
                    "confirm_window": int(confirm_window),
                    "confirm_cooldown": int(confirm_cooldown),
                    "transition_length": int(args.transition_length),
                    "tol": int(args.tol),
                    **m,
                }
            )

    if not rows_out:
        print("[warn] no records")
        return 0
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in rows_out:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[done] wrote {out_csv} rows={len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

