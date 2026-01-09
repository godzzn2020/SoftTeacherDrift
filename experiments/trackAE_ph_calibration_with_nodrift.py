#!/usr/bin/env python
"""
NEXT_STAGE V11 - Track AE：no-drift 约束下的 PH 校准（降低误报密度）

固定检测结构：
- trigger=two_stage(candidate=OR, confirm=weighted)
- confirm_theta=0.50, confirm_window=1, confirm_cooldown=200

扫 error PH 两个参数：
- error.threshold ∈ {0.05,0.08,0.10}
- error.min_instances ∈ {5,10,25}
divergence 保持默认（期望为 div_thr=0.05, div_min=30）

数据集（先生成并落盘到临时目录，再用软链接暴露给 loader；不做 data/synthetic 临时替换）：
- drift：sea_gradual_frequent、sine_gradual_frequent（seeds=1..5）
- no-drift：sea_nodrift（seeds=1..5）

输出：scripts/TRACKAE_PH_CALIBRATION_WITH_NODRIFT.csv（逐 seed）
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

from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track AE: PH calibration with no-drift constraint")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAE_PH_CALIBRATION_WITH_NODRIFT.csv")

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)

    p.add_argument("--confirm_theta", type=float, default=0.5)
    p.add_argument("--confirm_window", type=int, default=1)
    p.add_argument("--confirm_cooldown", type=int, default=200)

    p.add_argument("--error_thresholds", type=str, default="0.05,0.08,0.10")
    p.add_argument("--error_min_instances", type=str, default="5,10,25")

    p.add_argument("--concept_length", type=int, default=5000)
    p.add_argument("--transition_length", type=int, default=2000)

    p.add_argument("--active_synth_root", type=str, default="data/synthetic")
    p.add_argument("--tmp_synth_root", type=str, default="data/synthetic_v11_tmp/trackAE")
    return p.parse_args()


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


def per_drift_first_delays(
    gt_drifts: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Tuple[List[float], List[int]]:
    gt = sorted(int(d) for d in gt_drifts)
    confs = sorted(int(x) for x in confirmed)
    delays: List[float] = []
    miss_flags: List[int] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_f = first_event_in_range(confs, g, end)
        if first_f is None:
            delays.append(float(end - g))
        else:
            delays.append(float(first_f - g))
        tol_end = min(end, g + int(tol) + 1)
        miss_flags.append(0 if first_event_in_range(confs, g, tol_end) is not None else 1)
    return delays, miss_flags


def mtfa_from_false_alarms(detections: Sequence[int], horizon: int) -> Optional[float]:
    dets = sorted(int(x) for x in detections)
    if horizon <= 0:
        return None
    if len(dets) >= 2:
        gaps = [b - a for a, b in zip(dets[:-1], dets[1:])]
        return float(sum(gaps) / len(gaps)) if gaps else None
    if len(dets) == 1:
        return float(horizon - dets[0])
    return None


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def default_gt_starts(n_samples: int) -> List[int]:
    # 对齐 sea_abrupt4 / sine_abrupt4 默认：5 段 -> 4 个 drift start
    step = max(1, int(n_samples) // 5)
    return [int(step), int(2 * step), int(3 * step), int(4 * step)]


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


def ensure_log(
    exp_run: ExperimentRun,
    dataset_name: str,
    seed: int,
    base_cfg: ExperimentConfig,
    *,
    monitor_preset: str,
    trigger_threshold: float,
    trigger_weights: Dict[str, float],
    confirm_window: int,
    stream_kwargs: Dict[str, Any],
    device: str,
) -> Path:
    run_paths = exp_run.prepare_dataset_run(dataset_name, "ts_drift_adapt", seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    cfg = replace(
        base_cfg,
        dataset_name=str(dataset_name),
        dataset_type=str(base_cfg.dataset_type),
        stream_kwargs=dict(stream_kwargs),
        model_variant="ts_drift_adapt",
        seed=seed,
        log_path=str(log_path),
        monitor_preset=str(monitor_preset),
        trigger_mode="two_stage",
        trigger_weights=trigger_weights,
        trigger_threshold=float(trigger_threshold),
        confirm_window=int(confirm_window),
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)

    thrs = [float(x) for x in str(args.error_thresholds).split(",") if str(x).strip()]
    mins = [int(x) for x in str(args.error_min_instances).split(",") if str(x).strip()]

    tuned_base = "error_divergence_ph_meta"
    weights_base = {"error_rate": 0.5, "divergence": 0.3, "teacher_entropy": 0.2, "confirm_cooldown": float(int(args.confirm_cooldown))}
    theta = float(args.confirm_theta)
    window = int(args.confirm_window)
    warmup = int(args.warmup_samples)
    tol = int(args.tol)

    cfg_map = build_cfg_map(1)
    base_cfg_sea = cfg_map["sea_abrupt4"]
    base_cfg_sine = cfg_map["sine_abrupt4"]

    records: List[Dict[str, Any]] = []
    for thr in thrs:
        for mi in mins:
            preset = f"{tuned_base}@error.threshold={thr:.2f},error.min_instances={int(mi)}"
            exp_run = create_experiment_run(
                experiment_name="trackAE_ph_calibration_with_nodrift",
                results_root=results_root,
                logs_root=logs_root,
                run_name=f"thr{thr:.2f}_mi{int(mi)}",
            )

            # drift datasets
            for dataset_alias, base_dataset_name, base_cfg in [
                ("sea_gradual_frequent", "sea_abrupt4", base_cfg_sea),
                ("sine_gradual_frequent", "sine_abrupt4", base_cfg_sine),
            ]:
                for seed in seeds:
                    horizon = int(args.n_samples)
                    gt_starts = default_gt_starts(horizon)
                    stream_kwargs = {"drift_type": "gradual", "transition_length": int(args.transition_length)}
                    log_path = ensure_log(
                        exp_run,
                        base_dataset_name,
                        int(seed),
                        base_cfg,
                        monitor_preset=preset,
                        trigger_threshold=theta,
                        trigger_weights=dict(weights_base),
                        confirm_window=window,
                        stream_kwargs=stream_kwargs,
                        device=str(args.device),
                    )
                    summ = read_run_summary(log_path)
                    confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                    confirmed = merge_events(confirmed_raw, int(args.min_separation))
                    delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=tol) if gt_starts else ([], [])
                    miss = (float(sum(miss_flags)) / len(miss_flags)) if miss_flags else None
                    conf_p90 = percentile(delays, 0.90) if delays else None
                    win_m = (
                        compute_detection_metrics(gt_starts, confirmed_raw, int(horizon))
                        if horizon > 0 and gt_starts
                        else {"MTFA": None}
                    )
                    records.append(
                        {
                            "track": "AE",
                            "dataset": dataset_alias,
                            "base_dataset_name": base_dataset_name,
                            "dataset_kind": "drift",
                            "unit": "sample_idx",
                            "seed": int(seed),
                            "run_id": exp_run.run_id,
                            "experiment_name": exp_run.experiment_name,
                            "log_path": str(log_path),
                            "monitor_preset": preset,
                            "error.threshold": float(thr),
                            "error.min_instances": int(mi),
                            "trigger_mode": "two_stage",
                            "confirm_theta": float(theta),
                            "confirm_window": int(window),
                            "confirm_cooldown": int(args.confirm_cooldown),
                            "horizon": int(horizon),
                            "acc_final": summ.get("acc_final"),
                            f"acc_min@{warmup}": acc_min_after_warmup(summ, warmup),
                            "miss_tol500": miss,
                            "conf_P90": conf_p90,
                            "MTFA_win": win_m.get("MTFA"),
                            "stream_kwargs": json.dumps(stream_kwargs, ensure_ascii=False, sort_keys=True),
                        }
                    )

            # no-drift dataset（只看误报/MTFA/acc）
            dataset_alias = "sea_nodrift"
            base_dataset_name = "sea_abrupt4"
            for seed in seeds:
                horizon = int(args.n_samples)
                stream_kwargs = {"concept_ids": [0], "concept_length": int(args.n_samples), "drift_type": "abrupt"}
                log_path = ensure_log(
                    exp_run,
                    base_dataset_name,
                    int(seed),
                    base_cfg_sea,
                    monitor_preset=preset,
                    trigger_threshold=theta,
                    trigger_weights=dict(weights_base),
                    confirm_window=window,
                    stream_kwargs=stream_kwargs,
                    device=str(args.device),
                )
                summ = read_run_summary(log_path)
                confirmed = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                cc = int(summ.get("confirmed_count_total") or len(confirmed))
                rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                records.append(
                    {
                        "track": "AE",
                        "dataset": dataset_alias,
                        "base_dataset_name": base_dataset_name,
                        "dataset_kind": "nodrift",
                        "unit": "sample_idx",
                        "seed": int(seed),
                        "run_id": exp_run.run_id,
                        "experiment_name": exp_run.experiment_name,
                        "log_path": str(log_path),
                        "monitor_preset": preset,
                        "error.threshold": float(thr),
                        "error.min_instances": int(mi),
                        "trigger_mode": "two_stage",
                        "confirm_theta": float(theta),
                        "confirm_window": int(window),
                        "confirm_cooldown": int(args.confirm_cooldown),
                        "horizon": int(horizon),
                        "acc_final": summ.get("acc_final"),
                        "confirm_rate_per_10k": rate,
                        "confirmed_count_total": cc,
                        "MTFA_win": mtfa_from_false_alarms(merge_events(confirmed, int(args.min_separation)), int(horizon)),
                        "stream_kwargs": json.dumps(stream_kwargs, ensure_ascii=False, sort_keys=True),
                    }
                )

    if not records:
        print("[warn] no records")
        return 0
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[done] wrote {out_csv} rows={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
