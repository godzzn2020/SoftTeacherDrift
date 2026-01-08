#!/usr/bin/env python
"""
NEXT_STAGE V7 - Track S: 自适应 cooldown（最小实现）

固定检测配置：
- tuned PH：error.threshold=0.05, error.min_instances=5（divergence 默认）
- trigger_mode=two_stage（candidate=OR，confirm=weighted）
- confirm_theta=0.5, confirm_window=1

对比组：
- fixed_cd0
- fixed_cd200
- adaptive_cd（按最近窗口 confirm_rate_per_10k 动态切换 cooldown）

输出：scripts/TRACKS_ADAPTIVE_COOLDOWN.csv（run 粒度）
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

from data.streams import generate_default_abrupt_synth_datasets
from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track S: adaptive cooldown")
    p.add_argument("--sea_dataset", type=str, default="sea_abrupt4")
    p.add_argument("--insects_dataset", type=str, default="INSECTS_abrupt_balanced")
    p.add_argument("--sea_seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--insects_seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--sea_model_variant", type=str, default="ts_drift_adapt")
    p.add_argument("--severity_model_variant", type=str, default="ts_drift_adapt_severity")
    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_theta", type=float, default=0.5)
    p.add_argument("--confirm_window", type=int, default=1)
    p.add_argument("--fixed_cd", nargs="+", type=int, default=[0, 200])
    # adaptive config（单位：sample_idx / confirm_rate_per_10k）
    p.add_argument("--adaptive_window", type=int, default=10000)
    p.add_argument("--adaptive_lower_per10k", type=float, default=10.0)
    p.add_argument("--adaptive_upper_per10k", type=float, default=25.0)
    p.add_argument("--adaptive_cooldown_low", type=int, default=200)
    p.add_argument("--adaptive_cooldown_high", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--synthetic_root", type=str, default="data/synthetic")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKS_ADAPTIVE_COOLDOWN.csv")
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--recovery_W", type=int, default=1000)
    p.add_argument("--pre_window", type=int, default=1000)
    p.add_argument("--roll_points", type=int, default=5)
    p.add_argument("--insects_meta", type=str, default="datasets/real/INSECTS_abrupt_balanced.json")
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


def load_synth_meta(synth_root: Path, dataset: str, seed: int) -> Tuple[List[int], int]:
    meta_path = synth_root / dataset / f"{dataset}__seed{seed}_meta.json"
    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    drifts = [int(d["start"]) for d in obj.get("drifts", [])]
    horizon = int(obj.get("n_samples", 0) or 0)
    return drifts, horizon


def load_insects_positions(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    positions = obj.get("positions") or []
    return [int(x) for x in positions]


def merge_events(events: Sequence[int], min_sep: int) -> List[int]:
    if not events:
        return []
    ev = sorted(int(x) for x in events)
    merged = [ev[0]]
    for x in ev[1:]:
        if x - merged[-1] < min_sep:
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


def per_drift_delays_and_miss(
    gt_drifts: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Tuple[List[Optional[int]], List[int]]:
    gt = sorted(int(d) for d in gt_drifts)
    confs = sorted(int(x) for x in confirmed)
    delays: List[Optional[int]] = []
    miss_flags: List[int] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_conf = first_event_in_range(confs, g, end)
        delays.append(None if first_conf is None else int(first_conf - g))
        tol_end = min(end, g + int(tol) + 1)
        miss_flags.append(0 if first_event_in_range(confs, g, tol_end) is not None else 1)
    return delays, miss_flags


def percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
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


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def gap_stats(confirmed: Sequence[int]) -> Tuple[Optional[float], Optional[float]]:
    ev = sorted(int(x) for x in confirmed)
    if len(ev) < 2:
        return None, None
    gaps = [float(b - a) for a, b in zip(ev[:-1], ev[1:])]
    gaps_sorted = sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else None
    p10 = percentile(gaps_sorted, 0.10) if gaps_sorted else None
    return median, p10


def compute_recovery_metrics(
    acc_series: Sequence[Tuple[int, float]],
    drifts: Sequence[int],
    *,
    W: int,
    pre_window: int,
    roll_points: int,
) -> Dict[str, Optional[float]]:
    if not acc_series or not drifts:
        return {
            "post_mean_acc": None,
            "post_min_acc": None,
            "recovery_time_to_pre90": None,
        }
    acc_map = list(acc_series)
    post_means: List[Optional[float]] = []
    post_mins: List[Optional[float]] = []
    rec_times: List[Optional[float]] = []

    for g in drifts:
        post = [a for (x, a) in acc_map if g <= x <= g + W]
        if post:
            post_means.append(float(statistics.mean(post)))
            post_mins.append(float(min(post)))
        else:
            post_means.append(None)
            post_mins.append(None)

        pre = [a for (x, a) in acc_map if (g - pre_window) <= x < g]
        if not pre:
            rec_times.append(None)
            continue
        pre_mean = float(statistics.mean(pre))
        thr = 0.9 * pre_mean
        post_points = [(x, a) for (x, a) in acc_map if g <= x <= g + W]
        found: Optional[float] = None
        for idx in range(len(post_points)):
            start = max(0, idx - max(1, int(roll_points)) + 1)
            window_vals = [a for _, a in post_points[start : idx + 1]]
            if window_vals and (sum(window_vals) / len(window_vals)) >= thr:
                found = float(post_points[idx][0] - g)
                break
        rec_times.append(found)

    post_mean_mu, _ = mean_std(post_means)
    post_min_mu, _ = mean_std(post_mins)
    rec_mu, _ = mean_std(rec_times)
    return {
        "post_mean_acc": post_mean_mu,
        "post_min_acc": post_min_mu,
        "recovery_time_to_pre90": rec_mu,
    }


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


def ensure_log(
    exp_run: ExperimentRun,
    dataset: str,
    seed: int,
    base_cfg: ExperimentConfig,
    *,
    model_variant: str,
    monitor_preset: str,
    trigger_threshold: float,
    trigger_weights: Dict[str, float],
    confirm_window: int,
    use_severity_v2: bool,
    severity_gate: str,
    severity_gate_min_streak: int,
    device: str,
) -> Path:
    run_paths = exp_run.prepare_dataset_run(dataset, model_variant, seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    cfg = replace(
        base_cfg,
        model_variant=model_variant,
        seed=seed,
        log_path=str(log_path),
        monitor_preset=monitor_preset,
        trigger_mode="two_stage",
        trigger_weights=trigger_weights,
        trigger_threshold=float(trigger_threshold),
        confirm_window=int(confirm_window),
        use_severity_v2=bool(use_severity_v2),
        severity_gate=str(severity_gate),
        severity_gate_min_streak=int(severity_gate_min_streak),
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)
    weights_base = parse_weights(args.weights)
    insects_positions = load_insects_positions(Path(args.insects_meta))

    generate_default_abrupt_synth_datasets(seeds=list(args.sea_seeds), out_root=str(synth_root))

    groups: List[Dict[str, Any]] = []
    for cd in [int(x) for x in args.fixed_cd]:
        groups.append(
            {
                "group": f"fixed_cd{cd}",
                "cooldown": int(cd),
                "adaptive": False,
            }
        )
    groups.append(
        {
            "group": "adaptive_cd",
            "cooldown": int(args.adaptive_cooldown_low),
            "adaptive": True,
        }
    )

    records: List[Dict[str, Any]] = []
    for g in groups:
        group = str(g["group"])
        base_cd = int(g["cooldown"])
        adaptive = bool(g["adaptive"])

        exp_run = create_experiment_run(
            experiment_name="trackS_adaptive_cooldown",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{group}_theta{float(args.confirm_theta):.2f}_w{int(args.confirm_window)}",
        )

        # sea
        for seed in list(args.sea_seeds):
            cfg_map = build_cfg_map(seed)
            ds = str(args.sea_dataset)
            if ds.lower() not in cfg_map:
                raise ValueError(f"未知 dataset: {ds}")
            base_cfg = cfg_map[ds.lower()]
            weights = dict(weights_base)
            weights["confirm_cooldown"] = float(base_cd)
            if adaptive:
                weights["adaptive_cooldown"] = 1.0
                weights["adaptive_window"] = float(args.adaptive_window)
                weights["adaptive_lower_per10k"] = float(args.adaptive_lower_per10k)
                weights["adaptive_upper_per10k"] = float(args.adaptive_upper_per10k)
                weights["adaptive_cooldown_low"] = float(args.adaptive_cooldown_low)
                weights["adaptive_cooldown_high"] = float(args.adaptive_cooldown_high)

            log_path = ensure_log(
                exp_run,
                ds,
                seed,
                base_cfg,
                model_variant=str(args.sea_model_variant),
                monitor_preset=str(args.monitor_preset),
                trigger_threshold=float(args.confirm_theta),
                trigger_weights=weights,
                confirm_window=int(args.confirm_window),
                use_severity_v2=False,
                severity_gate="none",
                severity_gate_min_streak=1,
                device=str(args.device),
            )
            summ = read_run_summary(log_path)
            horizon = int(summ.get("horizon") or 0)
            confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
            confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))
            gt_drifts, meta_horizon = load_synth_meta(synth_root, ds, seed)
            horizon = int(meta_horizon or horizon or 0)
            win_m = (
                compute_detection_metrics(gt_drifts, confirmed_raw, int(horizon))
                if horizon > 0
                else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
            )
            delays, miss_flags = per_drift_delays_and_miss(gt_drifts, confirmed_merged, horizon=int(horizon), tol=int(args.tol500))
            miss_tol500 = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None
            conf_p90 = percentile([None if d is None else float(d) for d in delays], 0.90)
            conf_p99 = percentile([None if d is None else float(d) for d in delays], 0.99)
            med_gap, p10_gap = gap_stats(confirmed_raw)
            cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
            rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None

            records.append(
                {
                    "track": "S",
                    "dataset": ds,
                    "unit": "sample_idx",
                    "seed": seed,
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "group": group,
                    "monitor_preset": str(args.monitor_preset),
                    "trigger_mode": "two_stage",
                    "confirm_theta": float(args.confirm_theta),
                    "confirm_window": int(args.confirm_window),
                    "confirm_cooldown_base": int(base_cd),
                    "adaptive_cooldown": int(bool(adaptive)),
                    "adaptive_window": int(args.adaptive_window) if adaptive else 0,
                    "adaptive_lower_per10k": float(args.adaptive_lower_per10k) if adaptive else 0.0,
                    "adaptive_upper_per10k": float(args.adaptive_upper_per10k) if adaptive else 0.0,
                    "adaptive_cooldown_low": int(args.adaptive_cooldown_low) if adaptive else 0,
                    "adaptive_cooldown_high": int(args.adaptive_cooldown_high) if adaptive else 0,
                    "weights": args.weights,
                    "log_path": str(log_path),
                    "horizon": int(horizon),
                    "acc_final": summ.get("acc_final"),
                    "acc_min_raw": summ.get("acc_min"),
                    "acc_min_warmup": acc_min_after_warmup(summ, int(args.warmup_samples)),
                    "confirmed_count_total": cc,
                    "confirm_rate_per_10k": rate,
                    "median_gap_between_confirms": med_gap,
                    "p10_gap_between_confirms": p10_gap,
                    "miss_tol500": miss_tol500,
                    "conf_P90": conf_p90,
                    "conf_P99": conf_p99,
                    "MTFA_win": win_m.get("MTFA"),
                }
            )

        # INSECTS
        for seed in list(args.insects_seeds):
            cfg_map = build_cfg_map(seed)
            ds = str(args.insects_dataset)
            if ds.lower() not in cfg_map:
                raise ValueError(f"未知 dataset: {ds}")
            base_cfg = cfg_map[ds.lower()]
            # 这里保持“检测配置一致”，但为了评估负迁移恢复，统一使用 severity 版本模型（不启用 gate/v2）
            model_variant = str(args.severity_model_variant)
            weights = dict(weights_base)
            weights["confirm_cooldown"] = float(base_cd)
            if adaptive:
                weights["adaptive_cooldown"] = 1.0
                weights["adaptive_window"] = float(args.adaptive_window)
                weights["adaptive_lower_per10k"] = float(args.adaptive_lower_per10k)
                weights["adaptive_upper_per10k"] = float(args.adaptive_upper_per10k)
                weights["adaptive_cooldown_low"] = float(args.adaptive_cooldown_low)
                weights["adaptive_cooldown_high"] = float(args.adaptive_cooldown_high)

            log_path = ensure_log(
                exp_run,
                ds,
                seed,
                base_cfg,
                model_variant=model_variant,
                monitor_preset=str(args.monitor_preset),
                trigger_threshold=float(args.confirm_theta),
                trigger_weights=weights,
                confirm_window=int(args.confirm_window),
                use_severity_v2=False,
                severity_gate="none",
                severity_gate_min_streak=1,
                device=str(args.device),
            )
            summ = read_run_summary(log_path)
            series_raw = summ.get("acc_series") or []
            series: List[Tuple[int, float]] = []
            for item in series_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                series.append((int(item[0]), float(item[1])))
            rec = compute_recovery_metrics(
                series,
                insects_positions,
                W=int(args.recovery_W),
                pre_window=int(args.pre_window),
                roll_points=int(args.roll_points),
            )
            horizon = int(summ.get("horizon") or 0)
            confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
            med_gap, p10_gap = gap_stats(confirmed_raw)
            cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
            rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None

            records.append(
                {
                    "track": "S",
                    "dataset": ds,
                    "unit": "sample_idx",
                    "seed": seed,
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "group": group,
                    "monitor_preset": str(args.monitor_preset),
                    "trigger_mode": "two_stage",
                    "confirm_theta": float(args.confirm_theta),
                    "confirm_window": int(args.confirm_window),
                    "confirm_cooldown_base": int(base_cd),
                    "adaptive_cooldown": int(bool(adaptive)),
                    "adaptive_window": int(args.adaptive_window) if adaptive else 0,
                    "adaptive_lower_per10k": float(args.adaptive_lower_per10k) if adaptive else 0.0,
                    "adaptive_upper_per10k": float(args.adaptive_upper_per10k) if adaptive else 0.0,
                    "adaptive_cooldown_low": int(args.adaptive_cooldown_low) if adaptive else 0,
                    "adaptive_cooldown_high": int(args.adaptive_cooldown_high) if adaptive else 0,
                    "weights": args.weights,
                    "log_path": str(log_path),
                    "horizon": int(horizon),
                    "model_variant": model_variant,
                    "acc_final": summ.get("acc_final"),
                    "acc_min_raw": summ.get("acc_min"),
                    "acc_min_warmup": acc_min_after_warmup(summ, int(args.warmup_samples)),
                    "confirmed_count_total": cc,
                    "confirm_rate_per_10k": rate,
                    "median_gap_between_confirms": med_gap,
                    "p10_gap_between_confirms": p10_gap,
                    "recovery_W": int(args.recovery_W),
                    "post_mean@W1000": rec.get("post_mean_acc"),
                    "post_min@W1000": rec.get("post_min_acc"),
                    "recovery_time_to_pre90": rec.get("recovery_time_to_pre90"),
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

