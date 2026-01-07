#!/usr/bin/env python
"""
Track L：severity v2 + gating 强化验证（INSECTS）。

- 对比组（至少）：baseline / v2 / v2_gate
- 本实现提供 gating 变体：confirmed_streak(min_streak=m) 做 sweep（m ∈ {1,3,5}）

输出：scripts/TRACKL_GATING_SWEEP.csv
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
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track L：severity v2 gating sweep on INSECTS")
    p.add_argument("--dataset", type=str, default="INSECTS_abrupt_balanced")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    p.add_argument("--trigger_mode", type=str, default="or", choices=["or", "weighted", "two_stage"])
    p.add_argument("--trigger_threshold", type=float, default=0.5, help="用于 gating 的票分阈值（也会作为 trigger_threshold 透传）")
    p.add_argument("--confirm_window", type=int, default=200)
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--gate_streaks", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKL_GATING_SWEEP.csv")
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
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


def _safe_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _safe_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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


def compute_metrics_tolerance(
    gt_drifts: Sequence[int],
    detections: Sequence[int],
    horizon: int,
    tolerance: int,
) -> Dict[str, Optional[float]]:
    gt = sorted(int(d) for d in gt_drifts)
    dets = sorted(int(d) for d in detections)
    if not gt or horizon <= 0:
        return {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
    matched: set[int] = set()
    delays: List[int] = []
    missed = 0
    for drift in gt:
        match = None
        for det in dets:
            if det < drift:
                continue
            if det <= drift + tolerance and det not in matched:
                match = det
                break
            if det > drift + tolerance:
                break
        if match is None:
            missed += 1
        else:
            matched.add(match)
            delays.append(int(match - drift))
    false_alarms = [d for d in dets if d not in matched]
    mdr = missed / len(gt) if gt else None
    mtd = (sum(delays) / len(delays)) if delays else None
    if len(false_alarms) >= 2:
        gaps = [b - a for a, b in zip(false_alarms[:-1], false_alarms[1:])]
        mtfa = sum(gaps) / len(gaps)
    elif len(false_alarms) == 1:
        mtfa = float(horizon - false_alarms[0]) if horizon > 0 else None
    else:
        mtfa = None
    if mdr is None or mtd is None or mtfa is None or mdr >= 1.0 or mtd <= 0:
        mtr = None
    else:
        mtr = mtfa / (mtd * (1 - mdr))
    return {
        "MDR": float(mdr) if mdr is not None else None,
        "MTD": float(mtd) if mtd is not None else None,
        "MTFA": float(mtfa) if mtfa is not None else None,
        "MTR": float(mtr) if mtr is not None else None,
    }


def load_insects_positions(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    positions = obj.get("positions") or []
    return [int(x) for x in positions]


def read_log(csv_path: Path) -> Dict[str, object]:
    summary_path = csv_path.with_suffix(".summary.json")
    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            series_raw = obj.get("acc_series") or []
            series: List[Tuple[int, float]] = []
            for item in series_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                series.append((int(item[0]), float(item[1])))
            return {
                "acc_final": obj.get("acc_final"),
                "mean_acc": obj.get("mean_acc"),
                "acc_min": obj.get("acc_min"),
                "confirmed": [int(x) for x in (obj.get("confirmed_sample_idxs") or [])],
                "horizon": int(obj.get("horizon") or 0),
                "acc_series": series,
            }
        except Exception:
            pass
    accs: List[float] = []
    xs: List[int] = []
    confirmed: List[int] = []
    series: List[Tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x = _safe_int(r.get("sample_idx")) or _safe_int(r.get("seen_samples")) or _safe_int(r.get("step"))
            if x is None:
                continue
            xs.append(int(x))
            acc = _safe_float(r.get("metric_accuracy"))
            if acc is not None and not math.isnan(acc):
                accs.append(float(acc))
                series.append((int(x), float(acc)))
            if r.get("drift_flag") == "1":
                confirmed.append(int(x))
    horizon = int(xs[-1] + 1) if xs else 0
    return {
        "acc_final": accs[-1] if accs else None,
        "mean_acc": (sum(accs) / len(accs)) if accs else None,
        "acc_min": min(accs) if accs else None,
        "confirmed": confirmed,
        "horizon": horizon,
        "acc_series": series,
    }


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


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
    post_mean_mu, post_mean_sd = mean_std(post_means)
    post_min_mu, post_min_sd = mean_std(post_mins)
    rec_mu, rec_sd = mean_std(rec_times)
    return {
        "post_mean_acc": post_mean_mu,
        "post_mean_acc_std": post_mean_sd,
        "post_min_acc": post_min_mu,
        "post_min_acc_std": post_min_sd,
        "recovery_time_to_pre90": rec_mu,
        "recovery_time_to_pre90_std": rec_sd,
    }


def find_insects_cfg(seed: int) -> ExperimentConfig:
    for cfg in _default_experiment_configs(seed):
        if cfg.dataset_name == "INSECTS_abrupt_balanced":
            return cfg
    raise ValueError("未在 _default_experiment_configs 中找到 INSECTS_abrupt_balanced")


def ensure_log(
    exp_run: ExperimentRun,
    dataset: str,
    seed: int,
    base_cfg: ExperimentConfig,
    *,
    model_variant: str,
    monitor_preset: str,
    trigger_mode: str,
    trigger_threshold: float,
    trigger_weights: Dict[str, float],
    confirm_window: int,
    use_severity_v2: bool,
    severity_gate: str,
    severity_gate_min_streak: int,
    entropy_mode: str,
    severity_decay: float,
    freeze_baseline_steps: int,
    severity_scheduler_scale: float,
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
        trigger_mode=trigger_mode,
        trigger_weights=trigger_weights,
        trigger_threshold=float(trigger_threshold),
        confirm_window=int(confirm_window),
        use_severity_v2=bool(use_severity_v2),
        severity_gate=str(severity_gate),
        severity_gate_min_streak=int(severity_gate_min_streak),
        entropy_mode=str(entropy_mode),
        severity_decay=float(severity_decay),
        freeze_baseline_steps=int(freeze_baseline_steps),
        severity_scheduler_scale=float(severity_scheduler_scale),
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    weights = parse_weights(args.weights)
    insects_positions = load_insects_positions(Path(args.insects_meta))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)

    groups: List[Tuple[str, Dict[str, object]]] = []
    groups.append(("baseline", {"model_variant": "ts_drift_adapt", "use_v2": False, "gate": "none", "min_streak": 1}))
    groups.append(("v2", {"model_variant": "ts_drift_adapt_severity", "use_v2": True, "gate": "none", "min_streak": 1}))
    for m in [int(x) for x in args.gate_streaks]:
        groups.append(
            (
                f"v2_gate_m{m}",
                {"model_variant": "ts_drift_adapt_severity", "use_v2": True, "gate": "confirmed_streak", "min_streak": m},
            )
        )

    records: List[Dict[str, object]] = []
    for tag, g in groups:
        exp_run = create_experiment_run(
            experiment_name="trackL_gating_sweep",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{tag}",
        )
        for seed in list(args.seeds):
            base_cfg = find_insects_cfg(seed)
            log_path = ensure_log(
                exp_run,
                args.dataset,
                seed,
                base_cfg,
                model_variant=str(g["model_variant"]),
                monitor_preset=args.monitor_preset,
                trigger_mode=str(args.trigger_mode),
                trigger_threshold=float(args.trigger_threshold),
                trigger_weights=weights,
                confirm_window=int(args.confirm_window),
                use_severity_v2=bool(g["use_v2"]),
                severity_gate=str(g["gate"]),
                severity_gate_min_streak=int(g["min_streak"]),
                entropy_mode="uncertain",
                severity_decay=0.95,
                freeze_baseline_steps=5,
                severity_scheduler_scale=2.0,
                device=args.device,
            )
            stats = read_log(log_path)
            horizon = int(stats["horizon"])
            confirmed_raw: List[int] = list(stats["confirmed"])  # type: ignore[assignment]
            confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))
            win_m = compute_detection_metrics(insects_positions, confirmed_raw, horizon) if horizon > 0 else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
            tol_m = compute_metrics_tolerance(insects_positions, confirmed_merged, horizon, int(args.tol500))
            rec = compute_recovery_metrics(
                stats["acc_series"],  # type: ignore[arg-type]
                insects_positions,
                W=int(args.recovery_W),
                pre_window=int(args.pre_window),
                roll_points=int(args.roll_points),
            )
            records.append(
                {
                    "track": "L",
                    "unit": "sample_idx",
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "dataset": args.dataset,
                    "seed": seed,
                    "group": tag,
                    "model_variant": str(g["model_variant"]),
                    "monitor_preset": args.monitor_preset,
                    "trigger_mode": str(args.trigger_mode),
                    "trigger_threshold": float(args.trigger_threshold),
                    "confirm_window": int(args.confirm_window),
                    "weights": args.weights,
                    "use_severity_v2": int(bool(g["use_v2"])),
                    "severity_gate": str(g["gate"]),
                    "severity_gate_min_streak": int(g["min_streak"]),
                    "entropy_mode": "uncertain",
                    "severity_decay": 0.95,
                    "freeze_baseline_steps": 5,
                    "severity_scheduler_scale": 2.0,
                    "log_path": str(log_path),
                    "acc_final": stats["acc_final"],
                    "mean_acc": stats["mean_acc"],
                    "acc_min": stats["acc_min"],
                    "drift_flag_count": len(confirmed_raw),
                    "drift_events_merged": len(confirmed_merged),
                    "MDR_win": win_m.get("MDR"),
                    "MTD_win": win_m.get("MTD"),
                    "MTFA_win": win_m.get("MTFA"),
                    "MTR_win": win_m.get("MTR"),
                    "MDR_tol": tol_m.get("MDR"),
                    "MTD_tol": tol_m.get("MTD"),
                    "MTFA_tol": tol_m.get("MTFA"),
                    "MTR_tol": tol_m.get("MTR"),
                    "recovery_W": int(args.recovery_W),
                    "post_mean_acc": rec.get("post_mean_acc"),
                    "post_mean_acc_std": rec.get("post_mean_acc_std"),
                    "post_min_acc": rec.get("post_min_acc"),
                    "post_min_acc_std": rec.get("post_min_acc_std"),
                    "recovery_time_to_pre90": rec.get("recovery_time_to_pre90"),
                    "recovery_time_to_pre90_std": rec.get("recovery_time_to_pre90_std"),
                }
            )

    if not records:
        print("[warn] no records")
        return 0
    fieldnames = list(records[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[done] wrote {out_csv} rows={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
