#!/usr/bin/env python
"""
NEXT_STAGE V9 - Track Y（必做）：detector 敏感度 × gating 联动（机制验证）

dataset=INSECTS_abrupt_balanced
seeds=1..20

检测两档：
1) clean: error.threshold=0.05, min_instances=5
2) sensitive: error.threshold=0.03（可调），min_instances=5

恢复两档：
- v2
- v2_gate_mK（K 从 Track X 的赢家自动选择，或显式指定）

固定检测侧：
- trigger=two_stage(candidate=OR, confirm=weighted)
- confirm_theta=0.50, confirm_window=1, confirm_cooldown=200

输出：scripts/TRACKY_DETECTOR_GATING_INTERACTION.csv（逐 seed）
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
    p = argparse.ArgumentParser(description="Track Y: detector sensitivity × gating interaction (INSECTS)")
    p.add_argument("--dataset", type=str, default="INSECTS_abrupt_balanced")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    p.add_argument("--trackx_csv", type=str, default="scripts/TRACKX_INSECTS_GATING_SWEEP.csv")
    p.add_argument("--winner_m", type=int, default=0, help="若为 0，则从 TrackX 自动选择最佳 m（m1/m3/m5）")
    p.add_argument("--model_variant_severity", type=str, default="ts_drift_adapt_severity")
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_theta", type=float, default=0.5)
    p.add_argument("--confirm_window", type=int, default=1)
    p.add_argument("--confirm_cooldown", type=int, default=200)
    # detector
    p.add_argument("--clean_error_threshold", type=float, default=0.05)
    p.add_argument("--sensitive_error_threshold", type=float, default=0.03)
    p.add_argument("--error_min_instances", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKY_DETECTOR_GATING_INTERACTION.csv")
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--recovery_W", type=int, default=1000)
    p.add_argument("--pre_window", type=int, default=1000)
    p.add_argument("--roll_points", type=int, default=5)
    p.add_argument("--insects_meta", type=str, default="datasets/real/INSECTS_abrupt_balanced.json")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
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


def load_insects_positions(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    positions = obj.get("positions") or []
    return [int(x) for x in positions]


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


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(statistics.mean(vals))


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


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

    post_mean_mu, _ = mean_std(post_means)
    post_min_mu, _ = mean_std(post_mins)
    rec_mu, _ = mean_std(rec_times)
    return {
        "post_mean_acc": post_mean_mu,
        "post_min_acc": post_min_mu,
        "recovery_time_to_pre90": rec_mu,
    }


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
        use_severity_v2=True,
        severity_gate=str(severity_gate),
        severity_gate_min_streak=int(severity_gate_min_streak),
        entropy_mode="uncertain",
        severity_decay=0.95,
        freeze_baseline_steps=5,
        severity_scheduler_scale=2.0,
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def pick_winner_m(trackx_csv: Path, acc_tol: float) -> Tuple[int, str]:
    if not trackx_csv.exists():
        return 5, "fallback：trackx_csv 不存在（默认 m=5）"
    rows = list(csv.DictReader(trackx_csv.open("r", encoding="utf-8")))
    if not rows:
        return 5, "fallback：trackx_csv 为空（默认 m=5）"
    rows = [r for r in rows if r.get("dataset") == "INSECTS_abrupt_balanced"]
    gates = [r for r in rows if str(r.get("group") or "").startswith("v2_gate_m")]
    if not gates:
        return 5, "fallback：trackx_csv 无 v2_gate_m*（默认 m=5）"

    by_g: Dict[str, List[Dict[str, str]]] = {}
    for r in gates:
        by_g.setdefault(str(r.get("group") or ""), []).append(r)

    agg: List[Tuple[str, float, float]] = []  # (group, acc_final_mean, post_min_mean)
    for g, rs in by_g.items():
        acc_final = mean([_safe_float(x.get("acc_final")) for x in rs]) or float("-inf")
        post_min = mean([_safe_float(x.get("post_min@W1000")) for x in rs]) or float("-inf")
        agg.append((g, float(acc_final), float(post_min)))

    best_acc = max(x[1] for x in agg)
    eligible = [x for x in agg if x[1] >= best_acc - float(acc_tol)]
    if not eligible:
        eligible = agg
    eligible.sort(key=lambda x: (-x[2], x[0]))
    best_group = eligible[0][0]
    try:
        m = int(best_group.split("m")[-1].split("_")[0])
    except Exception:
        m = 5
    return m, f"picked from TrackX: best_group={best_group} (acc_tol={acc_tol})"


def make_monitor_preset(error_thr: float, error_min_instances: int) -> str:
    return f"error_divergence_ph_meta@error.threshold={float(error_thr)},error.min_instances={int(error_min_instances)}"


def main() -> int:
    args = parse_args()
    dataset = str(args.dataset)
    seeds = list(args.seeds)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    weights_base = parse_weights(args.weights)
    insects_positions = load_insects_positions(Path(args.insects_meta))

    m = int(args.winner_m or 0)
    picked_note = ""
    if m <= 0:
        m, picked_note = pick_winner_m(Path(args.trackx_csv), float(args.acc_tolerance))
    gate_group = f"v2_gate_m{m}"

    theta = float(args.confirm_theta)
    window = int(args.confirm_window)
    cooldown = int(args.confirm_cooldown)

    detectors = [
        ("clean", float(args.clean_error_threshold)),
        ("sensitive", float(args.sensitive_error_threshold)),
    ]
    recoveries = [
        ("v2", "none", 1),
        (gate_group, "confirmed_streak", int(m)),
    ]

    records: List[Dict[str, Any]] = []
    for det_tag, err_thr in detectors:
        monitor_preset = make_monitor_preset(err_thr, int(args.error_min_instances))
        for rec_tag, gate_mode, gate_m in recoveries:
            exp_run = create_experiment_run(
                experiment_name="trackY_detector_gating_interaction",
                results_root=results_root,
                logs_root=logs_root,
                run_name=f"{det_tag}_{rec_tag}_thr{err_thr:.3f}_m{gate_m}_theta{theta:.2f}_w{window}_cd{cooldown}",
            )
            for seed in seeds:
                cfg_map = build_cfg_map(seed)
                if dataset.lower() not in cfg_map:
                    raise ValueError(f"未知 dataset: {dataset}")
                base_cfg = cfg_map[dataset.lower()]
                weights = dict(weights_base)
                weights["confirm_cooldown"] = float(cooldown)
                log_path = ensure_log(
                    exp_run,
                    dataset,
                    seed,
                    base_cfg,
                    model_variant=str(args.model_variant_severity),
                    monitor_preset=monitor_preset,
                    trigger_threshold=theta,
                    trigger_weights=weights,
                    confirm_window=window,
                    severity_gate=str(gate_mode),
                    severity_gate_min_streak=int(gate_m),
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
                cc = int(summ.get("confirmed_count_total") or 0)
                rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                records.append(
                    {
                        "track": "Y",
                        "dataset": dataset,
                        "unit": "sample_idx",
                        "seed": seed,
                        "experiment_name": exp_run.experiment_name,
                        "run_id": exp_run.run_id,
                        "detector_mode": det_tag,
                        "error_threshold": float(err_thr),
                        "error_min_instances": int(args.error_min_instances),
                        "recovery_mode": rec_tag,
                        "use_severity_v2": 1,
                        "severity_gate": str(gate_mode),
                        "severity_gate_min_streak": int(gate_m),
                        "monitor_preset": monitor_preset,
                        "trigger_mode": "two_stage",
                        "confirm_theta": theta,
                        "confirm_window": window,
                        "confirm_cooldown": cooldown,
                        "weights": args.weights,
                        "log_path": str(log_path),
                        "horizon": horizon,
                        "acc_final": summ.get("acc_final"),
                        "acc_min_raw": summ.get("acc_min"),
                        f"acc_min@{int(args.warmup_samples)}": acc_min_after_warmup(summ, int(args.warmup_samples)),
                        "post_mean@W1000": rec.get("post_mean_acc"),
                        "post_min@W1000": rec.get("post_min_acc"),
                        "recovery_time_to_pre90": rec.get("recovery_time_to_pre90"),
                        "confirmed_count": cc,
                        "confirm_rate_per_10k": rate,
                        "picked_gate_m": int(m),
                        "picked_note": picked_note,
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

