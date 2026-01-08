#!/usr/bin/env python
"""
NEXT_STAGE V6 - Track P: Detector×Gating 联动（sea_abrupt4 + INSECTS_abrupt_balanced）

目的：验证“检测更敏感（confirm 更频繁）时，gating/cooldown 能抑制负迁移并抬高谷底”。

配置：
- 基线：two_stage + tuned PH（theta/window 来自 Track O 推荐；cooldown=0）
- 对照组：
  A) + severity v2（no gate）
  B) + v2_gate_m{1,3,5}
  C) + confirm_cooldown（cooldown 来自 Track O 推荐）
  D) + gate + cooldown（组合：v2_gate_m{1,3,5} + cooldown）

输出：scripts/TRACKP_GATE_COOLDOWN.csv（按 run 粒度记录）
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
    p = argparse.ArgumentParser(description="Track P: gate × cooldown interaction")
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
    p.add_argument("--tracko_csv", type=str, default="scripts/TRACKO_CONFIRM_SWEEP.csv")
    p.add_argument("--fallback_confirm_theta", type=float, default=0.3)
    p.add_argument("--fallback_confirm_window", type=int, default=1)
    p.add_argument("--fallback_confirm_cooldown", type=int, default=500)
    p.add_argument("--gate_streaks", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--synthetic_root", type=str, default="data/synthetic")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKP_GATE_COOLDOWN.csv")
    # metrics
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


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def per_drift_delays(
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


def pick_tracko_default(tracko_csv: Path, acc_tol: float = 0.01) -> Tuple[float, int, int, str]:
    """
    选择规则（与 V6 报告一致）：
    - acc_final_mean ≥ best - acc_tol
    - 先最小化 miss_tol500_mean（或 conf_P90_mean）
    - 再最大化 MTFA_win_mean
    - 再最大化 acc_min_mean
    """
    if not tracko_csv.exists():
        return 0.0, 1, 0, "fallback：tracko_csv 不存在"
    rows = list(csv.DictReader(tracko_csv.open("r", encoding="utf-8")))
    if not rows:
        return 0.0, 1, 0, "fallback：tracko_csv 为空"
    filtered = [r for r in rows if (r.get("dataset") == "sea_abrupt4" and r.get("trigger_mode") == "two_stage")]
    if not filtered:
        return 0.0, 1, 0, "fallback：tracko_csv 无 sea_abrupt4 two_stage 记录"

    by_cfg: Dict[str, List[Dict[str, str]]] = {}
    for r in filtered:
        tag = str(r.get("config_tag") or "")
        if not tag:
            continue
        by_cfg.setdefault(tag, []).append(r)

    agg: List[Dict[str, Any]] = []
    for tag, rs in by_cfg.items():
        seen_runs: set[Tuple[str, str]] = set()
        acc_final_list: List[Optional[float]] = []
        acc_min_list: List[Optional[float]] = []
        mtfa_win_list: List[Optional[float]] = []
        for r in rs:
            run_id = str(r.get("run_id") or "")
            seed = str(r.get("seed") or "")
            key = (run_id, seed)
            if key in seen_runs:
                continue
            seen_runs.add(key)
            acc_final_list.append(_safe_float(r.get("acc_final")))
            acc_min_list.append(_safe_float(r.get("acc_min")))
            mtfa_win_list.append(_safe_float(r.get("MTFA_win")))
        miss_list = [_safe_float(r.get("miss_tol500")) for r in rs]
        miss_vals = [float(x) for x in miss_list if x is not None]
        miss_mean = float(statistics.mean(miss_vals)) if miss_vals else 1.0
        delays = [_safe_float(r.get("delay_confirmed")) for r in rs]
        conf_p90 = percentile(delays, 0.90)
        sample = rs[0]
        acc_final_mu, _ = mean_std(acc_final_list)
        acc_min_mu, _ = mean_std(acc_min_list)
        mtfa_mu, _ = mean_std(mtfa_win_list)
        agg.append(
            {
                "config_tag": tag,
                "confirm_theta": float(sample.get("confirm_theta") or 0.3),
                "confirm_window": int(float(sample.get("confirm_window") or 1)),
                "confirm_cooldown": int(float(sample.get("confirm_cooldown") or 0)),
                "acc_final_mean": float(acc_final_mu) if acc_final_mu is not None else float("-inf"),
                "acc_min_mean": float(acc_min_mu) if acc_min_mu is not None else float("-inf"),
                "miss_mean": float(miss_mean),
                "conf_p90": float(conf_p90) if conf_p90 is not None else float("inf"),
                "mtfa_mean": float(mtfa_mu) if mtfa_mu is not None else float("-inf"),
            }
        )
    best_acc = max(float(r["acc_final_mean"]) for r in agg)
    eligible = [r for r in agg if float(r["acc_final_mean"]) >= best_acc - float(acc_tol)]
    if not eligible:
        eligible = agg
    eligible.sort(
        key=lambda r: (
            float(r["miss_mean"]),
            float(r["conf_p90"]),
            -float(r["mtfa_mean"]),
            -float(r["acc_min_mean"]),
            str(r["config_tag"]),
        )
    )
    best = eligible[0]
    reason = (
        f"规则：acc_final_mean≥best-{acc_tol}，先最小化 miss_tol500_mean（再看 conf_P90），"
        f"再最大化 MTFA_win_mean，再最大化 acc_min_mean；best_acc={best_acc:.4f}"
    )
    return float(best["confirm_theta"]), int(best["confirm_window"]), int(best["confirm_cooldown"]), reason


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)
    weights_base = parse_weights(args.weights)

    # 从 Track O 自动选择默认（theta/window/cooldown），若缺失则回退。
    theta_o, window_o, cooldown_o, picked_reason = pick_tracko_default(Path(args.tracko_csv))
    theta = float(theta_o or args.fallback_confirm_theta)
    window = int(window_o or args.fallback_confirm_window)
    cooldown = int(cooldown_o if cooldown_o is not None else args.fallback_confirm_cooldown)
    if theta_o == 0.0 and window_o == 1 and cooldown_o == 0 and "fallback" in picked_reason:
        picked_note = f"{picked_reason}；fallback(theta={theta},window={window},cooldown={cooldown})"
    else:
        picked_note = f"picked TrackO: {picked_reason}"

    # 数据准备
    generate_default_abrupt_synth_datasets(seeds=list(args.sea_seeds), out_root=str(synth_root))
    insects_positions = load_insects_positions(Path(args.insects_meta))

    groups: List[Dict[str, Any]] = []
    groups.append(
        {
            "group": "baseline",
            "model_variant": str(args.sea_model_variant),
            "use_severity_v2": False,
            "severity_gate": "none",
            "severity_gate_min_streak": 1,
            "cooldown": 0,
        }
    )
    groups.append(
        {
            "group": "v2",
            "model_variant": str(args.severity_model_variant),
            "use_severity_v2": True,
            "severity_gate": "none",
            "severity_gate_min_streak": 1,
            "cooldown": 0,
        }
    )
    for m in [int(x) for x in args.gate_streaks]:
        groups.append(
            {
                "group": f"v2_gate_m{m}",
                "model_variant": str(args.severity_model_variant),
                "use_severity_v2": True,
                "severity_gate": "confirmed_streak",
                "severity_gate_min_streak": int(m),
                "cooldown": 0,
            }
        )
    groups.append(
        {
            "group": "cooldown",
            "model_variant": str(args.sea_model_variant),
            "use_severity_v2": False,
            "severity_gate": "none",
            "severity_gate_min_streak": 1,
            "cooldown": int(cooldown),
        }
    )
    for m in [int(x) for x in args.gate_streaks]:
        groups.append(
            {
                "group": f"v2_gate_m{m}_cd",
                "model_variant": str(args.severity_model_variant),
                "use_severity_v2": True,
                "severity_gate": "confirmed_streak",
                "severity_gate_min_streak": int(m),
                "cooldown": int(cooldown),
            }
        )

    records: List[Dict[str, Any]] = []
    for g in groups:
        group = str(g["group"])
        model_variant = str(g["model_variant"])
        use_v2 = bool(g["use_severity_v2"])
        gate = str(g["severity_gate"])
        m = int(g["severity_gate_min_streak"])
        cd = int(g["cooldown"])
        weights = dict(weights_base)
        weights["confirm_cooldown"] = float(cd)

        exp_run = create_experiment_run(
            experiment_name="trackP_gate_cooldown",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{group}_theta{theta:.2f}_w{window}_cd{cd}",
        )

        # sea_abrupt4
        for seed in list(args.sea_seeds):
            cfg_map = build_cfg_map(seed)
            ds = str(args.sea_dataset)
            if ds.lower() not in cfg_map:
                raise ValueError(f"未知 dataset: {ds}")
            base_cfg = cfg_map[ds.lower()]
            log_path = ensure_log(
                exp_run,
                ds,
                seed,
                base_cfg,
                model_variant=model_variant,
                monitor_preset=str(args.monitor_preset),
                trigger_threshold=float(theta),
                trigger_weights=weights,
                confirm_window=int(window),
                use_severity_v2=use_v2,
                severity_gate=gate,
                severity_gate_min_streak=m,
                device=str(args.device),
            )
            gt_drifts, meta_horizon = load_synth_meta(synth_root, ds, seed)
            summ = read_run_summary(log_path)
            horizon = int(meta_horizon or summ.get("horizon") or 0)
            confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
            confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))
            win_m = (
                compute_detection_metrics(gt_drifts, confirmed_raw, int(horizon))
                if horizon > 0
                else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
            )
            tol_m = compute_metrics_tolerance(gt_drifts, confirmed_merged, int(horizon), int(args.tol500))
            delays, miss_flags = per_drift_delays(gt_drifts, confirmed_merged, horizon=int(horizon), tol=int(args.tol500))
            records.append(
                {
                    "track": "P",
                    "dataset": ds,
                    "unit": "sample_idx",
                    "seed": seed,
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "group": group,
                    "model_variant": model_variant,
                    "monitor_preset": str(args.monitor_preset),
                    "monitor_preset_base": summ.get("monitor_preset_base", ""),
                    "trigger_mode": "two_stage",
                    "confirm_theta": float(theta),
                    "confirm_window": int(window),
                    "confirm_cooldown": int(cd),
                    "weights": args.weights,
                    "use_severity_v2": int(bool(use_v2)),
                    "severity_gate": gate,
                    "severity_gate_min_streak": int(m),
                    "log_path": str(log_path),
                    "acc_final": summ.get("acc_final"),
                    "acc_min": summ.get("acc_min"),
                    "confirmed_count_total": summ.get("confirmed_count_total"),
                    "miss_tol500": float(sum(miss_flags) / len(miss_flags)) if miss_flags else None,
                    "MDR_tol500": tol_m.get("MDR"),
                    "conf_P90": percentile([None if d is None else float(d) for d in delays], 0.90),
                    "conf_P99": percentile([None if d is None else float(d) for d in delays], 0.99),
                    "MTFA_win": win_m.get("MTFA"),
                    "note": picked_note,
                }
            )

        # INSECTS_abrupt_balanced
        for seed in list(args.insects_seeds):
            cfg_map = build_cfg_map(seed)
            ds = str(args.insects_dataset)
            if ds.lower() not in cfg_map:
                raise ValueError(f"未知 dataset: {ds}")
            base_cfg = cfg_map[ds.lower()]
            log_path = ensure_log(
                exp_run,
                ds,
                seed,
                base_cfg,
                model_variant=model_variant,
                monitor_preset=str(args.monitor_preset),
                trigger_threshold=float(theta),
                trigger_weights=weights,
                confirm_window=int(window),
                use_severity_v2=use_v2,
                severity_gate=gate,
                severity_gate_min_streak=m,
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
            records.append(
                {
                    "track": "P",
                    "dataset": ds,
                    "unit": "sample_idx",
                    "seed": seed,
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "group": group,
                    "model_variant": model_variant,
                    "monitor_preset": str(args.monitor_preset),
                    "monitor_preset_base": summ.get("monitor_preset_base", ""),
                    "trigger_mode": "two_stage",
                    "confirm_theta": float(theta),
                    "confirm_window": int(window),
                    "confirm_cooldown": int(cd),
                    "weights": args.weights,
                    "use_severity_v2": int(bool(use_v2)),
                    "severity_gate": gate,
                    "severity_gate_min_streak": int(m),
                    "log_path": str(log_path),
                    "acc_final": summ.get("acc_final"),
                    "acc_min": summ.get("acc_min"),
                    "confirmed_count_total": summ.get("confirmed_count_total"),
                    "recovery_W": int(args.recovery_W),
                    "post_mean@W1000": rec.get("post_mean_acc"),
                    "post_min@W1000": rec.get("post_min_acc"),
                    "recovery_time_to_pre90": rec.get("recovery_time_to_pre90"),
                    "note": picked_note,
                }
            )

    if not records:
        print("[warn] no records")
        return 0
    fieldnames: List[str] = []
    seen_keys: set[str] = set()
    for r in records:
        for k in r.keys():
            if k in seen_keys:
                continue
            seen_keys.add(k)
            fieldnames.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[done] wrote {out_csv} rows={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
