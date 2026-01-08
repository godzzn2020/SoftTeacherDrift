#!/usr/bin/env python
"""
NEXT_STAGE V8 - Track U：合成流泛化（检测创新点定稿）

datasets = [sea_abrupt4, sine_abrupt4, stagger_abrupt3]
seeds = 1..5

对比组：
A) OR + default PH
B) OR + tuned PH（error.threshold=0.05,error.min_instances=5）
C) weighted + tuned PH（trigger_threshold=0.50）
D) two_stage + tuned PH + fixed cooldown=200（candidate=OR, confirm=weighted）

输出（run 粒度）：scripts/TRACKU_SYNTH_GENERALIZATION.csv
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
    p = argparse.ArgumentParser(description="Track U: synthetic generalization")
    p.add_argument("--datasets", type=str, default="sea_abrupt4,sine_abrupt4,stagger_abrupt3")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--model_variant", type=str, default="ts_drift_adapt")
    p.add_argument("--monitor_preset_default", type=str, default="error_divergence_ph_meta")
    p.add_argument(
        "--monitor_preset_tuned",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--confirm_window", type=int, default=1)
    p.add_argument("--confirm_cooldown", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--synthetic_root", type=str, default="data/synthetic")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKU_SYNTH_GENERALIZATION.csv")
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--warmup_samples", type=int, default=2000)
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
    candidates: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Tuple[List[float], List[float], List[int]]:
    gt = sorted(int(d) for d in gt_drifts)
    cands = sorted(int(x) for x in candidates)
    confs = sorted(int(x) for x in confirmed)
    delays_cand: List[float] = []
    delays_conf: List[float] = []
    miss_flags: List[int] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_c = first_event_in_range(cands, g, end)
        first_f = first_event_in_range(confs, g, end)
        delays_cand.append(float(end - g) if first_c is None else float(first_c - g))
        delays_conf.append(float(end - g) if first_f is None else float(first_f - g))
        tol_end = min(end, g + int(tol) + 1)
        miss_flags.append(0 if first_event_in_range(confs, g, tol_end) is not None else 1)
    return delays_cand, delays_conf, miss_flags


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
    trigger_mode: str,
    trigger_threshold: float,
    trigger_weights: Dict[str, float],
    confirm_window: int,
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
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = list(args.seeds)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    weights_base = parse_weights(args.weights)

    generate_default_abrupt_synth_datasets(seeds=seeds, out_root=str(synth_root))

    theta = float(args.theta)
    window = int(args.confirm_window)
    cooldown = int(args.confirm_cooldown)

    groups: List[Dict[str, Any]] = [
        {
            "group": "A_or_defaultPH",
            "monitor_preset": str(args.monitor_preset_default),
            "trigger_mode": "or",
            "trigger_threshold": 0.5,
            "confirm_window": window,
            "cooldown": 0,
        },
        {
            "group": "B_or_tunedPH",
            "monitor_preset": str(args.monitor_preset_tuned),
            "trigger_mode": "or",
            "trigger_threshold": 0.5,
            "confirm_window": window,
            "cooldown": 0,
        },
        {
            "group": "C_weighted_tunedPH",
            "monitor_preset": str(args.monitor_preset_tuned),
            "trigger_mode": "weighted",
            "trigger_threshold": theta,
            "confirm_window": window,
            "cooldown": 0,
        },
        {
            "group": "D_two_stage_tunedPH_cd200",
            "monitor_preset": str(args.monitor_preset_tuned),
            "trigger_mode": "two_stage",
            "trigger_threshold": theta,
            "confirm_window": 1,
            "cooldown": cooldown,
        },
    ]

    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        for g in groups:
            group = str(g["group"])
            exp_run = create_experiment_run(
                experiment_name="trackU_synth_generalization",
                results_root=results_root,
                logs_root=logs_root,
                run_name=f"{dataset}_{group}_theta{theta:.2f}_w{int(g['confirm_window'])}_cd{int(g['cooldown'])}",
            )
            for seed in seeds:
                cfg_map = build_cfg_map(seed)
                if dataset.lower() not in cfg_map:
                    raise ValueError(f"未知 dataset: {dataset}")
                base_cfg = cfg_map[dataset.lower()]
                weights = dict(weights_base)
                if int(g["cooldown"]) > 0:
                    weights["confirm_cooldown"] = float(int(g["cooldown"]))
                log_path = ensure_log(
                    exp_run,
                    dataset,
                    seed,
                    base_cfg,
                    model_variant=str(args.model_variant),
                    monitor_preset=str(g["monitor_preset"]),
                    trigger_mode=str(g["trigger_mode"]),
                    trigger_threshold=float(g["trigger_threshold"]),
                    trigger_weights=weights,
                    confirm_window=int(g["confirm_window"]),
                    device=str(args.device),
                )
                summ = read_run_summary(log_path)
                gt_drifts, meta_horizon = load_synth_meta(synth_root, dataset, seed)
                horizon = int(meta_horizon or summ.get("horizon") or 0)
                candidates_raw = [int(x) for x in (summ.get("candidate_sample_idxs") or [])]
                confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                candidates_merged = merge_events(candidates_raw, int(args.min_separation))
                confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))

                win_m = (
                    compute_detection_metrics(gt_drifts, confirmed_raw, int(horizon))
                    if horizon > 0
                    else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
                )
                tol_m = compute_metrics_tolerance(gt_drifts, confirmed_merged, int(horizon), int(args.tol500))
                delays_cand, delays_conf, miss_flags = per_drift_first_delays(
                    gt_drifts,
                    candidates_merged,
                    confirmed_merged,
                    horizon=int(horizon),
                    tol=int(args.tol500),
                )

                cand_p50 = percentile(delays_cand, 0.50)
                cand_p90 = percentile(delays_cand, 0.90)
                cand_p99 = percentile(delays_cand, 0.99)
                conf_p50 = percentile(delays_conf, 0.50)
                conf_p90 = percentile(delays_conf, 0.90)
                conf_p99 = percentile(delays_conf, 0.99)
                miss_tol500 = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None

                cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
                confirm_rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None

                records.append(
                    {
                        "track": "U",
                        "dataset": dataset,
                        "unit": "sample_idx",
                        "seed": seed,
                        "experiment_name": exp_run.experiment_name,
                        "run_id": exp_run.run_id,
                        "group": group,
                        "model_variant": str(args.model_variant),
                        "monitor_preset": str(g["monitor_preset"]),
                        "trigger_mode": str(g["trigger_mode"]),
                        "confirm_theta": float(g["trigger_threshold"]),
                        "confirm_window": int(g["confirm_window"]),
                        "confirm_cooldown": int(g["cooldown"]),
                        "weights": args.weights,
                        "log_path": str(log_path),
                        "horizon": int(horizon),
                        "acc_final": summ.get("acc_final"),
                        "acc_min_raw": summ.get("acc_min"),
                        f"acc_min@{int(args.warmup_samples)}": acc_min_after_warmup(summ, int(args.warmup_samples)),
                        "miss_tol500": miss_tol500,
                        "MDR_tol500": tol_m.get("MDR"),
                        "cand_P50": cand_p50,
                        "cand_P90": cand_p90,
                        "cand_P99": cand_p99,
                        "conf_P50": conf_p50,
                        "conf_P90": conf_p90,
                        "conf_P99": conf_p99,
                        "MTFA_win": win_m.get("MTFA"),
                        "confirmed_count_total": cc,
                        "confirm_rate_per_10k": confirm_rate,
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

