#!/usr/bin/env python
"""
Track J：two_stage 超参小扫（confirm_theta × confirm_window），并输出统一指标 + delay 统计。

输出：scripts/TRACKJ_TWOSTAGE_SWEEP.csv
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

from data.streams import generate_default_abrupt_synth_datasets
from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track J：two_stage sweep (theta × window)")
    p.add_argument("--datasets", type=str, default="sea_abrupt4,stagger_abrupt3")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_thetas", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    p.add_argument("--confirm_windows", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--synthetic_root", type=str, default="data/synthetic")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKJ_TWOSTAGE_SWEEP.csv")
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
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


def percentile(values: Sequence[int], q: float) -> Optional[float]:
    vals = sorted(int(x) for x in values if x is not None)  # type: ignore[comparison-overlap]
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


def read_log_series(csv_path: Path) -> Dict[str, object]:
    summary_path = csv_path.with_suffix(".summary.json")
    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            return {
                "acc_final": obj.get("acc_final"),
                "mean_acc": obj.get("mean_acc"),
                "acc_min": obj.get("acc_min"),
                "candidates": [int(x) for x in (obj.get("candidate_sample_idxs") or [])],
                "confirmed": [int(x) for x in (obj.get("confirmed_sample_idxs") or [])],
                "horizon": int(obj.get("horizon") or 0),
            }
        except Exception:
            pass
    accs: List[float] = []
    xs: List[int] = []
    candidates: List[int] = []
    confirmed: List[int] = []
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
            vote_count = _safe_int(r.get("monitor_vote_count"))
            if (vote_count is not None and vote_count >= 1) or (r.get("candidate_flag") == "1"):
                candidates.append(int(x))
            if r.get("drift_flag") == "1":
                confirmed.append(int(x))
    horizon = int(xs[-1] + 1) if xs else 0
    return {
        "acc_final": accs[-1] if accs else None,
        "mean_acc": (sum(accs) / len(accs)) if accs else None,
        "acc_min": min(accs) if accs else None,
        "candidates": candidates,
        "confirmed": confirmed,
        "horizon": horizon,
    }


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def per_drift_confirm_delays(
    gt_drifts: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Dict[str, object]:
    gt = sorted(int(d) for d in gt_drifts)
    confs = sorted(int(x) for x in confirmed)
    delays: List[int] = []
    miss_flags: List[int] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_conf = first_event_in_range(confs, g, end)
        if first_conf is None:
            delays.append(int(end - g))
        else:
            delays.append(int(first_conf - g))
        tol_end = min(end, g + int(tol) + 1)
        miss_flags.append(0 if first_event_in_range(confs, g, tol_end) is not None else 1)
    p50 = percentile(delays, 0.50)
    p90 = percentile(delays, 0.90)
    p99 = percentile(delays, 0.99)
    miss_rate = (sum(miss_flags) / len(miss_flags)) if miss_flags else None
    return {
        "delay_p50": p50,
        "delay_p90": p90,
        "delay_p99": p99,
        "miss_rate_tol500": miss_rate,
        "n_drifts": len(gt),
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
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = list(args.seeds)
    weights = parse_weights(args.weights)
    thetas = [float(x) for x in args.confirm_thetas]
    windows = [int(x) for x in args.confirm_windows]
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    generate_default_abrupt_synth_datasets(seeds=seeds, out_root=str(synth_root))

    records: List[Dict[str, object]] = []
    for theta in thetas:
        for window in windows:
            exp_run = create_experiment_run(
                experiment_name="trackJ_twostage_sweep",
                results_root=results_root,
                logs_root=logs_root,
                run_name=f"theta{theta:.2f}_w{window}",
            )
            for dataset in datasets:
                for seed in seeds:
                    cfg_map = build_cfg_map(seed)
                    if dataset.lower() not in cfg_map:
                        raise ValueError(f"未知 dataset: {dataset}")
                    base_cfg = cfg_map[dataset.lower()]
                    log_path = ensure_log(
                        exp_run,
                        dataset,
                        seed,
                        base_cfg,
                        model_variant="ts_drift_adapt",
                        monitor_preset=args.monitor_preset,
                        trigger_threshold=theta,
                        trigger_weights=weights,
                        confirm_window=window,
                        device=args.device,
                    )
                    gt_drifts, meta_horizon = load_synth_meta(synth_root, dataset, seed)
                    stats = read_log_series(log_path)
                    horizon = int(meta_horizon or stats["horizon"] or 0)
                    confirmed_raw: List[int] = list(stats["confirmed"])  # type: ignore[assignment]
                    confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))
                    win_m = compute_detection_metrics(gt_drifts, confirmed_raw, int(horizon)) if horizon > 0 else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
                    tol_m = compute_metrics_tolerance(gt_drifts, confirmed_merged, int(horizon), int(args.tol500))
                    delay = per_drift_confirm_delays(gt_drifts, confirmed_raw, horizon=int(horizon), tol=int(args.tol500))
                    records.append(
                        {
                            "track": "J",
                            "unit": "sample_idx",
                            "experiment_name": exp_run.experiment_name,
                            "run_id": exp_run.run_id,
                            "dataset": dataset,
                            "seed": seed,
                            "model_variant": "ts_drift_adapt",
                            "monitor_preset": args.monitor_preset,
                            "trigger_mode": "two_stage",
                            "confirm_theta": float(theta),
                            "confirm_window": int(window),
                            "weights": args.weights,
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
                            "delay_p50": delay["delay_p50"],
                            "delay_p90": delay["delay_p90"],
                            "delay_p99": delay["delay_p99"],
                            "miss_rate_tol500": delay["miss_rate_tol500"],
                            "n_drifts": delay["n_drifts"],
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
