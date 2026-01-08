#!/usr/bin/env python
"""
NEXT_STAGE V6 - Track O: Confirm-side sweep（sea_abrupt4）

固定：
- dataset=sea_abrupt4
- model=ts_drift_adapt
- monitor_preset：tuned PH（error.threshold=0.05,error.min_instances=5；divergence 默认）
- trigger_mode=two_stage（candidate=OR，confirm=weighted）

扫参：
- confirm_theta ∈ {0.3,0.4,0.5,0.6}
- confirm_window ∈ {1,2,3}
- confirm_cooldown ∈ {0,200,500,1000}（单位：sample_idx）

输出：scripts/TRACKO_CONFIRM_SWEEP.csv（按 drift 粒度记录，便于做 conf_P90/P99 与 miss_tol500 口径）

注意：本脚本只读取本轮新生成的 run_id 日志对应的 *.summary.json（或必要列），不做 logs/ 或 results/ 的全局搜索。
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
    p = argparse.ArgumentParser(description="Track O: confirm-side sweep (theta × window × cooldown)")
    p.add_argument("--dataset", type=str, default="sea_abrupt4")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--model_variant", type=str, default="ts_drift_adapt")
    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_thetas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6])
    p.add_argument("--confirm_windows", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--confirm_cooldowns", nargs="+", type=int, default=[0, 200, 500, 1000])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--synthetic_root", type=str, default="data/synthetic")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKO_CONFIRM_SWEEP.csv")
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


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


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


def per_drift_delay_rows(
    gt_drifts: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> List[Dict[str, Any]]:
    gt = sorted(int(d) for d in gt_drifts)
    confs = sorted(int(x) for x in confirmed)
    rows: List[Dict[str, Any]] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_conf = first_event_in_range(confs, g, end)
        tol_end = min(end, g + int(tol) + 1)
        miss_tol500 = 0 if first_event_in_range(confs, g, tol_end) is not None else 1
        rows.append(
            {
                "drift_idx": i,
                "drift_pos": g,
                "segment_end": end,
                "first_confirmed_step": first_conf,
                "delay_confirmed": None if first_conf is None else int(first_conf - g),
                "miss_tol500": int(miss_tol500),
            }
        )
    return rows


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


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def main() -> int:
    args = parse_args()
    dataset = str(args.dataset)
    seeds = list(args.seeds)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    weights_base = parse_weights(args.weights)

    generate_default_abrupt_synth_datasets(seeds=seeds, out_root=str(synth_root))

    records: List[Dict[str, Any]] = []
    for theta in [float(x) for x in args.confirm_thetas]:
        for window in [int(x) for x in args.confirm_windows]:
            for cooldown in [int(x) for x in args.confirm_cooldowns]:
                config_tag = f"theta{theta:.2f}_w{window}_cd{cooldown}"
                weights = dict(weights_base)
                # 冷却通过 trigger_weights 透传，训练环会解析该 key 并注入 DriftMonitor。
                weights["confirm_cooldown"] = float(cooldown)
                exp_run = create_experiment_run(
                    experiment_name="trackO_confirm_sweep",
                    results_root=results_root,
                    logs_root=logs_root,
                    run_name=config_tag,
                )
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
                        model_variant=str(args.model_variant),
                        monitor_preset=str(args.monitor_preset),
                        trigger_threshold=float(theta),
                        trigger_weights=weights,
                        confirm_window=int(window),
                        device=str(args.device),
                    )
                    gt_drifts, meta_horizon = load_synth_meta(synth_root, dataset, seed)
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
                    per_drift = per_drift_delay_rows(
                        gt_drifts,
                        confirmed_merged,
                        horizon=int(horizon),
                        tol=int(args.tol500),
                    )
                    ph_params = summ.get("monitor_ph_params") or {}
                    ph_error = ph_params.get("error_rate") if isinstance(ph_params, dict) else {}
                    ph_div = ph_params.get("divergence") if isinstance(ph_params, dict) else {}
                    for row in per_drift:
                        records.append(
                            {
                                "track": "O",
                                "unit": "sample_idx",
                                "dataset": dataset,
                                "seed": seed,
                                "experiment_name": exp_run.experiment_name,
                                "run_id": exp_run.run_id,
                                "config_tag": config_tag,
                                "model_variant": str(args.model_variant),
                                "monitor_preset": str(args.monitor_preset),
                                "monitor_preset_base": summ.get("monitor_preset_base", ""),
                                "trigger_mode": "two_stage",
                                "confirm_theta": float(theta),
                                "confirm_window": int(window),
                                "confirm_cooldown": int(cooldown),
                                "weights": args.weights,
                                "log_path": str(log_path),
                                "acc_final": summ.get("acc_final"),
                                "mean_acc": summ.get("mean_acc"),
                                "acc_min": summ.get("acc_min"),
                                "MDR_win": win_m.get("MDR"),
                                "MTD_win": win_m.get("MTD"),
                                "MTFA_win": win_m.get("MTFA"),
                                "MTR_win": win_m.get("MTR"),
                                "MDR_tol500": tol_m.get("MDR"),
                                "MTD_tol500": tol_m.get("MTD"),
                                "MTFA_tol500": tol_m.get("MTFA"),
                                "MTR_tol500": tol_m.get("MTR"),
                                "confirmed_count_total": summ.get("confirmed_count_total"),
                                "ph_error_threshold": _safe_float(ph_error.get("threshold") if isinstance(ph_error, dict) else None),
                                "ph_error_min_instances": _safe_int(ph_error.get("min_instances") if isinstance(ph_error, dict) else None),
                                "ph_divergence_threshold": _safe_float(ph_div.get("threshold") if isinstance(ph_div, dict) else None),
                                "ph_divergence_min_instances": _safe_int(ph_div.get("min_instances") if isinstance(ph_div, dict) else None),
                                **row,
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

