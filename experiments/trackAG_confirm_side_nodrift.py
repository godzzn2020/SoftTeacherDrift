#!/usr/bin/env python
"""
NEXT_STAGE V12 - Track AG（必做）：confirm-side sweep（no-drift 约束优化）

固定 detector：
- trigger=two_stage(candidate=OR, confirm=weighted)
- monitor_preset=error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5
- divergence 保持 preset 默认（期望 0.05/30）

只扫 confirm-side：
  confirm_theta ∈ {0.50,0.55,0.60}
  confirm_window ∈ {1,2,3}
  confirm_cooldown ∈ {200,500,1000}

数据集与 seeds：
- drift：sea_gradual_frequent、sine_gradual_frequent（seeds=1..5）
- no-drift：sea_nodrift（seeds=1..5）

实现说明（不做 data/synthetic 临时替换）：
- 训练实际跑 base_dataset（sea_abrupt4/sine_abrupt4），通过 cfg.stream_kwargs 控制 gradual/no-drift
- CSV 的 dataset 字段使用“别名”（sea_gradual_frequent/sea_nodrift）用于报告口径

输出：scripts/TRACKAG_CONFIRM_SIDE_NODRIFT.csv（按 config 聚合，包含 mean 字段与 run 索引 JSON）
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
    p = argparse.ArgumentParser(description="Track AG: confirm-side sweep with no-drift constraint")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAG_CONFIRM_SIDE_NODRIFT.csv")

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--transition_length", type=int, default=2000)

    p.add_argument("--confirm_thetas", type=str, default="0.50,0.55,0.60")
    p.add_argument("--confirm_windows", type=str, default="1,2,3")
    p.add_argument("--confirm_cooldowns", type=str, default="200,500,1000")

    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5",
    )
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


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


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


def std(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


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


def default_gt_starts(n_samples: int) -> List[int]:
    step = max(1, int(n_samples) // 5)
    return [int(step), int(2 * step), int(3 * step), int(4 * step)]


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
        delays.append(float(end - g) if first_f is None else float(first_f - g))
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

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    confirm_thetas = [float(x) for x in str(args.confirm_thetas).split(",") if str(x).strip()]
    confirm_windows = [int(x) for x in str(args.confirm_windows).split(",") if str(x).strip()]
    confirm_cooldowns = [int(x) for x in str(args.confirm_cooldowns).split(",") if str(x).strip()]

    weights_base = parse_weights(str(args.weights))
    warmup = int(args.warmup_samples)
    tol = int(args.tol)
    horizon = int(args.n_samples)
    gt_starts = default_gt_starts(horizon)

    cfg_map = build_cfg_map(1)
    base_cfg_sea = cfg_map["sea_abrupt4"]
    base_cfg_sine = cfg_map["sine_abrupt4"]

    # dataset alias -> (base_dataset_name, base_cfg, stream_kwargs)
    datasets: List[Tuple[str, str, ExperimentConfig, Dict[str, Any]]] = [
        ("sea_gradual_frequent", "sea_abrupt4", base_cfg_sea, {"drift_type": "gradual", "transition_length": int(args.transition_length)}),
        ("sine_gradual_frequent", "sine_abrupt4", base_cfg_sine, {"drift_type": "gradual", "transition_length": int(args.transition_length)}),
        ("sea_nodrift", "sea_abrupt4", base_cfg_sea, {"concept_ids": [0], "concept_length": int(args.n_samples), "drift_type": "abrupt"}),
    ]

    rows_out: List[Dict[str, Any]] = []
    for theta in confirm_thetas:
        for window in confirm_windows:
            for cooldown in confirm_cooldowns:
                exp_run = create_experiment_run(
                    experiment_name="trackAG_confirm_side_nodrift",
                    results_root=Path(args.results_root),
                    logs_root=Path(args.logs_root),
                    run_name=f"theta{theta:.2f}_w{int(window)}_cd{int(cooldown)}",
                )
                # 收集每个 dataset 的 per-seed 结果，用于 mean 聚合
                by_ds: Dict[str, List[Dict[str, Any]]] = {}
                run_index: Dict[str, Dict[str, Any]] = {}
                for ds_alias, base_name, base_cfg, stream_kwargs in datasets:
                    by_ds[ds_alias] = []
                    run_index[ds_alias] = {"base_dataset_name": base_name, "stream_kwargs": stream_kwargs, "runs": []}
                    for seed in seeds:
                        log_path = ensure_log(
                            exp_run,
                            dataset_name=base_name,
                            seed=seed,
                            base_cfg=base_cfg,
                            monitor_preset=str(args.monitor_preset),
                            confirm_theta=float(theta),
                            confirm_window=int(window),
                            confirm_cooldown=int(cooldown),
                            trigger_weights=weights_base,
                            stream_kwargs=stream_kwargs,
                            device=str(args.device),
                        )
                        summ = read_run_summary(log_path)
                        confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                        confirmed = merge_events(confirmed_raw, int(args.min_separation))

                        rec: Dict[str, Any] = {
                            "seed": int(seed),
                            "acc_final": _safe_float(summ.get("acc_final")),
                            f"acc_min@{warmup}": acc_min_after_warmup(summ, warmup),
                        }
                        if ds_alias.endswith("nodrift"):
                            cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
                            rec["confirm_rate_per_10k"] = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                            rec["MTFA_win"] = mtfa_from_false_alarms(confirmed, horizon)
                            rec["confirmed_count_total"] = cc
                        else:
                            delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=tol)
                            rec["miss_tol500"] = (float(sum(miss_flags)) / len(miss_flags)) if miss_flags else None
                            rec["conf_P90"] = percentile(delays, 0.90) if delays else None
                            win_m = compute_detection_metrics(gt_starts, confirmed_raw, horizon) if gt_starts else {"MTFA": None}
                            rec["MTFA_win"] = _safe_float(win_m.get("MTFA"))
                        by_ds[ds_alias].append(rec)
                        run_index[ds_alias]["runs"].append({"seed": int(seed), "run_id": exp_run.run_id, "log_path": str(log_path)})

                # 聚合输出：每个 config 一行，含 sea/sine drift + nodrift
                sea = by_ds.get("sea_gradual_frequent") or []
                sine = by_ds.get("sine_gradual_frequent") or []
                nodrift = by_ds.get("sea_nodrift") or []

                def _agg(rs: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
                    vals = [_safe_float(r.get(key)) for r in rs]
                    return mean(vals), std(vals)

                sea_miss_mu, _ = _agg(sea, "miss_tol500")
                sea_conf_mu, _ = _agg(sea, "conf_P90")
                sine_miss_mu, _ = _agg(sine, "miss_tol500")
                sine_conf_mu, _ = _agg(sine, "conf_P90")

                nd_rate_mu, _ = _agg(nodrift, "confirm_rate_per_10k")
                nd_mtfa_mu, _ = _agg(nodrift, "MTFA_win")

                drift_acc_mu = mean([mean([_safe_float(r.get("acc_final")) for r in sea]), mean([_safe_float(r.get("acc_final")) for r in sine])])
                drift_accmin_mu = mean([mean([_safe_float(r.get(f"acc_min@{warmup}")) for r in sea]), mean([_safe_float(r.get(f"acc_min@{warmup}")) for r in sine])])
                drift_mtfa_mu = mean([mean([_safe_float(r.get("MTFA_win")) for r in sea]), mean([_safe_float(r.get("MTFA_win")) for r in sine])])

                rows_out.append(
                    {
                        "track": "AG",
                        "unit": "sample_idx",
                        "monitor_preset": str(args.monitor_preset),
                        "confirm_theta": float(theta),
                        "confirm_window": int(window),
                        "confirm_cooldown": int(cooldown),
                        "n_seeds": int(len(seeds)),
                        "sea_miss_tol500_mean": sea_miss_mu,
                        "sea_conf_P90_mean": sea_conf_mu,
                        "sine_miss_tol500_mean": sine_miss_mu,
                        "sine_conf_P90_mean": sine_conf_mu,
                        "drift_acc_final_mean": drift_acc_mu,
                        f"drift_acc_min@{warmup}_mean": drift_accmin_mu,
                        "drift_MTFA_win_mean": drift_mtfa_mu,
                        "no_drift_confirm_rate_per_10k_mean": nd_rate_mu,
                        "no_drift_MTFA_win_mean": nd_mtfa_mu,
                        "run_index_json": json.dumps(run_index, ensure_ascii=False),
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

