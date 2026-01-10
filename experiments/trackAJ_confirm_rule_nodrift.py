#!/usr/bin/env python
"""
NEXT_STAGE V13 - Track AJ（必做）：Confirm-rule ablation for no-drift

目标：在满足 drift 约束（miss==0 且 confP90<500）下，显著降低 no-drift confirm_rate。

固定 detector：
- trigger=two_stage(candidate=OR, confirm=*)
- monitor_preset=error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5（divergence 默认 0.05/30）
- candidate 侧保持 OR，不改

对比 confirm 规则（至少 3 组）：
A) confirm=weighted（baseline）
   - theta=0.50, window=3, cooldown=200
B) confirm=k_of_n(k=2)
   - window=3, cooldown=200
   - k_of_n 的 hit 定义为 vote_score>=theta（沿用 0.50），在窗口内累计 hit>=k 才确认
C) confirm=weighted + divergence_gate（L=window steps）
D) confirm=weighted + divergence_gate（L=500 sample_idx）

数据集与 seeds：
- drift：sea_gradual_frequent、sine_gradual_frequent（seeds=1..5）
- no-drift：sea_nodrift（seeds=1..5）

实现说明：
- 训练实际跑 base_dataset（sea_abrupt4/sine_abrupt4），通过 cfg.stream_kwargs 控制 gradual/no-drift；
  CSV 的 dataset 字段使用别名用于口径一致。

输出：scripts/TRACKAJ_CONFIRM_RULE_NODRIFT.csv（按 group 聚合，含 run_index_json 用于精确定位）
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
    p = argparse.ArgumentParser(description="Track AJ: confirm-rule ablation for no-drift")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAJ_CONFIRM_RULE_NODRIFT.csv")

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--transition_length", type=int, default=2000)

    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument(
        "--divgate_value_thrs",
        type=str,
        default="0.01",
        help="divergence_gate 的 value_thr 候选（逗号分隔，仅用于 confirm-side ablation）",
    )
    p.add_argument(
        "--errgate_thrs",
        type=str,
        default="0.5,0.6667",
        help="error_gate 的阈值候选（逗号分隔，仅用于 confirm_rule=weighted+error_gate）",
    )
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


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def fmt_thr(thr: float) -> str:
    s = f"{float(thr):.3f}"
    s = s.rstrip("0").rstrip(".")
    return s


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
    run_dataset_key: str,
    base_dataset_name: str,
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
    run_paths = exp_run.prepare_dataset_run(str(run_dataset_key), "ts_drift_adapt", seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    weights = dict(trigger_weights)
    weights["confirm_cooldown"] = float(int(confirm_cooldown))
    cfg = replace(
        base_cfg,
        dataset_name=str(base_dataset_name),
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
    horizon = int(args.n_samples)
    gt_starts = default_gt_starts(horizon)
    warmup = int(args.warmup_samples)
    tol = int(args.tol)

    cfg_map = build_cfg_map(1)
    base_cfg_sea = cfg_map["sea_abrupt4"]
    base_cfg_sine = cfg_map["sine_abrupt4"]

    # fixed confirm-side params (from V12 winner)
    theta = 0.50
    window = 3
    cooldown = 200
    weights_base = parse_weights(str(args.weights))
    divgate_value_thrs = parse_float_list(str(args.divgate_value_thrs))
    errgate_thrs = parse_float_list(str(args.errgate_thrs))

    datasets: List[Tuple[str, str, ExperimentConfig, Dict[str, Any]]] = [
        ("sea_gradual_frequent", "sea_abrupt4", base_cfg_sea, {"drift_type": "gradual", "transition_length": int(args.transition_length)}),
        ("sine_gradual_frequent", "sine_abrupt4", base_cfg_sine, {"drift_type": "gradual", "transition_length": int(args.transition_length)}),
        ("sea_nodrift", "sea_abrupt4", base_cfg_sea, {"concept_ids": [0], "concept_length": int(args.n_samples), "drift_type": "abrupt"}),
    ]

    groups: List[Dict[str, Any]] = [
        {
            "group": "A_weighted",
            "confirm_rule": 0.0,
            "confirm_k": 2.0,
            "divergence_gate": 0.0,
            "divergence_gate_steps": 0.0,
            "divergence_gate_samples": 0.0,
            "divergence_gate_value_thr": float("nan"),
        },
        {
            "group": "B_k_of_n_k2",
            "confirm_rule": 1.0,
            "confirm_k": 2.0,
            "divergence_gate": 0.0,
            "divergence_gate_steps": 0.0,
            "divergence_gate_samples": 0.0,
            "divergence_gate_value_thr": float("nan"),
            "confirm_error_gate_thr": float("nan"),
        },
    ]
    for thr in divgate_value_thrs:
        thr_s = fmt_thr(thr)
        groups.append(
            {
                "group": f"C_weighted_divgate_LstepsW_thr{thr_s}",
                "confirm_rule": 0.0,
                "confirm_k": 2.0,
                "divergence_gate": 1.0,
                "divergence_gate_steps": float(window),
                "divergence_gate_samples": 0.0,
                "divergence_gate_value_thr": float(thr),
                "confirm_error_gate_thr": float("nan"),
            }
        )
        groups.append(
            {
                "group": f"D_weighted_divgate_Lsamples500_thr{thr_s}",
                "confirm_rule": 0.0,
                "confirm_k": 2.0,
                "divergence_gate": 1.0,
                "divergence_gate_steps": 0.0,
                "divergence_gate_samples": 500.0,
                "divergence_gate_value_thr": float(thr),
                "confirm_error_gate_thr": float("nan"),
            }
        )
    for thr in errgate_thrs:
        thr_s = fmt_thr(thr)
        groups.append(
            {
                "group": f"E_weighted_errgate_thr{thr_s}",
                "confirm_rule": 2.0,
                "confirm_k": 2.0,
                "divergence_gate": 0.0,
                "divergence_gate_steps": 0.0,
                "divergence_gate_samples": 0.0,
                "divergence_gate_value_thr": float("nan"),
                "confirm_error_gate_thr": float(thr),
            }
        )

    records: List[Dict[str, Any]] = []
    for g in groups:
        group = str(g["group"])
        exp_run = create_experiment_run(
            experiment_name="trackAJ_confirm_rule_nodrift",
            results_root=Path(args.results_root),
            logs_root=Path(args.logs_root),
            run_name=group,
        )
        by_ds: Dict[str, List[Dict[str, Any]]] = {}
        run_index: Dict[str, Dict[str, Any]] = {}
        for ds_alias, base_name, base_cfg, stream_kwargs in datasets:
            by_ds[ds_alias] = []
            run_index[ds_alias] = {"base_dataset_name": base_name, "stream_kwargs": stream_kwargs, "runs": []}
            for seed in seeds:
                tw = dict(weights_base)
                tw["confirm_rule"] = float(g["confirm_rule"])
                tw["confirm_k"] = float(g["confirm_k"])
                tw["divergence_gate"] = float(g["divergence_gate"])
                tw["divergence_gate_steps"] = float(g["divergence_gate_steps"])
                tw["divergence_gate_samples"] = float(g["divergence_gate_samples"])
                if not math.isnan(float(g["divergence_gate_value_thr"])):
                    tw["divergence_gate_value_thr"] = float(g["divergence_gate_value_thr"])
                if not math.isnan(float(g.get("confirm_error_gate_thr", float("nan")))):
                    tw["confirm_error_gate_thr"] = float(g["confirm_error_gate_thr"])

                log_path = ensure_log(
                    exp_run,
                    run_dataset_key=ds_alias,
                    base_dataset_name=base_name,
                    seed=seed,
                    base_cfg=base_cfg,
                    monitor_preset=str(args.monitor_preset),
                    confirm_theta=float(theta),
                    confirm_window=int(window),
                    confirm_cooldown=int(cooldown),
                    trigger_weights=tw,
                    stream_kwargs=stream_kwargs,
                    device=str(args.device),
                )
                summ = read_run_summary(log_path)
                confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                confirmed = merge_events(confirmed_raw, int(args.min_separation))

                rec: Dict[str, Any] = {
                    "seed": int(seed),
                    "acc_final": _safe_float(summ.get("acc_final")),
                    "mean_acc": _safe_float(summ.get("mean_acc")),
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

        sea = by_ds.get("sea_gradual_frequent") or []
        sine = by_ds.get("sine_gradual_frequent") or []
        nodrift = by_ds.get("sea_nodrift") or []

        records.append(
            {
                "track": "AJ",
                "group": group,
                "monitor_preset": str(args.monitor_preset),
                "confirm_theta": float(theta),
                "confirm_window": int(window),
                "confirm_cooldown": int(cooldown),
                "confirm_rule_code": float(g["confirm_rule"]),
                "confirm_k": float(g["confirm_k"]),
                "divergence_gate": float(g["divergence_gate"]),
                "divergence_gate_steps": float(g["divergence_gate_steps"]),
                "divergence_gate_samples": float(g["divergence_gate_samples"]),
                "divergence_gate_value_thr": float(g["divergence_gate_value_thr"]),
                "confirm_error_gate_thr": float(g.get("confirm_error_gate_thr", float("nan"))),
                "n_seeds": int(len(seeds)),
                "sea_miss": mean([_safe_float(r.get("miss_tol500")) for r in sea]),
                "sea_confP90": mean([_safe_float(r.get("conf_P90")) for r in sea]),
                "sine_miss": mean([_safe_float(r.get("miss_tol500")) for r in sine]),
                "sine_confP90": mean([_safe_float(r.get("conf_P90")) for r in sine]),
                "drift_acc_final": mean([mean([_safe_float(r.get("acc_final")) for r in sea]), mean([_safe_float(r.get("acc_final")) for r in sine])]),
                "no_drift_confirm_rate_per_10k": mean([_safe_float(r.get("confirm_rate_per_10k")) for r in nodrift]),
                "no_drift_MTFA_win": mean([_safe_float(r.get("MTFA_win")) for r in nodrift]),
                "no_drift_acc_final": mean([_safe_float(r.get("acc_final")) for r in nodrift]),
                "no_drift_mean_acc": mean([_safe_float(r.get("mean_acc")) for r in nodrift]),
                "run_index_json": json.dumps(run_index, ensure_ascii=False),
            }
        )

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
