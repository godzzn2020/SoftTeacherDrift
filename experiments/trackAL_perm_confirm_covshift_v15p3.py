#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 2: concept-ish vs covariate-ish drift (V15.3 winner config).

输出：
- 聚合 CSV（按 dataset + labelled_ratio 聚合，含 run_index_json）
- 明细 CSV（逐 seed）

支持 --num_shards/--shard_idx 按任务切片（labelled_ratio x dataset）。
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
    p = argparse.ArgumentParser(description="V15.3 Exp2: concept vs covshift (winner config)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--log_root_suffix", type=str, default="", help="用于输出隔离；例如 v15p3_exp2_s0")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, required=True, help="聚合输出 CSV（每个 shard 不同）")
    p.add_argument("--out_raw_csv", type=str, required=True, help="明细输出 CSV（每个 shard 不同）")

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--labelled_ratios", type=str, default="0.05,0.2")
    p.add_argument("--dataset_suite", choices=["concept", "covshift", "both"], default="both")
    p.add_argument("--no_stagger", action="store_true", help="禁用 stagger_abrupt3")
    p.add_argument("--covshift_segment_length", type=int, default=10000)

    p.add_argument("--n_steps", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)

    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_theta", type=float, default=0.50)
    p.add_argument("--confirm_window", type=int, default=3)
    p.add_argument("--confirm_cooldown", type=int, default=200)

    p.add_argument("--perm_alpha", type=float, default=0.03)
    p.add_argument("--perm_pre_n", type=int, default=500)
    p.add_argument("--perm_post_n", type=int, default=10)
    p.add_argument("--perm_n_perm", type=int, default=200)
    p.add_argument("--group_name", type=str, default="")

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    return p.parse_args()


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


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


def build_cfg_map(seed: int, n_steps: int, batch_size: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        cfg2 = replace(cfg, n_steps=int(n_steps), batch_size=int(batch_size))
        mapping[str(cfg2.dataset_name).lower()] = cfg2
    return mapping


def read_run_summary(log_path: Path) -> Dict[str, Any]:
    sp = log_path.with_suffix(".summary.json")
    return json.loads(sp.read_text(encoding="utf-8"))


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
    labelled_ratio: float,
    monitor_preset: str,
    confirm_theta: float,
    confirm_window: int,
    confirm_cooldown: int,
    trigger_weights: Dict[str, Any],
    stream_kwargs: Dict[str, Any],
    device: str,
) -> Path:
    run_paths = exp_run.prepare_dataset_run(dataset_name, "ts_drift_adapt", seed)
    log_path = run_paths.log_csv_path()
    sp = log_path.with_suffix(".summary.json")
    if log_path.exists() and log_path.stat().st_size > 0 and sp.exists() and sp.stat().st_size > 0:
        return log_path

    tw = dict(trigger_weights)
    tw["__confirm_cooldown"] = float(int(confirm_cooldown))

    cfg = replace(
        base_cfg,
        dataset_name=str(dataset_name),
        dataset_type=str(base_cfg.dataset_type),
        stream_kwargs=dict(stream_kwargs),
        labeled_ratio=float(labelled_ratio),
        model_variant="ts_drift_adapt",
        seed=int(seed),
        log_path=str(log_path),
        monitor_preset=str(monitor_preset),
        trigger_mode="two_stage",
        trigger_weights=tw,
        trigger_threshold=float(confirm_theta),
        confirm_window=int(confirm_window),
    )
    _ = run_experiment(cfg, device=str(device))
    run_paths.update_legacy_pointer()
    return log_path


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            fieldnames.append(k)
            seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    experiment_name = "trackAL_perm_confirm_covshift_v15p3"
    out_csv = Path(args.out_csv)
    out_raw_csv = Path(args.out_raw_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_raw_csv.parent.mkdir(parents=True, exist_ok=True)

    logs_root = Path(str(args.logs_root))
    suffix = str(getattr(args, "log_root_suffix", "") or "").strip()
    if suffix:
        logs_root = Path(f"{logs_root}_{suffix}")

    num_shards = int(args.num_shards)
    shard_idx = int(args.shard_idx)
    if num_shards < 1:
        raise ValueError("--num_shards 必须 >= 1")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"--shard_idx 越界：{shard_idx}（期望 0..{num_shards - 1}）")

    seeds = parse_int_list(str(args.seeds))
    if not seeds:
        raise ValueError("--seeds 不能为空")
    labelled_ratios = parse_float_list(str(args.labelled_ratios))
    if not labelled_ratios:
        raise ValueError("--labelled_ratios 不能为空")

    weights_base = parse_weights(str(args.weights))
    perm_alpha = float(args.perm_alpha)
    perm_pre_n = int(args.perm_pre_n)
    perm_post_n = int(args.perm_post_n)
    perm_n_perm = int(args.perm_n_perm)

    group_name = str(args.group_name or "").strip()
    if not group_name:
        group_name = f"P_perm_vote_score_a{perm_alpha:g}_pre{perm_pre_n}_post{perm_post_n}_n5"

    tw = dict(weights_base)
    tw["__confirm_rule"] = "perm_test"
    tw["__perm_stat"] = "vote_score"
    tw["__perm_alpha"] = float(perm_alpha)
    tw["__perm_pre_n"] = float(perm_pre_n)
    tw["__perm_post_n"] = float(perm_post_n)
    tw["__perm_n_perm"] = float(perm_n_perm)
    tw["__perm_min_effect"] = 0.0
    tw["__perm_rng_seed"] = 0.0

    cfg_map = build_cfg_map(1, n_steps=int(args.n_steps), batch_size=int(args.batch_size))
    base_cfg_sea = cfg_map["sea_abrupt4"]
    base_cfg_sine = cfg_map["sine_abrupt4"]
    base_cfg_stagger = cfg_map.get("stagger_abrupt3")

    base_cfg_covshift = replace(base_cfg_sea, dataset_type="covshift_linear")
    covshift_stream_kwargs = {"segment_length": int(args.covshift_segment_length)}

    datasets: List[Dict[str, Any]] = []
    if args.dataset_suite in {"concept", "both"}:
        datasets.extend(
            [
                {"dataset": "sea_abrupt4", "base_name": "sea_abrupt4", "base_cfg": base_cfg_sea, "stream_kwargs": {}, "kind": "concept"},
                {"dataset": "sine_abrupt4", "base_name": "sine_abrupt4", "base_cfg": base_cfg_sine, "stream_kwargs": {}, "kind": "concept"},
            ]
        )
        if not bool(args.no_stagger):
            if base_cfg_stagger is None:
                raise KeyError("default cfg map 未包含 stagger_abrupt3")
            datasets.append(
                {"dataset": "stagger_abrupt3", "base_name": "stagger_abrupt3", "base_cfg": base_cfg_stagger, "stream_kwargs": {}, "kind": "concept"}
            )

    if args.dataset_suite in {"covshift", "both"}:
        datasets.extend(
            [
                {"dataset": "covshift_mean3", "base_name": "covshift_mean3", "base_cfg": base_cfg_covshift, "stream_kwargs": covshift_stream_kwargs, "kind": "covshift"},
                {"dataset": "covshift_scale3", "base_name": "covshift_scale3", "base_cfg": base_cfg_covshift, "stream_kwargs": covshift_stream_kwargs, "kind": "covshift"},
                {"dataset": "covshift_corr3", "base_name": "covshift_corr3", "base_cfg": base_cfg_covshift, "stream_kwargs": covshift_stream_kwargs, "kind": "covshift"},
            ]
        )

    tasks: List[Tuple[float, Dict[str, Any]]] = []
    for lr in labelled_ratios:
        for ds in datasets:
            tasks.append((float(lr), ds))

    if num_shards > 1:
        tasks = [t for i, t in enumerate(tasks) if (i % num_shards) == shard_idx]
        print(f"[shard] num_shards={num_shards} shard_idx={shard_idx} tasks={len(tasks)}")

    tasks_by_lr: Dict[float, List[Dict[str, Any]]] = {}
    for lr, ds in tasks:
        tasks_by_lr.setdefault(lr, []).append(ds)

    n_samples = int(args.n_steps) * int(args.batch_size)
    tol = int(args.tol)
    min_sep = int(args.min_separation)
    gt_starts_default = default_gt_starts(n_samples)

    agg_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    for lr, ds_list in tasks_by_lr.items():
        exp_run = create_experiment_run(
            experiment_name=experiment_name,
            results_root=Path(args.results_root),
            logs_root=logs_root,
            run_name=f"{group_name}_lr{lr}",
        )

        for ds in ds_list:
            dataset_name = str(ds["dataset"])
            base_name = str(ds["base_name"])
            base_cfg = ds["base_cfg"]
            stream_kwargs = dict(ds["stream_kwargs"])
            ds_kind = str(ds["kind"])

            run_index: Dict[str, Any] = {
                dataset_name: {
                    "base_dataset_name": base_name,
                    "dataset_kind": ds_kind,
                    "labelled_ratio": float(lr),
                    "stream_kwargs": stream_kwargs,
                    "runs": [],
                }
            }
            per_seed_records: List[Dict[str, Any]] = []
            for seed in seeds:
                log_path = ensure_log(
                    exp_run,
                    dataset_name=dataset_name,
                    seed=int(seed),
                    base_cfg=base_cfg,
                    labelled_ratio=float(lr),
                    monitor_preset=str(args.monitor_preset),
                    confirm_theta=float(args.confirm_theta),
                    confirm_window=int(args.confirm_window),
                    confirm_cooldown=int(args.confirm_cooldown),
                    trigger_weights=tw,
                    stream_kwargs=stream_kwargs,
                    device=str(args.device),
                )
                summ = read_run_summary(log_path)
                horizon = int(summ.get("horizon") or n_samples)
                confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                confirmed = merge_events(confirmed_raw, min_sep)
                run_id = f"{exp_run.run_id}__{dataset_name}"
                run_index[dataset_name]["runs"].append({"seed": int(seed), "run_id": str(run_id), "log_path": str(log_path)})

                rec: Dict[str, Any] = {
                    "track": "AL",
                    "phase": "exp2",
                    "group": str(group_name),
                    "dataset": dataset_name,
                    "base_dataset_name": base_name,
                    "dataset_kind": ds_kind,
                    "labelled_ratio": float(lr),
                    "seed": int(seed),
                    "run_id": str(run_id),
                    "log_path": str(log_path),
                    "horizon": int(horizon),
                    "confirm_rule": "perm_test",
                    "perm_stat": "vote_score",
                    "perm_alpha": float(perm_alpha),
                    "perm_pre_n": int(perm_pre_n),
                    "perm_post_n": int(perm_post_n),
                    "perm_n_perm": int(perm_n_perm),
                    "confirm_theta": float(args.confirm_theta),
                    "confirm_window": int(args.confirm_window),
                    "confirm_cooldown": int(args.confirm_cooldown),
                    "acc_final": _safe_float(summ.get("acc_final")),
                }

                if ds_kind == "concept":
                    gt_starts = list(gt_starts_default)
                    if dataset_name.lower() == "stagger_abrupt3":
                        gt_starts = [x for x in (20000, 40000) if int(x) < int(horizon)]
                    delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=tol)
                    rec["miss_tol500"] = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None
                    rec["conf_P90"] = percentile(delays, 0.90)
                    rec["confirm_rate_per_10k"] = None
                    rec["MTFA_win"] = None
                else:
                    cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
                    rec["confirm_rate_per_10k"] = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                    rec["MTFA_win"] = mtfa_from_false_alarms(confirmed, horizon)
                    rec["miss_tol500"] = None
                    rec["conf_P90"] = None

                raw_rows.append(dict(rec))
                per_seed_records.append(dict(rec))

            agg: Dict[str, Any] = {
                "track": "AL",
                "phase": "exp2",
                "dataset": dataset_name,
                "dataset_kind": ds_kind,
                "base_dataset_name": base_name,
                "labelled_ratio": float(lr),
                "unit": "sample_idx",
                "n_runs": int(len(seeds)),
                "group": str(group_name),
                "monitor_preset": str(args.monitor_preset),
                "confirm_theta": float(args.confirm_theta),
                "confirm_window": int(args.confirm_window),
                "confirm_cooldown": int(args.confirm_cooldown),
                "confirm_rule": "perm_test",
                "perm_stat": "vote_score",
                "perm_alpha": float(perm_alpha),
                "perm_pre_n": int(perm_pre_n),
                "perm_post_n": int(perm_post_n),
                "perm_n_perm": int(perm_n_perm),
                "weights": str(args.weights),
                "horizon_mean": mean([_safe_float(r.get("horizon")) for r in per_seed_records]),
                "acc_final_mean": mean([_safe_float(r.get("acc_final")) for r in per_seed_records]),
                "acc_final_std": std([_safe_float(r.get("acc_final")) for r in per_seed_records]),
                "miss_tol500_mean": mean([_safe_float(r.get("miss_tol500")) for r in per_seed_records]),
                "conf_P90_mean": mean([_safe_float(r.get("conf_P90")) for r in per_seed_records]),
                "MTFA_win_mean": mean([_safe_float(r.get("MTFA_win")) for r in per_seed_records]),
                "MTFA_win_std": std([_safe_float(r.get("MTFA_win")) for r in per_seed_records]),
                "confirm_rate_per_10k_mean": mean([_safe_float(r.get("confirm_rate_per_10k")) for r in per_seed_records]),
                "confirm_rate_per_10k_std": std([_safe_float(r.get("confirm_rate_per_10k")) for r in per_seed_records]),
                "run_index_json": json.dumps(run_index, ensure_ascii=False),
            }
            agg_rows.append(agg)

    write_csv(out_csv, agg_rows)
    write_csv(out_raw_csv, raw_rows)
    print(f"[done] wrote {out_csv} rows={len(agg_rows)}")
    print(f"[done] wrote {out_raw_csv} rows={len(raw_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
