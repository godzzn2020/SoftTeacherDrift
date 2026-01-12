#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15.2 稳定性复核（小规模）：
- 固定一组候选（baseline + 3 个 vote_score 候选）
- 扩展 seeds（默认 1..10）
- 扩展 drift 维度：可选加入 stagger_abrupt3 与 sea/sine 的 gradual_frequent

输出：
- 聚合 CSV（按 dataset+group，跨 seeds 统计 mean/std；含 run_index_json）
- 明细 CSV（逐 seed 逐 dataset；用于稳定性统计与最差 case 追溯）

注意：支持 --num_shards/--shard_idx 按 group 分片；每个 shard 必须显式隔离 out_csv/log_root_suffix/out_raw_csv。
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
    p = argparse.ArgumentParser(description="V15.2: vote_score stability check")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--log_root_suffix", type=str, default="", help="用于输出隔离；例如 v15p2_s0 -> logs_v15p2_s0")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, required=True, help="聚合输出 CSV（每个 shard 不同）")
    p.add_argument("--out_raw_csv", type=str, required=True, help="明细输出 CSV（每个 shard 不同）")

    p.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--transition_length", type=int, default=2000)
    p.add_argument("--include_gradual", action="store_true", help="额外加入 sea/sine gradual_frequent")
    p.add_argument("--include_stagger", action="store_true", help="额外加入 stagger_abrupt3")

    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_theta", type=float, default=0.50)
    p.add_argument("--confirm_window", type=int, default=3)
    p.add_argument("--confirm_cooldown", type=int, default=200)

    p.add_argument("--perm_n_perm", type=int, default=200)
    p.add_argument("--perm_pre_n", type=int, default=500)
    p.add_argument("--perm_post_n", type=int, default=10)
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
        mapping[str(cfg.dataset_name).lower()] = cfg
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

    weights_base = parse_weights(str(args.weights))
    perm_n_perm = int(args.perm_n_perm)
    perm_pre_n = int(args.perm_pre_n)
    perm_post_n = int(args.perm_post_n)

    # 固定候选列表（按 prompt）
    groups: List[Dict[str, Any]] = []
    groups.append(
        {
            "group": "A_weighted_n5",
            "confirm_rule": "weighted",
            "perm_stat": "N/A",
            "delta_k": "N/A",
            "perm_alpha": "N/A",
            "perm_pre_n": "N/A",
            "perm_post_n": "N/A",
            "perm_n_perm": "N/A",
            "trigger_weights": dict(weights_base),
        }
    )
    for alpha in (0.02, 0.025, 0.03):
        name = f"P_perm_vote_score_a{alpha:g}_pre500_post10_n5"
        tw = dict(weights_base)
        tw["__confirm_rule"] = "perm_test"
        tw["__perm_stat"] = "vote_score"
        tw["__perm_alpha"] = float(alpha)
        tw["__perm_pre_n"] = float(perm_pre_n)
        tw["__perm_post_n"] = float(perm_post_n)
        tw["__perm_n_perm"] = float(perm_n_perm)
        tw["__perm_min_effect"] = 0.0
        tw["__perm_rng_seed"] = 0.0
        groups.append(
            {
                "group": name,
                "confirm_rule": "perm_test",
                "perm_stat": "vote_score",
                "delta_k": "N/A",
                "perm_alpha": float(alpha),
                "perm_pre_n": int(perm_pre_n),
                "perm_post_n": int(perm_post_n),
                "perm_n_perm": int(perm_n_perm),
                "trigger_weights": tw,
            }
        )

    groups = sorted(groups, key=lambda g: str(g.get("group") or ""))
    if num_shards > 1:
        groups = [g for i, g in enumerate(groups) if (i % num_shards) == shard_idx]
        print(f"[shard] num_shards={num_shards} shard_idx={shard_idx} groups={len(groups)}")

    n_samples = int(args.n_samples)
    warmup = int(args.warmup_samples)
    tol = int(args.tol)
    min_sep = int(args.min_separation)
    gt_starts = default_gt_starts(n_samples)

    # 为 nodrift/gradual 的 dataset 别名注册 stream 配置（仅本进程内）
    from data import streams as _streams

    sea_base = dict(_streams.SEA_CONFIGS.get("sea_abrupt4") or {"concept_ids": [0, 1, 2, 3, 0], "concept_length": 10000})
    _streams.SEA_CONFIGS.setdefault("sea_nodrift", {"concept_ids": [0], "concept_length": int(n_samples)})
    _streams.SEA_CONFIGS.setdefault("sea_gradual_frequent", sea_base)

    sine_base = dict(_streams.SINE_DEFAULT.get("sine_abrupt4") or {"classification_functions": [0, 1, 2, 3, 0], "segment_length": 10000, "balance_classes": False, "has_noise": False})
    _streams.SINE_DEFAULT.setdefault("sine_nodrift", {**sine_base, "classification_functions": [0]})
    _streams.SINE_DEFAULT.setdefault("sine_gradual_frequent", sine_base)

    cfg_map = build_cfg_map(1)
    base_cfg_sea = cfg_map["sea_abrupt4"]
    base_cfg_sine = cfg_map["sine_abrupt4"]
    base_cfg_stagger = cfg_map.get("stagger_abrupt3")

    datasets: List[Tuple[str, str, ExperimentConfig, Dict[str, Any], str]] = [
        ("sea_nodrift", "sea_abrupt4", base_cfg_sea, {"concept_ids": [0], "concept_length": int(n_samples), "drift_type": "abrupt"}, "nodrift"),
        ("sine_nodrift", "sine_abrupt4", base_cfg_sine, {"concept_ids": [0], "concept_length": int(n_samples), "drift_type": "abrupt"}, "nodrift"),
        ("sea_abrupt4", "sea_abrupt4", base_cfg_sea, {}, "drift"),
        ("sine_abrupt4", "sine_abrupt4", base_cfg_sine, {}, "drift"),
    ]
    if bool(args.include_gradual):
        tl = int(args.transition_length)
        datasets.extend(
            [
                ("sea_gradual_frequent", "sea_abrupt4", base_cfg_sea, {"drift_type": "gradual", "transition_length": tl}, "drift"),
                ("sine_gradual_frequent", "sine_abrupt4", base_cfg_sine, {"drift_type": "gradual", "transition_length": tl}, "drift"),
            ]
        )
    if bool(args.include_stagger):
        if base_cfg_stagger is None:
            raise KeyError("default cfg map 未包含 stagger_abrupt3")
        datasets.append(("stagger_abrupt3", "stagger_abrupt3", base_cfg_stagger, {}, "drift"))

    # 执行
    phase = "stability"
    agg_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    for g in groups:
        exp_run = create_experiment_run(
            experiment_name="trackAL_perm_confirm_stability_v15p2",
            results_root=Path(args.results_root),
            logs_root=logs_root,
            run_name=str(g["group"]),
        )

        for ds_alias, base_name, base_cfg, stream_kwargs, ds_kind in datasets:
            run_index: Dict[str, Any] = {ds_alias: {"base_dataset_name": base_name, "dataset_kind": ds_kind, "stream_kwargs": stream_kwargs, "runs": []}}
            per_seed_records: List[Dict[str, Any]] = []
            for seed in seeds:
                log_path = ensure_log(
                    exp_run,
                    dataset_name=ds_alias,
                    seed=int(seed),
                    base_cfg=base_cfg,
                    monitor_preset=str(args.monitor_preset),
                    confirm_theta=float(args.confirm_theta),
                    confirm_window=int(args.confirm_window),
                    confirm_cooldown=int(args.confirm_cooldown),
                    trigger_weights=dict(g["trigger_weights"]),
                    stream_kwargs=stream_kwargs,
                    device=str(args.device),
                )
                summ = read_run_summary(log_path)
                horizon = int(summ.get("horizon") or n_samples)
                confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                confirmed = merge_events(confirmed_raw, min_sep)
                run_id = f"{exp_run.run_id}__{ds_alias}"
                run_index[ds_alias]["runs"].append({"seed": int(seed), "run_id": str(run_id), "log_path": str(log_path)})

                rec: Dict[str, Any] = {
                    "track": "AL",
                    "phase": phase,
                    "group": str(g["group"]),
                    "dataset": str(ds_alias),
                    "base_dataset_name": str(base_name),
                    "dataset_kind": str(ds_kind),
                    "seed": int(seed),
                    "run_id": str(run_id),
                    "log_path": str(log_path),
                    "horizon": int(horizon),
                    "confirm_rule": str(g["confirm_rule"]),
                    "perm_stat": str(g["perm_stat"]),
                    "perm_alpha": str(g["perm_alpha"]),
                    "perm_pre_n": str(g["perm_pre_n"]),
                    "perm_post_n": str(g["perm_post_n"]),
                    "perm_n_perm": str(g["perm_n_perm"]),
                    "delta_k": str(g["delta_k"]),
                    "acc_final": _safe_float(summ.get("acc_final")),
                }

                if ds_kind == "nodrift":
                    cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
                    rec["confirm_rate_per_10k"] = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                    rec["MTFA_win"] = mtfa_from_false_alarms(confirmed, horizon)
                    rec["miss_tol500"] = None
                    rec["conf_P90"] = None
                else:
                    delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=tol)
                    rec["miss_tol500"] = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None
                    rec["conf_P90"] = percentile(delays, 0.90)
                    rec["confirm_rate_per_10k"] = None
                    rec["MTFA_win"] = None

                raw_rows.append(dict(rec))
                per_seed_records.append(dict(rec))

            # 聚合一行（跨 seeds）
            agg: Dict[str, Any] = {
                "track": "AL",
                "phase": phase,
                "dataset": str(ds_alias),
                "dataset_kind": str(ds_kind),
                "base_dataset_name": str(base_name),
                "unit": "sample_idx",
                "n_runs": int(len(seeds)),
                "group": str(g["group"]),
                "monitor_preset": str(args.monitor_preset),
                "confirm_theta": float(args.confirm_theta),
                "confirm_window": int(args.confirm_window),
                "confirm_cooldown": int(args.confirm_cooldown),
                "confirm_rule": str(g["confirm_rule"]),
                "perm_stat": str(g["perm_stat"]),
                "delta_k": str(g["delta_k"]),
                "perm_alpha": str(g["perm_alpha"]),
                "perm_pre_n": str(g["perm_pre_n"]),
                "perm_post_n": str(g["perm_post_n"]),
                "perm_n_perm": str(g["perm_n_perm"]),
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
