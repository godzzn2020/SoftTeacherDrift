#!/usr/bin/env python
"""
NEXT_STAGE V14 - Track AL（核心）：Permutation-test confirm sweep

目标：在满足 drift 约束（sea_abrupt4 + sine_abrupt4：miss_tol500==0 且 conf_P90<500）下，
显著降低 no-drift（sea_nodrift + sine_nodrift）的 confirm_rate_per_10k（次选最大化 MTFA_win）。

固定：
- trigger_mode=two_stage（candidate=OR, confirm=confirm_rule）
- monitor_preset=error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5（divergence 默认 0.05/30）
- confirm_theta=0.50, confirm_window=3, confirm_cooldown=200

对比：
- Baseline：confirm_rule=weighted
- Perm：confirm_rule=perm_test（sweep __perm_* 参数）

输出：scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv（按 dataset+group 聚合，含 run_index_json 精确定位）
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
    p = argparse.ArgumentParser(description="Track AL: permutation-test confirm sweep")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument(
        "--log_root_suffix",
        type=str,
        default="",
        help="日志根目录后缀（用于输出隔离；例如 v14fix1 -> logs_v14fix1）",
    )
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv")

    p.add_argument("--seeds_quick", type=str, default="1,2,3,4,5")
    p.add_argument("--seeds_full", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    p.add_argument("--run_full", action="store_true", help="对 quick winner/topK 追加 full seeds")
    p.add_argument("--full_topk", type=int, default=2, help="满足约束且 no-drift 最优的前 K 个组合做 full")

    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--transition_length", type=int, default=2000)
    p.add_argument("--include_gradual", action="store_true")

    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5",
    )
    p.add_argument(
        "--signal_set",
        type=str,
        default=None,
        choices=["error", "proxy", "all"],
        help="信号组合（可选）：error=仅监督，proxy=entropy+divergence，all=三者；为空则沿用 monitor_preset",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")

    p.add_argument("--confirm_theta", type=float, default=0.50)
    p.add_argument("--confirm_window", type=int, default=3)
    p.add_argument("--confirm_cooldown", type=int, default=200)

    # sweep grid
    p.add_argument(
        "--perm_stats",
        type=str,
        default="fused_score,delta_fused_score,vote_score",
        help="逗号分隔 perm_stat 列表（例如 vote_score）",
    )
    p.add_argument("--perm_alphas", type=str, default="0.05,0.02,0.01,0.005")
    p.add_argument("--perm_pre_ns", type=str, default="200,500")
    p.add_argument("--perm_post_ns", type=str, default="10,20,30,50")
    p.add_argument("--perm_n_perm", type=int, default=200)
    p.add_argument("--delta_ks", type=str, default="25,50")
    p.add_argument("--num_shards", type=int, default=1, help="将 groups 切成 N 片（用于并行）；默认 1=不切片")
    p.add_argument("--shard_idx", type=int, default=0, help="当前分片编号 [0, num_shards)")
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
    # sea/sine_abrupt4 的默认 meta：4 次漂移，等间隔（与历史脚本保持一致口径）
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


def acc_min_after_warmup(summary: Dict[str, Any], warmup: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals: List[float] = []
    for item in series:
        try:
            pos = int(item[0])
            acc = float(item[1])
        except Exception:
            continue
        if pos >= int(warmup) and not math.isnan(acc):
            vals.append(acc)
    return float(min(vals)) if vals else None


def ensure_log(
    exp_run: ExperimentRun,
    dataset_name: str,
    seed: int,
    base_cfg: ExperimentConfig,
    *,
    monitor_preset: str,
    signal_set: Optional[str],
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
        signal_set=signal_set,
        trigger_mode="two_stage",
        trigger_weights=tw,
        trigger_threshold=float(confirm_theta),
        confirm_window=int(confirm_window),
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


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
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 输出隔离（强制推荐）：避免复用旧 logs 导致 nodrift/drift 混淆
    logs_root = Path(str(args.logs_root))
    suffix = str(getattr(args, "log_root_suffix", "") or "").strip()
    if suffix:
        logs_root = Path(f"{logs_root}_{suffix}")

    seeds_quick = parse_int_list(str(args.seeds_quick))
    seeds_full = parse_int_list(str(args.seeds_full))
    weights_base = parse_weights(str(args.weights))

    n_samples = int(args.n_samples)
    warmup = int(args.warmup_samples)
    tol = int(args.tol)
    min_sep = int(args.min_separation)
    gt_starts = default_gt_starts(n_samples)

    # 关键修复配套：为 nodrift/gradual 的 dataset 别名注册可运行的 stream 配置
    # - 仅影响本脚本进程内的 streams 配置，不改动任何代码文件
    # - 目的：允许 dataset_name=sea_nodrift/sine_nodrift 写入 summary 并按 ds_alias 分目录落盘
    from data import streams as _streams

    sea_base = dict(_streams.SEA_CONFIGS.get("sea_abrupt4") or {"concept_ids": [0, 1, 2, 3, 0], "concept_length": 10000})
    _streams.SEA_CONFIGS.setdefault("sea_nodrift", {"concept_ids": [0], "concept_length": int(sea_base.get("concept_length") or 10000)})
    _streams.SEA_CONFIGS.setdefault("sea_gradual_frequent", sea_base)

    sine_base = dict(_streams.SINE_DEFAULT.get("sine_abrupt4") or {"classification_functions": [0, 1, 2, 3, 0], "segment_length": 10000, "balance_classes": False, "has_noise": False})
    _streams.SINE_DEFAULT.setdefault(
        "sine_nodrift",
        {**sine_base, "classification_functions": [0]},
    )
    _streams.SINE_DEFAULT.setdefault("sine_gradual_frequent", sine_base)

    cfg_map = build_cfg_map(1)
    base_cfg_sea = cfg_map["sea_abrupt4"]
    base_cfg_sine = cfg_map["sine_abrupt4"]

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

    # group list
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

    perm_alphas = parse_float_list(str(args.perm_alphas))
    perm_pre_ns = parse_int_list(str(args.perm_pre_ns))
    perm_post_ns = parse_int_list(str(args.perm_post_ns))
    delta_ks = parse_int_list(str(args.delta_ks))
    perm_n_perm = int(args.perm_n_perm)

    perm_stats = [s.strip() for s in str(getattr(args, "perm_stats", "") or "").split(",") if s.strip()]
    allowed_stats = {"fused_score", "delta_fused_score", "vote_score"}
    if not perm_stats:
        raise ValueError("--perm_stats 不能为空")
    for s in perm_stats:
        if s not in allowed_stats:
            raise ValueError(f"--perm_stats 包含不支持的 stat：{s}（允许：{sorted(allowed_stats)}）")

    for stat in perm_stats:
        for alpha in perm_alphas:
            for pre_n in perm_pre_ns:
                for post_n in perm_post_ns:
                    if stat == "delta_fused_score":
                        for dk in delta_ks:
                            name = f"P_perm_{stat}_a{alpha:g}_pre{pre_n}_post{post_n}_dk{dk}_n5"
                            tw = dict(weights_base)
                            tw["__confirm_rule"] = "perm_test"
                            tw["__perm_stat"] = stat
                            tw["__perm_alpha"] = float(alpha)
                            tw["__perm_pre_n"] = float(pre_n)
                            tw["__perm_post_n"] = float(post_n)
                            tw["__perm_n_perm"] = float(perm_n_perm)
                            tw["__perm_delta_k"] = float(dk)
                            tw["__perm_min_effect"] = 0.0
                            tw["__perm_rng_seed"] = 0.0
                            groups.append(
                                {
                                    "group": name,
                                    "confirm_rule": "perm_test",
                                    "perm_stat": stat,
                                    "delta_k": int(dk),
                                    "perm_alpha": float(alpha),
                                    "perm_pre_n": int(pre_n),
                                    "perm_post_n": int(post_n),
                                    "perm_n_perm": int(perm_n_perm),
                                    "trigger_weights": tw,
                                }
                            )
                    else:
                        name = f"P_perm_{stat}_a{alpha:g}_pre{pre_n}_post{post_n}_n5"
                        tw = dict(weights_base)
                        tw["__confirm_rule"] = "perm_test"
                        tw["__perm_stat"] = stat
                        tw["__perm_alpha"] = float(alpha)
                        tw["__perm_pre_n"] = float(pre_n)
                        tw["__perm_post_n"] = float(post_n)
                        tw["__perm_n_perm"] = float(perm_n_perm)
                        tw["__perm_min_effect"] = 0.0
                        tw["__perm_rng_seed"] = 0.0
                        groups.append(
                            {
                                "group": name,
                                "confirm_rule": "perm_test",
                                "perm_stat": stat,
                                "delta_k": "N/A",
                                "perm_alpha": float(alpha),
                                "perm_pre_n": int(pre_n),
                                "perm_post_n": int(post_n),
                                "perm_n_perm": int(perm_n_perm),
                                "trigger_weights": tw,
                            }
                        )

    num_shards = int(args.num_shards)
    shard_idx = int(args.shard_idx)
    if num_shards < 1:
        raise ValueError("--num_shards 必须 >= 1")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"--shard_idx 越界：{shard_idx}（期望 0..{num_shards - 1}）")
    if bool(args.run_full) and num_shards > 1:
        raise ValueError("--run_full 与分片并行不兼容（topK 选择需要全量 groups）；请先分片跑 quick，merge 后再单独跑 full。")

    groups = sorted(groups, key=lambda g: str(g.get("group") or ""))
    if num_shards > 1:
        groups = [g for i, g in enumerate(groups) if (i % num_shards) == shard_idx]
        print(f"[shard] num_shards={num_shards} shard_idx={shard_idx} groups={len(groups)}")

    # run quick
    rows_quick: List[Dict[str, Any]] = []

    def run_groups(groups_to_run: List[Dict[str, Any]], seeds: List[int], *, tag: str) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for g in groups_to_run:
            exp_run = create_experiment_run(
                experiment_name="trackAL_perm_confirm_sweep",
                results_root=Path(args.results_root),
                logs_root=logs_root,
                run_name=str(g["group"]),
            )
            by_ds: Dict[str, List[Dict[str, Any]]] = {}
            run_index: Dict[str, Any] = {}
            for ds_alias, base_name, base_cfg, stream_kwargs, ds_kind in datasets:
                by_ds[ds_alias] = []
                run_index[ds_alias] = {"base_dataset_name": base_name, "dataset_kind": ds_kind, "stream_kwargs": stream_kwargs, "runs": []}
                for seed in seeds:
                    log_path = ensure_log(
                        exp_run,
                        dataset_name=ds_alias,
                        seed=seed,
                        base_cfg=base_cfg,
                        monitor_preset=str(args.monitor_preset),
                        signal_set=args.signal_set,
                        confirm_theta=float(args.confirm_theta),
                        confirm_window=int(args.confirm_window),
                        confirm_cooldown=int(args.confirm_cooldown),
                        trigger_weights=dict(g["trigger_weights"]),
                        stream_kwargs=stream_kwargs,
                        device=str(args.device),
                    )
                    summ = read_run_summary(log_path)
                    signal_set_val = summ.get("signal_set") or (args.signal_set or "")
                    horizon = int(summ.get("horizon") or n_samples)
                    confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                    confirmed = merge_events(confirmed_raw, min_sep)

                    # RUN_INDEX 约束：避免同一 run_id 跨 dataset 复用（审计脚本将 fail-fast）
                    run_id = f"{exp_run.run_id}__{ds_alias}"

                    rec: Dict[str, Any] = {
                        "seed": int(seed),
                        "run_id": str(run_id),
                        "log_path": str(log_path),
                        "horizon": int(horizon),
                        "signal_set": signal_set_val,
                        "acc_final": _safe_float(summ.get("acc_final")),
                        f"acc_min@{warmup}": acc_min_after_warmup(summ, warmup),
                        "candidate_count_total": int(summ.get("candidate_count_total") or 0),
                        "confirmed_count_total": int(summ.get("confirmed_count_total") or len(confirmed_raw)),
                    }

                    if ds_kind == "nodrift":
                        cc = int(summ.get("confirmed_count_total") or len(confirmed_raw))
                        rec["confirm_rate_per_10k"] = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                        rec["MTFA_win"] = mtfa_from_false_alarms(confirmed, horizon)
                        rec["miss_tol500"] = None
                        rec["conf_P90"] = None
                    else:
                        delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=tol)
                        rec["miss_tol500"] = (float(sum(miss_flags)) / float(len(miss_flags))) if miss_flags else None
                        rec["conf_P90"] = percentile(delays, 0.90) if delays else None
                        win_m = compute_detection_metrics(gt_starts, confirmed_raw, horizon) if gt_starts else {"MTFA": None}
                        rec["MTFA_win"] = _safe_float(win_m.get("MTFA"))
                        rec["confirm_rate_per_10k"] = None

                    by_ds[ds_alias].append(rec)
                    run_index[ds_alias]["runs"].append({"seed": int(seed), "run_id": str(run_id), "log_path": str(log_path)})

            # aggregate rows per dataset
            for ds_alias, _base_name, _base_cfg, _stream_kwargs, ds_kind in datasets:
                rs = by_ds.get(ds_alias) or []

                def col(name: str) -> List[Optional[float]]:
                    return [(_safe_float(r.get(name)) if name not in {f"acc_min@{warmup}"} else _safe_float(r.get(name))) for r in rs]

                out_rows.append(
                    {
                        "track": "AL",
                        "phase": tag,
                        "dataset": ds_alias,
                        "dataset_kind": ds_kind,
                        "base_dataset_name": run_index[ds_alias]["base_dataset_name"],
                        "unit": "sample_idx",
                        "n_runs": int(len(rs)),
                        "group": str(g["group"]),
                        "monitor_preset": str(args.monitor_preset),
                        "signal_set": rs[0].get("signal_set") if rs else (args.signal_set or ""),
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
                        "horizon_mean": mean([_safe_float(r.get("horizon")) for r in rs]),
                        "acc_final_mean": mean([_safe_float(r.get("acc_final")) for r in rs]),
                        "acc_final_std": std([_safe_float(r.get("acc_final")) for r in rs]),
                        f"acc_min@{warmup}_mean": mean([_safe_float(r.get(f"acc_min@{warmup}")) for r in rs]),
                        f"acc_min@{warmup}_std": std([_safe_float(r.get(f"acc_min@{warmup}")) for r in rs]),
                        "miss_tol500_mean": mean([_safe_float(r.get("miss_tol500")) for r in rs]),
                        "conf_P90_mean": mean([_safe_float(r.get("conf_P90")) for r in rs]),
                        "MTFA_win_mean": mean([_safe_float(r.get("MTFA_win")) for r in rs]),
                        "MTFA_win_std": std([_safe_float(r.get("MTFA_win")) for r in rs]),
                        "confirm_rate_per_10k_mean": mean([_safe_float(r.get("confirm_rate_per_10k")) for r in rs]),
                        "confirm_rate_per_10k_std": std([_safe_float(r.get("confirm_rate_per_10k")) for r in rs]),
                        "candidate_count_mean": mean([_safe_float(r.get("candidate_count_total")) for r in rs]),
                        "confirmed_count_mean": mean([_safe_float(r.get("confirmed_count_total")) for r in rs]),
                        "run_index_json": json.dumps(run_index, ensure_ascii=False),
                    }
                )
        return out_rows

    rows_quick = run_groups(groups, seeds_quick, tag="quick")

    # pick topK for full (based on quick summary)
    full_groups: List[Dict[str, Any]] = []
    if bool(args.run_full):
        # build per-group aggregates needed for selection
        by_group: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in rows_quick:
            gname = str(r.get("group") or "")
            ds = str(r.get("dataset") or "")
            by_group.setdefault(gname, {})[ds] = r

        def safe(v: Any) -> Optional[float]:
            return _safe_float(v)

        eligible: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
        for gname, dsmap in by_group.items():
            sea = dsmap.get("sea_abrupt4")
            sine = dsmap.get("sine_abrupt4")
            if not sea or not sine:
                continue
            sea_miss = safe(sea.get("miss_tol500_mean"))
            sine_miss = safe(sine.get("miss_tol500_mean"))
            sea_conf = safe(sea.get("conf_P90_mean"))
            sine_conf = safe(sine.get("conf_P90_mean"))
            if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
                continue
            if not (sea_miss <= 1e-12 and sine_miss <= 1e-12 and sea_conf < 500 and sine_conf < 500):
                continue
            eligible.append((gname, dsmap))

        # acc constraint: drift_acc_final >= best-0.01
        def drift_acc(dsmap: Dict[str, Dict[str, Any]]) -> float:
            sea = safe(dsmap.get("sea_abrupt4", {}).get("acc_final_mean"))
            sine = safe(dsmap.get("sine_abrupt4", {}).get("acc_final_mean"))
            vals = [v for v in [sea, sine] if v is not None]
            return float(sum(vals) / len(vals)) if vals else float("-inf")

        best_acc = max([drift_acc(m) for _, m in eligible] or [float("-inf")])
        eligible2 = [(g, m) for g, m in eligible if drift_acc(m) >= best_acc - 0.01]
        if not eligible2:
            eligible2 = eligible

        def nd_rate(dsmap: Dict[str, Dict[str, Any]]) -> float:
            a = safe(dsmap.get("sea_nodrift", {}).get("confirm_rate_per_10k_mean"))
            b = safe(dsmap.get("sine_nodrift", {}).get("confirm_rate_per_10k_mean"))
            vals = [v for v in [a, b] if v is not None]
            return float(sum(vals) / len(vals)) if vals else float("inf")

        def nd_mtfa(dsmap: Dict[str, Dict[str, Any]]) -> float:
            a = safe(dsmap.get("sea_nodrift", {}).get("MTFA_win_mean"))
            b = safe(dsmap.get("sine_nodrift", {}).get("MTFA_win_mean"))
            vals = [v for v in [a, b] if v is not None]
            return float(sum(vals) / len(vals)) if vals else float("-inf")

        eligible2.sort(key=lambda x: (nd_rate(x[1]), -nd_mtfa(x[1]), x[0]))
        picked = [g for g, _ in eligible2[: max(1, int(args.full_topk))]]

        # map back to group config and create full variants
        gmap = {str(g["group"]): g for g in groups}
        for gname in picked:
            g = gmap.get(gname)
            if not g:
                continue
            g2 = dict(g)
            g2["group"] = str(gname).replace("_n5", "_n20")
            full_groups.append(g2)

    rows_full: List[Dict[str, Any]] = []
    if full_groups:
        rows_full = run_groups(full_groups, seeds_full, tag="full")

    # write csv
    all_rows = rows_quick + rows_full
    if not all_rows:
        print("[warn] no rows")
        return 0

    # stable field order
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in all_rows:
        for k in r.keys():
            if k in seen:
                continue
            fieldnames.append(k)
            seen.add(k)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"[done] wrote {out_csv} rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
