#!/usr/bin/env python
"""
NEXT_STAGE V5 - Track N: Gating 规则化（INSECTS）

对比组：
- v2（no gate）
- v2_gate_m{1,3,5}（confirmed_streak gating）

触发策略：优先使用 Track M 选出的默认（若 Track M CSV 不存在或为空则回退 V4 默认）。

输出：scripts/TRACKN_GATING_RULE.csv（按 run 粒度记录）
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
    p = argparse.ArgumentParser(description="Track N: Gating rule on INSECTS")
    p.add_argument("--dataset", type=str, default="INSECTS_abrupt_balanced")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--model_variant", type=str, default="ts_drift_adapt_severity")
    p.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--trackm_csv", type=str, default="scripts/TRACKM_LATENCY_SWEEP.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKN_GATING_RULE.csv")
    # two_stage fallback（V4 默认）
    p.add_argument("--fallback_confirm_theta", type=float, default=0.3)
    p.add_argument("--fallback_confirm_window", type=int, default=1)
    # gating
    p.add_argument("--gate_streaks", nargs="+", type=int, default=[1, 3, 5])
    # recovery metrics
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


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def load_insects_positions(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    positions = obj.get("positions") or []
    return [int(x) for x in positions]


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


def pick_trackm_default(trackm_csv: Path) -> Tuple[str, float, int, str]:
    """
    按 Track M 选择规则选出默认：
      - 约束：acc_final_mean ≥ best - 0.01
      - 主目标：最小化 miss_tol500_mean
      - 次目标：最小化 delay_confirmed_P90（用于强调实时性）
      - 再次：最大化 acc_min_mean
      - 再次：最大化 MTFA_win_mean（再看 MTR_win_mean）
    """
    if not trackm_csv.exists():
        return "", 0.3, 1, "fallback：trackm_csv 不存在"
    rows = list(csv.DictReader(trackm_csv.open("r", encoding="utf-8")))
    if not rows:
        return "", 0.3, 1, "fallback：trackm_csv 为空"

    # 仅考虑 sea_abrupt4 的 two_stage 配置
    filtered = [r for r in rows if (r.get("dataset") == "sea_abrupt4" and r.get("trigger_mode") == "two_stage")]
    if not filtered:
        return "", 0.3, 1, "fallback：trackm_csv 无 sea_abrupt4 two_stage 记录"

    # 以 config_tag 聚合
    by_cfg: Dict[str, List[Dict[str, str]]] = {}
    for r in filtered:
        tag = str(r.get("config_tag") or "")
        if not tag:
            continue
        by_cfg.setdefault(tag, []).append(r)

    agg: Dict[str, Dict[str, Any]] = {}
    for tag, rs in by_cfg.items():
        # 去重：按 (run_id, seed) 取 acc/窗口指标
        seen_runs: set[Tuple[str, str]] = set()
        acc_final_list: List[Optional[float]] = []
        acc_min_list: List[Optional[float]] = []
        mtfa_win_list: List[Optional[float]] = []
        mtr_win_list: List[Optional[float]] = []
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
            mtr_win_list.append(_safe_float(r.get("MTR_win")))

        # miss 与 delay：按 drift 粒度统计（直接用全部 drift 行）
        miss_list = [_safe_float(r.get("miss_tol500")) for r in rs]
        miss_vals = [float(x) for x in miss_list if x is not None]
        delay_list = [_safe_float(r.get("delay_confirmed")) for r in rs]
        delay_vals = [float(x) for x in delay_list if x is not None]
        miss_mean = float(statistics.mean(miss_vals)) if miss_vals else 1.0
        delay_p90 = percentile(delay_vals, 0.9) if delay_vals else float("inf")

        acc_final_mu, _ = mean_std(acc_final_list)
        acc_min_mu, _ = mean_std(acc_min_list)
        mtfa_mu, _ = mean_std(mtfa_win_list)
        mtr_mu, _ = mean_std(mtr_win_list)

        # 取代表性的 preset/theta/window（同 tag 内应一致）
        sample = rs[0]
        agg[tag] = {
            "monitor_preset": str(sample.get("monitor_preset") or ""),
            "confirm_theta": float(sample.get("confirm_theta") or 0.3),
            "confirm_window": int(float(sample.get("confirm_window") or 1)),
            "acc_final_mean": float(acc_final_mu) if acc_final_mu is not None else float("-inf"),
            "acc_min_mean": float(acc_min_mu) if acc_min_mu is not None else float("-inf"),
            "miss_mean": float(miss_mean),
            "delay_p90": float(delay_p90) if delay_p90 is not None else float("inf"),
            "mtfa_win_mean": float(mtfa_mu) if mtfa_mu is not None else float("-inf"),
            "mtr_win_mean": float(mtr_mu) if mtr_mu is not None else float("-inf"),
        }

    if not agg:
        return "", 0.3, 1, "fallback：trackm_csv 无可用聚合"

    best_acc = max(v["acc_final_mean"] for v in agg.values())
    eligible = [(tag, v) for tag, v in agg.items() if v["acc_final_mean"] >= best_acc - 0.01]
    if not eligible:
        eligible = list(agg.items())

    eligible.sort(
        key=lambda kv: (
            kv[1]["miss_mean"],
            kv[1]["delay_p90"],
            -kv[1]["acc_min_mean"],
            -kv[1]["mtfa_win_mean"],
            -kv[1]["mtr_win_mean"],
            kv[0],
        )
    )
    tag, best = eligible[0]
    reason = (
        f"from TrackM(tag={tag}): acc_final_mean={best['acc_final_mean']:.4f}, "
        f"miss_mean={best['miss_mean']:.3f}, delay_p90={best['delay_p90']:.1f}"
    )
    return best["monitor_preset"], float(best["confirm_theta"]), int(best["confirm_window"]), reason


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


def main() -> int:
    args = parse_args()
    weights = parse_weights(args.weights)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    insects_positions = load_insects_positions(Path(args.insects_meta))

    trackm_preset, theta, window, picked_reason = pick_trackm_default(Path(args.trackm_csv))
    monitor_preset = trackm_preset or str(args.monitor_preset)
    if trackm_preset:
        pick_msg = f"picked TrackM preset: {picked_reason}"
    else:
        pick_msg = f"fallback preset: theta={theta}, window={window}"

    theta = float(theta or args.fallback_confirm_theta)
    window = int(window or args.fallback_confirm_window)

    groups: List[Tuple[str, str, int]] = []
    groups.append(("v2", "none", 1))
    for m in [int(x) for x in args.gate_streaks]:
        groups.append((f"v2_gate_m{m}", "confirmed_streak", int(m)))

    records: List[Dict[str, Any]] = []
    for group, gate_mode, m in groups:
        exp_run = create_experiment_run(
            experiment_name="trackN_gating_rule",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{group}_theta{theta:.2f}_w{window}",
        )
        for seed in list(args.seeds):
            cfg_map = build_cfg_map(seed)
            if str(args.dataset).lower() not in cfg_map:
                raise ValueError(f"未知 dataset: {args.dataset}")
            base_cfg = cfg_map[str(args.dataset).lower()]
            log_path = ensure_log(
                exp_run,
                str(args.dataset),
                seed,
                base_cfg,
                model_variant=str(args.model_variant),
                monitor_preset=monitor_preset,
                trigger_threshold=float(theta),
                trigger_weights=weights,
                confirm_window=int(window),
                use_severity_v2=True,
                severity_gate=str(gate_mode),
                severity_gate_min_streak=int(m),
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
            ph_params = summ.get("monitor_ph_params") or {}
            ph_error = ph_params.get("error_rate") if isinstance(ph_params, dict) else {}
            ph_div = ph_params.get("divergence") if isinstance(ph_params, dict) else {}
            records.append(
                {
                    "track": "N",
                    "unit": "sample_idx",
                    "dataset": str(args.dataset),
                    "seed": seed,
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "group": group,
                    "model_variant": str(args.model_variant),
                    "monitor_preset": monitor_preset,
                    "monitor_preset_base": summ.get("monitor_preset_base", ""),
                    "picked_from_trackm": int(bool(trackm_preset)),
                    "picked_reason": picked_reason,
                    "trigger_mode": "two_stage",
                    "confirm_theta": float(theta),
                    "confirm_window": int(window),
                    "weights": args.weights,
                    "use_severity_v2": 1,
                    "severity_gate": str(gate_mode),
                    "severity_gate_min_streak": int(m),
                    "ph_error_threshold": _safe_float(ph_error.get("threshold") if isinstance(ph_error, dict) else None),
                    "ph_error_min_instances": _safe_int(ph_error.get("min_instances") if isinstance(ph_error, dict) else None),
                    "ph_divergence_threshold": _safe_float(ph_div.get("threshold") if isinstance(ph_div, dict) else None),
                    "ph_divergence_min_instances": _safe_int(ph_div.get("min_instances") if isinstance(ph_div, dict) else None),
                    "log_path": str(log_path),
                    "acc_final": summ.get("acc_final"),
                    "mean_acc": summ.get("mean_acc"),
                    "acc_min": summ.get("acc_min"),
                    "recovery_W": int(args.recovery_W),
                    "post_mean_acc": rec.get("post_mean_acc"),
                    "post_min_acc": rec.get("post_min_acc"),
                    "recovery_time_to_pre90": rec.get("recovery_time_to_pre90"),
                    "note": pick_msg,
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

