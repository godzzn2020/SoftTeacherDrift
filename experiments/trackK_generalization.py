#!/usr/bin/env python
"""
Track K：泛化验证（至少 INSECTS_abrupt_balanced）。

- 对比模式：or / weighted / two_stage（two_stage 使用 Track J 推荐组合，若缺失则回退默认）
- 输出：scripts/TRACKK_GENERALIZATION.csv
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

from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track K：generalization (INSECTS)")
    p.add_argument("--dataset", type=str, default="INSECTS_abrupt_balanced")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--weighted_theta", type=float, default=0.5)
    p.add_argument("--trackj_csv", type=str, default="scripts/TRACKJ_TWOSTAGE_SWEEP.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKK_GENERALIZATION.csv")
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


def load_insects_positions(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    positions = obj.get("positions") or []
    return [int(x) for x in positions]


def read_log(csv_path: Path) -> Dict[str, object]:
    summary_path = csv_path.with_suffix(".summary.json")
    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            series_raw = obj.get("acc_series") or []
            series: List[Tuple[int, float]] = []
            for item in series_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                series.append((int(item[0]), float(item[1])))
            return {
                "acc_final": obj.get("acc_final"),
                "mean_acc": obj.get("mean_acc"),
                "acc_min": obj.get("acc_min"),
                "candidates": [int(x) for x in (obj.get("candidate_sample_idxs") or [])],
                "confirmed": [int(x) for x in (obj.get("confirmed_sample_idxs") or [])],
                "horizon": int(obj.get("horizon") or 0),
                "acc_series": series,
            }
        except Exception:
            pass
    accs: List[float] = []
    xs: List[int] = []
    candidates: List[int] = []
    confirmed: List[int] = []
    series: List[Tuple[int, float]] = []
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
                series.append((int(x), float(acc)))
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
        "acc_series": series,
    }


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


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
    xs = [int(x) for x, _ in acc_series]
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
        # 在 post 窗口内用“点数”近似滚动均值，找到首次达到阈值的点
        post_points = [(x, a) for (x, a) in acc_map if g <= x <= g + W]
        found: Optional[float] = None
        for idx in range(len(post_points)):
            start = max(0, idx - max(1, int(roll_points)) + 1)
            window_vals = [a for _, a in post_points[start : idx + 1]]
            if window_vals and (sum(window_vals) / len(window_vals)) >= thr:
                found = float(post_points[idx][0] - g)
                break
        rec_times.append(found)

    post_mean_mu, post_mean_sd = mean_std(post_means)
    post_min_mu, post_min_sd = mean_std(post_mins)
    rec_mu, rec_sd = mean_std(rec_times)
    return {
        "post_mean_acc": post_mean_mu,
        "post_mean_acc_std": post_mean_sd,
        "post_min_acc": post_min_mu,
        "post_min_acc_std": post_min_sd,
        "recovery_time_to_pre90": rec_mu,
        "recovery_time_to_pre90_std": rec_sd,
    }


def recommend_twostage_from_trackj(trackj_csv: Path) -> Tuple[float, int, str]:
    if not trackj_csv.exists():
        return 0.4, 2, "fallback：trackj_csv 不存在"
    rows: List[Dict[str, str]] = []
    with trackj_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.4, 2, "fallback：trackj_csv 为空"

    # 聚合到 dataset × (theta, window)
    items: Dict[Tuple[str, float, int], Dict[str, List[float]]] = {}
    for r in rows:
        ds = r.get("dataset") or ""
        theta = _safe_float(r.get("confirm_theta"))
        window = _safe_int(r.get("confirm_window"))
        acc = _safe_float(r.get("acc_final"))
        acc_min = _safe_float(r.get("acc_min"))
        miss = _safe_float(r.get("miss_rate_tol500"))
        if not ds or theta is None or window is None:
            continue
        key = (ds, float(theta), int(window))
        if key not in items:
            items[key] = {"acc": [], "acc_min": [], "miss": []}
        if acc is not None:
            items[key]["acc"].append(float(acc))
        if acc_min is not None:
            items[key]["acc_min"].append(float(acc_min))
        if miss is not None:
            items[key]["miss"].append(float(miss))

    # 先对每个 dataset 得到 best acc
    best_acc: Dict[str, float] = {}
    for (ds, theta, window), v in items.items():
        if not v["acc"]:
            continue
        mu = float(statistics.mean(v["acc"]))
        best_acc[ds] = max(best_acc.get(ds, float("-inf")), mu)

    # 选“全局默认”：同时满足两个数据集 acc 不差（<=0.01），再最小化平均 miss，再最大化平均 acc_min
    datasets = sorted(best_acc.keys())
    if not datasets:
        return 0.4, 2, "fallback：无法从 trackj_csv 得到有效 acc"
    candidates: List[Tuple[float, int, float, float]] = []
    for theta in sorted({k[1] for k in items.keys()}):
        for window in sorted({k[2] for k in items.keys()}):
            ok = True
            miss_list: List[float] = []
            accmin_list: List[float] = []
            for ds in datasets:
                v = items.get((ds, theta, window))
                if not v or not v["acc"]:
                    ok = False
                    break
                mu = float(statistics.mean(v["acc"]))
                if mu < best_acc[ds] - 0.01:
                    ok = False
                    break
                miss_list.append(float(statistics.mean(v["miss"])) if v["miss"] else 1.0)
                accmin_list.append(float(statistics.mean(v["acc_min"])) if v["acc_min"] else float("-inf"))
            if not ok:
                continue
            candidates.append((theta, window, float(statistics.mean(miss_list)), float(statistics.mean(accmin_list))))
    if not candidates:
        return 0.4, 2, "fallback：无满足 acc 约束的全局候选"
    candidates.sort(key=lambda x: (x[2], -x[3], x[0], x[1]))
    theta, window, miss_mu, accmin_mu = candidates[0]
    return float(theta), int(window), f"from TrackJ：miss_mean={miss_mu:.3f}, acc_min_mean={accmin_mu:.3f}"


def find_insects_cfg(seed: int) -> ExperimentConfig:
    for cfg in _default_experiment_configs(seed):
        if cfg.dataset_name == "INSECTS_abrupt_balanced":
            return cfg
    raise ValueError("未在 _default_experiment_configs 中找到 INSECTS_abrupt_balanced")


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
    weights = parse_weights(args.weights)
    insects_positions = load_insects_positions(Path(args.insects_meta))
    trackj_theta, trackj_window, trackj_reason = recommend_twostage_from_trackj(Path(args.trackj_csv))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)

    variants = [
        ("or", dict(trigger_mode="or", trigger_threshold=0.5, confirm_window=200)),
        ("weighted", dict(trigger_mode="weighted", trigger_threshold=float(args.weighted_theta), confirm_window=200)),
        ("two_stage", dict(trigger_mode="two_stage", trigger_threshold=float(trackj_theta), confirm_window=int(trackj_window))),
    ]

    records: List[Dict[str, object]] = []
    for mode, cfgm in variants:
        exp_run = create_experiment_run(
            experiment_name="trackK_generalization",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{mode}",
        )
        for seed in list(args.seeds):
            base_cfg = find_insects_cfg(seed)
            log_path = ensure_log(
                exp_run,
                args.dataset,
                seed,
                base_cfg,
                model_variant="ts_drift_adapt",
                monitor_preset=args.monitor_preset,
                trigger_mode=str(cfgm["trigger_mode"]),
                trigger_threshold=float(cfgm["trigger_threshold"]),
                trigger_weights=weights,
                confirm_window=int(cfgm["confirm_window"]),
                device=args.device,
            )
            stats = read_log(log_path)
            horizon = int(stats["horizon"])
            confirmed_raw: List[int] = list(stats["confirmed"])  # type: ignore[assignment]
            confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))
            win_m = compute_detection_metrics(insects_positions, confirmed_raw, horizon) if horizon > 0 else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
            tol_m = compute_metrics_tolerance(insects_positions, confirmed_merged, horizon, int(args.tol500))
            rec = compute_recovery_metrics(
                stats["acc_series"],  # type: ignore[arg-type]
                insects_positions,
                W=int(args.recovery_W),
                pre_window=int(args.pre_window),
                roll_points=int(args.roll_points),
            )
            records.append(
                {
                    "track": "K",
                    "unit": "sample_idx",
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "dataset": args.dataset,
                    "seed": seed,
                    "model_variant": "ts_drift_adapt",
                    "monitor_preset": args.monitor_preset,
                    "mode": mode,
                    "trigger_mode": str(cfgm["trigger_mode"]),
                    "trigger_threshold": float(cfgm["trigger_threshold"]),
                    "confirm_window": int(cfgm["confirm_window"]),
                    "weights": args.weights,
                    "two_stage_choice": f"theta={trackj_theta},window={trackj_window}",
                    "two_stage_choice_reason": trackj_reason,
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
                    "recovery_W": int(args.recovery_W),
                    "post_mean_acc": rec.get("post_mean_acc"),
                    "post_mean_acc_std": rec.get("post_mean_acc_std"),
                    "post_min_acc": rec.get("post_min_acc"),
                    "post_min_acc_std": rec.get("post_min_acc_std"),
                    "recovery_time_to_pre90": rec.get("recovery_time_to_pre90"),
                    "recovery_time_to_pre90_std": rec.get("recovery_time_to_pre90_std"),
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
