#!/usr/bin/env python
"""汇总 Track D/E（Severity v2 + 监控融合）并生成 V3 报告与 CSV（尽量 stdlib-only）。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from evaluation.drift_metrics import compute_detection_metrics
except Exception:
    compute_detection_metrics = None  # type: ignore[assignment]


TRACK_RUN_TOKENS = {
    # Track D
    "D_insects_base": ("run_real_adaptive", "trackD_insects_base"),
    "D_insects_v1": ("run_real_adaptive", "trackD_insects_v1"),
    "D_insects_v2": ("run_real_adaptive", "trackD_insects_v2_unc_d095_f5"),
    "D_sea_base": ("stage1_multi_seed", "trackD_sea_base"),
    "D_sea_v1": ("stage1_multi_seed", "trackD_sea_v1"),
    "D_sea_v2": ("stage1_multi_seed", "trackD_sea_v2_unc_d095_f5"),
    # Track E
    "E_or": ("stage1_multi_seed", "trackE_or"),
    "E_k2": ("stage1_multi_seed", "trackE_k2"),
    "E_weighted": ("stage1_multi_seed", "trackE_weighted_w532_t05"),
}


@dataclass(frozen=True)
class RunRef:
    experiment_name: str
    run_name_token: str
    run_id: str


@dataclass(frozen=True)
class LogRef:
    experiment_name: str
    run_id: str
    dataset: str
    model_variant: str
    seed: int
    log_path: Path


@dataclass
class RunMetrics:
    # identity
    experiment_name: str
    run_name_token: str
    run_id: str
    dataset: str
    model_variant: str
    seed: int
    # config snapshot (from log first row if available)
    monitor_preset: str = ""
    trigger_mode: str = ""
    trigger_k: Optional[int] = None
    trigger_threshold: Optional[float] = None
    trigger_weights: str = ""
    severity_scheduler_scale: Optional[float] = None
    use_severity_v2: Optional[int] = None
    entropy_mode: str = ""
    decay: Optional[float] = None
    freeze_baseline_steps: Optional[int] = None
    # classification
    acc_final: Optional[float] = None
    mean_acc: Optional[float] = None
    acc_min: Optional[float] = None
    # detection signals
    drift_flag_count: Optional[int] = None
    detections: List[int] = None  # type: ignore[assignment]
    horizon_T: Optional[int] = None
    # drift metrics (window-based)
    mdr_win: Optional[float] = None
    mtd_win: Optional[float] = None
    mtfa_win: Optional[float] = None
    mtr_win: Optional[float] = None
    # drift metrics (tolerance-based)
    mdr_tol: Optional[float] = None
    mtd_tol: Optional[float] = None
    mtfa_tol: Optional[float] = None
    mtr_tol: Optional[float] = None
    # recovery metrics (pooled per-drift later; here store run-level mean across drifts)
    recovery_window: int = 1000
    post_mean_acc_mean: Optional[float] = None
    post_min_acc_mean: Optional[float] = None
    post_mean_acc_values: List[float] = None  # type: ignore[assignment]
    post_min_acc_values: List[float] = None  # type: ignore[assignment]
    n_gt_drifts: Optional[int] = None
    n_recovery_drifts: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Track D/E 并生成 V3 报告与 CSV")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="scripts")
    parser.add_argument("--report_name", type=str, default="NEXT_ROUND_V3_REPORT.md")
    parser.add_argument("--run_index_name", type=str, default="NEXT_ROUND_V3_RUN_INDEX.csv")
    parser.add_argument("--metrics_table_name", type=str, default="NEXT_ROUND_V3_METRICS_TABLE.csv")
    parser.add_argument("--recovery_window", type=int, default=1000)
    parser.add_argument("--match_tolerance", type=int, default=500)
    parser.add_argument("--min_separation", type=int, default=200)
    parser.add_argument("--insects_meta", type=str, default="datasets/real/INSECTS_abrupt_balanced.json")
    parser.add_argument("--synthetic_root", type=str, default="data/synthetic")
    return parser.parse_args()


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return None
        return float(text)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return None
        return int(float(text))
    except Exception:
        return None


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]  # type: ignore[arg-type]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]  # type: ignore[arg-type]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "N/A"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{nd}f}"


def _fmt_ms(v: Optional[float], nd: int = 1) -> str:
    return _fmt(v, nd)


def iter_log_csv_paths(logs_root: Path, experiment_name: str) -> Iterable[Path]:
    base = logs_root / experiment_name
    if not base.exists():
        return []
    for root, _dirs, files in os.walk(base):
        for name in files:
            if name.endswith(".csv"):
                yield Path(root) / name


def find_run_ids_by_suffix(logs_root: Path, experiment_name: str, run_name_token: str) -> List[str]:
    run_ids = set()
    for csv_path in iter_log_csv_paths(logs_root, experiment_name):
        run_id = csv_path.parent.name
        seed_dir = csv_path.parent.parent.name
        if not seed_dir.startswith("seed"):
            continue
        if run_id.endswith(f"_{run_name_token}"):
            run_ids.add(run_id)
    return sorted(run_ids)


def choose_latest_run_id(run_ids: Sequence[str]) -> Optional[str]:
    if not run_ids:
        return None
    return sorted(run_ids)[-1]


def collect_logs_for_run_id(logs_root: Path, experiment_name: str, run_id: str) -> List[LogRef]:
    refs: List[LogRef] = []
    for csv_path in iter_log_csv_paths(logs_root, experiment_name):
        if csv_path.parent.name != run_id:
            continue
        seed_dir = csv_path.parent.parent.name
        model_dir = csv_path.parent.parent.parent.name
        dataset_dir = csv_path.parent.parent.parent.parent.name
        seed = _safe_int(seed_dir.replace("seed", ""))
        if seed is None:
            continue
        refs.append(
            LogRef(
                experiment_name=experiment_name,
                run_id=run_id,
                dataset=dataset_dir,
                model_variant=model_dir,
                seed=seed,
                log_path=csv_path,
            )
        )
    refs.sort(key=lambda r: (r.dataset, r.model_variant, r.seed, str(r.log_path)))
    return refs


def load_synth_meta(synth_root: Path, dataset: str, seed: int) -> Tuple[List[int], int]:
    meta_path = synth_root / dataset / f"{dataset}__seed{seed}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    drifts = [int(d["start"]) for d in meta.get("drifts", [])]
    T = int(meta.get("n_samples", 0) or 0)
    return drifts, T


def load_insects_meta(path: Path) -> Tuple[List[int], int]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    drifts = [int(p) for p in meta.get("positions", [])]
    T = int(meta.get("n_samples", 0) or 0)
    return drifts, T


def infer_x_col(fieldnames: Sequence[str]) -> str:
    if "sample_idx" in fieldnames:
        return "sample_idx"
    if "seen_samples" in fieldnames:
        return "seen_samples"
    return "step"


def merge_detections(events: Sequence[int], min_sep: int) -> List[int]:
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
    drifts = sorted(int(d) for d in gt_drifts)
    dets = sorted(int(d) for d in detections)
    if not drifts or horizon <= 0:
        return {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
    matched: set[int] = set()
    delays: List[int] = []
    missed = 0
    for drift in drifts:
        match_idx: Optional[int] = None
        for idx, det in enumerate(dets):
            if idx in matched:
                continue
            if det < drift:
                continue
            if det <= drift + tolerance:
                match_idx = idx
                break
            if det > drift + tolerance:
                break
        if match_idx is None:
            missed += 1
        else:
            matched.add(match_idx)
            delays.append(max(0, dets[match_idx] - drift))
    false_dets = [dets[i] for i in range(len(dets)) if i not in matched]
    mdr = missed / len(drifts) if drifts else None
    mtd = (sum(delays) / len(delays)) if delays else None
    if len(false_dets) >= 2:
        gaps = [b - a for a, b in zip(false_dets[:-1], false_dets[1:])]
        mtfa = sum(gaps) / len(gaps)
    elif len(false_dets) == 1:
        mtfa = float(horizon - false_dets[0]) if horizon > 0 else None
    else:
        mtfa = None
    if mdr is None or mtd is None or mtfa is None or mdr >= 1.0 or mtd <= 0:
        mtr = None
    else:
        mtr = mtfa / (mtd * (1 - mdr))
    return {"MDR": float(mdr) if mdr is not None else None, "MTD": float(mtd) if mtd is not None else None, "MTFA": float(mtfa) if mtfa is not None else None, "MTR": float(mtr) if mtr is not None else None}


def compute_recovery_metrics(
    rows: List[Dict[str, str]],
    gt_drifts: Sequence[int],
    x_col: str,
    window: int,
) -> Tuple[List[float], List[float]]:
    accs: List[Tuple[int, float]] = []
    for r in rows:
        x = _safe_int(r.get(x_col))
        a = _safe_float(r.get("metric_accuracy"))
        if x is None or a is None:
            continue
        accs.append((x, a))
    if not accs:
        return [], []
    post_means: List[float] = []
    post_mins: List[float] = []
    for g in gt_drifts:
        lo = int(g)
        hi = int(g) + int(window)
        vals = [a for (x, a) in accs if lo <= x < hi]
        if not vals:
            continue
        post_means.append(float(sum(vals) / len(vals)))
        post_mins.append(float(min(vals)))
    return post_means, post_mins


def compute_run_metrics(
    log_ref: LogRef,
    run_name_token: str,
    gt_drifts: Sequence[int],
    horizon_T: int,
    recovery_window: int,
    match_tolerance: int,
    min_separation: int,
) -> RunMetrics:
    rows: List[Dict[str, str]] = []
    with log_ref.log_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    metrics = RunMetrics(
        experiment_name=log_ref.experiment_name,
        run_name_token=run_name_token,
        run_id=log_ref.run_id,
        dataset=log_ref.dataset,
        model_variant=log_ref.model_variant,
        seed=log_ref.seed,
        detections=[],
        horizon_T=horizon_T if horizon_T > 0 else None,
        recovery_window=recovery_window,
        n_gt_drifts=len(list(gt_drifts)),
    )
    metrics.post_mean_acc_values = []
    metrics.post_min_acc_values = []
    if not rows:
        return metrics
    fieldnames = list(rows[0].keys())
    x_col = infer_x_col(fieldnames)
    acc_list: List[float] = []
    drift_flags = 0
    dets: List[int] = []

    # config snapshot
    first = rows[0]
    metrics.monitor_preset = str(first.get("monitor_preset", "")) if "monitor_preset" in first else ""
    metrics.trigger_mode = str(first.get("trigger_mode", "")) if "trigger_mode" in first else ""
    metrics.trigger_k = _safe_int(first.get("trigger_k")) if "trigger_k" in first else None
    metrics.trigger_threshold = _safe_float(first.get("trigger_threshold")) if "trigger_threshold" in first else None
    metrics.trigger_weights = str(first.get("trigger_weights", "")) if "trigger_weights" in first else ""
    metrics.severity_scheduler_scale = _safe_float(first.get("severity_scheduler_scale")) if "severity_scheduler_scale" in first else None
    metrics.use_severity_v2 = _safe_int(first.get("use_severity_v2")) if "use_severity_v2" in first else None
    metrics.entropy_mode = str(first.get("entropy_mode", "")) if "entropy_mode" in first else ""
    metrics.decay = _safe_float(first.get("decay")) if "decay" in first else None
    metrics.freeze_baseline_steps = _safe_int(first.get("freeze_baseline_steps")) if "freeze_baseline_steps" in first else None

    last_x: Optional[int] = None
    for r in rows:
        a = _safe_float(r.get("metric_accuracy"))
        if a is not None:
            acc_list.append(a)
        df = _safe_int(r.get("drift_flag"))
        if df == 1:
            drift_flags += 1
            x = _safe_int(r.get(x_col))
            if x is not None:
                dets.append(int(x))
        x = _safe_int(r.get(x_col))
        if x is not None:
            last_x = x

    metrics.acc_final = acc_list[-1] if acc_list else None
    metrics.mean_acc = float(sum(acc_list) / len(acc_list)) if acc_list else None
    metrics.acc_min = min(acc_list) if acc_list else None
    metrics.drift_flag_count = drift_flags
    metrics.detections = sorted(dets)
    if metrics.horizon_T is None and last_x is not None:
        metrics.horizon_T = int(last_x) + 1

    # detection metrics
    effective_T = int(horizon_T) if int(horizon_T) > 0 else int(metrics.horizon_T or 0)
    if compute_detection_metrics is not None and gt_drifts and effective_T > 0:
        try:
            win = compute_detection_metrics(gt_drifts, metrics.detections, effective_T)
            metrics.mdr_win = _safe_float(win.get("MDR"))
            metrics.mtd_win = _safe_float(win.get("MTD"))
            metrics.mtfa_win = _safe_float(win.get("MTFA"))
            metrics.mtr_win = _safe_float(win.get("MTR"))
        except Exception:
            pass

    merged = merge_detections(metrics.detections, min_separation)
    tol = compute_metrics_tolerance(gt_drifts, merged, effective_T, int(match_tolerance))
    metrics.mdr_tol = tol["MDR"]
    metrics.mtd_tol = tol["MTD"]
    metrics.mtfa_tol = tol["MTFA"]
    metrics.mtr_tol = tol["MTR"]

    # recovery
    post_means, post_mins = compute_recovery_metrics(rows, gt_drifts, x_col, recovery_window)
    metrics.post_mean_acc_values = post_means
    metrics.post_min_acc_values = post_mins
    metrics.post_mean_acc_mean = _mean(post_means)
    metrics.post_min_acc_mean = _mean(post_mins)
    metrics.n_recovery_drifts = len(post_means)
    return metrics


def group_summary(values: Sequence[Optional[float]]) -> str:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not vals:
        return "N/A"
    mu = float(statistics.mean([float(v) for v in vals]))
    sd = float(statistics.stdev([float(v) for v in vals])) if len(vals) >= 2 else float("nan")
    return f"{mu:.4f}±{sd:.4f}" if not math.isnan(sd) else f"{mu:.4f}±NaN"


def group_summary_pooled(runs: Sequence[RunMetrics], attr: str) -> str:
    pooled: List[float] = []
    for r in runs:
        vals = getattr(r, attr, None)
        if isinstance(vals, list):
            pooled.extend([float(x) for x in vals if x is not None and not math.isnan(float(x))])
    return group_summary(pooled)  # type: ignore[arg-type]


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    args = parse_args()
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)  # noqa: F841 (reserved for future)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_report = out_dir / args.report_name
    out_run_index = out_dir / args.run_index_name
    out_metrics = out_dir / args.metrics_table_name

    insects_meta_path = Path(args.insects_meta)
    synth_root = Path(args.synthetic_root)
    insects_drifts, insects_T = ([], 0)
    if insects_meta_path.exists():
        insects_drifts, insects_T = load_insects_meta(insects_meta_path)

    run_refs: List[RunRef] = []
    warnings: List[str] = []
    for label, (exp, token) in TRACK_RUN_TOKENS.items():
        run_ids = find_run_ids_by_suffix(logs_root, exp, token)
        chosen = choose_latest_run_id(run_ids)
        if not chosen:
            warnings.append(f"[missing] {label}: {exp} token={token} 未找到 run_id")
            continue
        run_refs.append(RunRef(experiment_name=exp, run_name_token=token, run_id=chosen))

    # collect per-run metrics
    all_runs: List[RunMetrics] = []
    for ref in run_refs:
        logs = collect_logs_for_run_id(logs_root, ref.experiment_name, ref.run_id)
        for lr in logs:
            if lr.dataset == "INSECTS_abrupt_balanced":
                gt, T = insects_drifts, insects_T
            else:
                try:
                    gt, T = load_synth_meta(synth_root, lr.dataset, lr.seed)
                except Exception:
                    gt, T = ([], 0)
            rm = compute_run_metrics(
                lr,
                run_name_token=ref.run_name_token,
                gt_drifts=gt,
                horizon_T=T,
                recovery_window=int(args.recovery_window),
                match_tolerance=int(args.match_tolerance),
                min_separation=int(args.min_separation),
            )
            all_runs.append(rm)

    # RUN INDEX CSV
    run_index_rows: List[Dict[str, object]] = []
    for rm in all_runs:
        run_index_rows.append(
            {
                "experiment_name": rm.experiment_name,
                "run_name_token": rm.run_name_token,
                "run_id": rm.run_id,
                "dataset": rm.dataset,
                "seed": rm.seed,
                "model_variant": rm.model_variant,
                "monitor_preset": rm.monitor_preset,
                "trigger_mode": rm.trigger_mode,
                "trigger_k": rm.trigger_k,
                "trigger_threshold": rm.trigger_threshold,
                "trigger_weights": rm.trigger_weights,
                "severity_scheduler_scale": rm.severity_scheduler_scale,
                "use_severity_v2": rm.use_severity_v2,
                "entropy_mode": rm.entropy_mode,
                "decay": rm.decay,
                "freeze_baseline_steps": rm.freeze_baseline_steps,
            }
        )
    write_csv(
        out_run_index,
        run_index_rows,
        [
            "experiment_name",
            "run_name_token",
            "run_id",
            "dataset",
            "seed",
            "model_variant",
            "monitor_preset",
            "trigger_mode",
            "trigger_k",
            "trigger_threshold",
            "trigger_weights",
            "severity_scheduler_scale",
            "use_severity_v2",
            "entropy_mode",
            "decay",
            "freeze_baseline_steps",
        ],
    )

    # METRICS TABLE CSV (per run)
    metrics_rows: List[Dict[str, object]] = []
    for rm in all_runs:
        base = {
            "experiment_name": rm.experiment_name,
            "run_name_token": rm.run_name_token,
            "run_id": rm.run_id,
            "dataset": rm.dataset,
            "seed": rm.seed,
            "model_variant": rm.model_variant,
            "monitor_preset": rm.monitor_preset,
            "trigger_mode": rm.trigger_mode,
            "trigger_k": rm.trigger_k,
            "trigger_threshold": rm.trigger_threshold,
            "trigger_weights": rm.trigger_weights,
            "severity_scheduler_scale": rm.severity_scheduler_scale,
            "use_severity_v2": rm.use_severity_v2,
            "entropy_mode": rm.entropy_mode,
            "decay": rm.decay,
            "freeze_baseline_steps": rm.freeze_baseline_steps,
        }
        metrics_rows.append(
            {
                **base,
                "acc_final": rm.acc_final,
                "mean_acc": rm.mean_acc,
                "acc_min": rm.acc_min,
                "drift_flag_count": rm.drift_flag_count,
                "MDR_win": rm.mdr_win,
                "MTD_win": rm.mtd_win,
                "MTFA_win": rm.mtfa_win,
                "MTR_win": rm.mtr_win,
                "MDR_tol": rm.mdr_tol,
                "MTD_tol": rm.mtd_tol,
                "MTFA_tol": rm.mtfa_tol,
                "MTR_tol": rm.mtr_tol,
                "recovery_window": rm.recovery_window,
                "post_mean_acc_mean": rm.post_mean_acc_mean,
                "post_min_acc_mean": rm.post_min_acc_mean,
                "n_gt_drifts": rm.n_gt_drifts,
                "n_recovery_drifts": rm.n_recovery_drifts,
            }
        )
    write_csv(
        out_metrics,
        metrics_rows,
        list(metrics_rows[0].keys()) if metrics_rows else [],
    )

    # helpers for report tables
    def select(group_filter) -> List[RunMetrics]:
        return [r for r in all_runs if group_filter(r)]

    def md_table(headers: List[str], rows: List[List[str]]) -> str:
        if not rows:
            return "_N/A_"
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    # Track D summary (dataset x group)
    d_groups = [
        ("INSECTS_abrupt_balanced", "A) baseline", lambda r: r.run_name_token == "trackD_insects_base"),
        ("INSECTS_abrupt_balanced", "B) v1", lambda r: r.run_name_token == "trackD_insects_v1"),
        ("INSECTS_abrupt_balanced", "C) v2(unc,d=0.95,f=5)", lambda r: r.run_name_token == "trackD_insects_v2_unc_d095_f5"),
        ("sea_abrupt4", "A) baseline", lambda r: r.run_name_token == "trackD_sea_base"),
        ("sea_abrupt4", "B) v1", lambda r: r.run_name_token == "trackD_sea_v1"),
        ("sea_abrupt4", "C) v2(unc,d=0.95,f=5)", lambda r: r.run_name_token == "trackD_sea_v2_unc_d095_f5"),
    ]
    d_rows: List[List[str]] = []
    d_group_runs: Dict[Tuple[str, str], List[RunMetrics]] = {}
    for dataset, label, pred in d_groups:
        runs = select(pred)
        d_group_runs[(dataset, label)] = runs
        d_rows.append(
            [
                dataset,
                label,
                str(len(runs)),
                group_summary([r.acc_final for r in runs]),
                group_summary([r.acc_min for r in runs]),
                group_summary_pooled(runs, "post_mean_acc_values"),
                group_summary_pooled(runs, "post_min_acc_values"),
                group_summary([r.mdr_win for r in runs]),
                group_summary([r.mtd_win for r in runs]),
                group_summary([r.mtfa_win for r in runs]),
                group_summary([r.mtr_win for r in runs]),
                group_summary([r.mdr_tol for r in runs]),
                group_summary([r.mtd_tol for r in runs]),
                group_summary([r.mtfa_tol for r in runs]),
                group_summary([r.mtr_tol for r in runs]),
            ]
        )

    # Track E summary (dataset x trigger_mode)
    e_groups = [
        ("sea_abrupt4", "or", lambda r: r.run_name_token == "trackE_or" and r.dataset == "sea_abrupt4"),
        ("sea_abrupt4", "k_of_n(k=2)", lambda r: r.run_name_token == "trackE_k2" and r.dataset == "sea_abrupt4"),
        ("sea_abrupt4", "weighted(w=0.5/0.3/0.2,t=0.5)", lambda r: r.run_name_token == "trackE_weighted_w532_t05" and r.dataset == "sea_abrupt4"),
        ("stagger_abrupt3", "or", lambda r: r.run_name_token == "trackE_or" and r.dataset == "stagger_abrupt3"),
        ("stagger_abrupt3", "k_of_n(k=2)", lambda r: r.run_name_token == "trackE_k2" and r.dataset == "stagger_abrupt3"),
        ("stagger_abrupt3", "weighted(w=0.5/0.3/0.2,t=0.5)", lambda r: r.run_name_token == "trackE_weighted_w532_t05" and r.dataset == "stagger_abrupt3"),
    ]
    e_rows: List[List[str]] = []
    e_group_runs: Dict[Tuple[str, str], List[RunMetrics]] = {}
    for dataset, label, pred in e_groups:
        runs = [r for r in all_runs if pred(r) and r.model_variant == "ts_drift_adapt"]
        e_group_runs[(dataset, label)] = runs
        e_rows.append(
            [
                dataset,
                label,
                str(len(runs)),
                group_summary([r.acc_final for r in runs]),
                group_summary([r.mean_acc for r in runs]),
                group_summary([r.mdr_win for r in runs]),
                group_summary([r.mtd_win for r in runs]),
                group_summary([r.mtfa_win for r in runs]),
                group_summary([r.mtr_win for r in runs]),
                group_summary([r.mdr_tol for r in runs]),
                group_summary([r.mtd_tol for r in runs]),
                group_summary([r.mtfa_tol for r in runs]),
                group_summary([r.mtr_tol for r in runs]),
            ]
        )

    # write report
    lines: List[str] = []
    lines.append("# NEXT_ROUND V3 Report (Track D/E)")
    lines.append("")
    lines.append(f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- logs_root：`{logs_root}`；results_root：`{results_root}`")
    lines.append(f"- 产物：`{out_report}` / `{out_run_index}` / `{out_metrics}`")
    lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in warnings])
        lines.append("")

    lines.append("## 0) Baseline 机制回顾（必须引用）")
    lines.append("- 多信号 drift monitor 的默认触发为 **OR**：任一 detector 触发即 `drift_flag=1`（现支持 `or/k_of_n/weighted`，默认仍为 OR）。")
    lines.append("- `drift_severity` 默认仅在 `drift_flag=1` 时为正（非 drift 时为 0），用于驱动 severity-aware 调度。")
    lines.append("- v1 熵项默认是 **overconfident**：`x_entropy_pos = max(0, baseline_entropy - entropy)`（教师更自信被视为 drift 风险）。")
    lines.append("- v1 severity-aware 调度仅在 `drift_flag` 的时刻通过 `drift_severity` 缩放 `alpha/lr/lambda_u/tau`。")
    lines.append("")

    lines.append("## 1) What changed")
    lines.append("- `soft_drift/severity.py`：新增 `entropy_mode ∈ {overconfident, uncertain, abs}`。")
    lines.append("- `training/loop.py`：新增 `severity_carry = max(severity_carry*decay, drift_severity)` 与 `freeze_baseline_steps`（v2 默认关闭）。")
    lines.append("- `drift/detectors.py`：新增 `trigger_mode ∈ {or, k_of_n, weighted}`（默认 OR），并记录 `monitor_vote_*`/`monitor_fused_severity`。")
    lines.append("- 推荐 weighted 参数（本轮 Track E 使用）：`w(error)=0.5, w(div)=0.3, w(entropy)=0.2, threshold=0.5`（偏向“error 必须触发”）。")
    lines.append("")

    lines.append("## 2) Track D：Severity-Aware v2（恢复收益）")
    lines.append(f"- 评估：acc_final/acc_min + drift metrics 两套口径（window-based vs tolerance={args.match_tolerance}/min_sep={args.min_separation}）+ 恢复指标 W={args.recovery_window}。")
    lines.append("")
    lines.append("### D-Table（汇总）")
    lines.append(md_table(
        [
            "dataset",
            "group",
            "n_runs",
            "acc_final(mean±std)",
            "acc_min(mean±std)",
            f"post_mean_acc@W{args.recovery_window}(mean±std)",
            f"post_min_acc@W{args.recovery_window}(mean±std)",
            "MDR_win",
            "MTD_win",
            "MTFA_win",
            "MTR_win",
            f"MDR_tol(t={args.match_tolerance})",
            f"MTD_tol(t={args.match_tolerance})",
            "MTFA_tol",
            "MTR_tol",
        ],
        d_rows,
    ))
    lines.append("")
    lines.append("### D-结论（要点）")
    lines.append("- 重点看 `acc_min` 与 post-drift 恢复指标（是否抬高谷底/加速恢复），其次看 `acc_final` 是否保持。")
    lines.append("- 若 v2 不提升：结合 `drift_flag_count`、`drift_severity` 与 carry 的持续性，判断是否 carry 衰减过快/基线被过快吸收。")
    # 结合本轮结果给出一句话结论
    sea_base = d_group_runs.get(("sea_abrupt4", "A) baseline"), [])
    sea_v1 = d_group_runs.get(("sea_abrupt4", "B) v1"), [])
    sea_v2 = d_group_runs.get(("sea_abrupt4", "C) v2(unc,d=0.95,f=5)"), [])
    insects_base = d_group_runs.get(("INSECTS_abrupt_balanced", "A) baseline"), [])
    insects_v1 = d_group_runs.get(("INSECTS_abrupt_balanced", "B) v1"), [])
    insects_v2 = d_group_runs.get(("INSECTS_abrupt_balanced", "C) v2(unc,d=0.95,f=5)"), [])
    if sea_base and sea_v1 and sea_v2:
        lines.append(
            f"- sea_abrupt4：v1/v2 的 acc_min 均值分别为 {_fmt(_mean([r.acc_min for r in sea_v1 if r.acc_min is not None]))} / {_fmt(_mean([r.acc_min for r in sea_v2 if r.acc_min is not None]))}，高于 baseline {_fmt(_mean([r.acc_min for r in sea_base if r.acc_min is not None]))}；但 v2 未明显优于 v1。"
        )
    if insects_base and insects_v1 and insects_v2:
        lines.append(
            f"- INSECTS：v2 的 acc_final/acc_min 均值 {_fmt(_mean([r.acc_final for r in insects_v2 if r.acc_final is not None]))}/{_fmt(_mean([r.acc_min for r in insects_v2 if r.acc_min is not None]))} 低于 baseline {_fmt(_mean([r.acc_final for r in insects_base if r.acc_final is not None]))}/{_fmt(_mean([r.acc_min for r in insects_base if r.acc_min is not None]))}，本轮未体现恢复收益。"
        )
    lines.append("")

    lines.append("## 3) Track E：监控融合策略（trigger_mode）")
    lines.append("")
    lines.append("### E-Table（汇总）")
    lines.append(md_table(
        [
            "dataset",
            "trigger_mode",
            "n_runs",
            "acc_final(mean±std)",
            "mean_acc(mean±std)",
            "MDR_win",
            "MTD_win",
            "MTFA_win",
            "MTR_win",
            f"MDR_tol(t={args.match_tolerance})",
            f"MTD_tol(t={args.match_tolerance})",
            "MTFA_tol",
            "MTR_tol",
        ],
        e_rows,
    ))
    lines.append("")
    lines.append("### E-结论（要点）")
    lines.append("- `k_of_n(k=2)` 在本轮 preset=error+divergence 下相当激进：检测显著减少但漏检会上升，且分类往往受影响（见表）。")
    lines.append("- `weighted(w=0.5/0.3/0.2,t=0.5)` 倾向“error 必须触发”，可在减少误报的同时维持检测覆盖（具体以 MDR/MTFA 为准）。")
    sea_or = e_group_runs.get(("sea_abrupt4", "or"), [])
    sea_k2 = e_group_runs.get(("sea_abrupt4", "k_of_n(k=2)"), [])
    sea_w = e_group_runs.get(("sea_abrupt4", "weighted(w=0.5/0.3/0.2,t=0.5)"), [])
    if sea_or and sea_k2 and sea_w:
        lines.append(
            f"- sea_abrupt4：k=2 明显掉点（acc_final 均值 {_fmt(_mean([r.acc_final for r in sea_k2 if r.acc_final is not None]))} vs OR {_fmt(_mean([r.acc_final for r in sea_or if r.acc_final is not None]))}），weighted 接近 OR（{_fmt(_mean([r.acc_final for r in sea_w if r.acc_final is not None]))}）。"
        )
    lines.append("")

    lines.append("## 4) 讨论：为何 v1 不显著、v2 为何更合理")
    lines.append("- v1 只在 `drift_flag` 当步做一次性缩放，若漂移影响持续多步（或检测延迟），单步调参可能不足以帮助恢复。")
    lines.append("- v2 的 `severity_carry` 让严重度在漂移后窗口内持续生效（可解释为“恢复期”的调参记忆），并通过 `decay` 控制影响时长。")
    lines.append("- `freeze_baseline_steps` 防止漂移后 baseline 过快贴合新分布导致严重度迅速归零，从而丢失恢复驱动。")
    lines.append("- `entropy_mode=uncertain/abs` 能覆盖“教师更不确定”的漂移形态，避免只对过度自信敏感。")
    lines.append("")

    lines.append("## 5) 下一步建议")
    lines.append("- 若 v2 在 `acc_min`/恢复指标上稳定提升：建议 sweep `decay∈{0.9,0.95,0.98}` 与 `freeze_baseline_steps∈{0,5,10}`，并对 entropy_mode 做 `overconfident vs uncertain vs abs` 对照。")
    lines.append("- 若 weighted 融合更稳：建议在 `all_signals_ph_meta` 下引入 entropy detector，并把权重/阈值与误报成本绑定（例如阈值从 0.5→0.7）。")
    lines.append("")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] report -> {out_report}")
    print(f"[done] run index -> {out_run_index}")
    print(f"[done] metrics table -> {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
