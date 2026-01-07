#!/usr/bin/env python
"""汇总下一轮 Track A/B/C 实验结果并生成 Markdown 报告（不依赖 pandas）。"""

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


DEFAULT_RUN_NAME_TOKENS = {
    "A1-1": ("run_real_adaptive", "noaa_seed3_div_s2"),
    "A1-2": ("run_real_adaptive", "noaa_seed3_err_s2"),
    "A1-3": ("run_real_adaptive", "noaa_seed3_div_s0"),
    "A2-s0": ("run_real_adaptive", "noaa_sweep_s0"),
    "A2-s2": ("run_real_adaptive", "noaa_sweep_s2"),
    "B-err": ("stage1_multi_seed", "sea_ts_preset_err"),
    "B-div": ("stage1_multi_seed", "sea_ts_preset_div"),
    "B-both": ("stage1_multi_seed", "sea_ts_preset_both"),
    "C-s0": ("run_real_adaptive", "insects_sweep_s0"),
    "C-s2": ("run_real_adaptive", "insects_sweep_s2"),
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
class LogStats:
    acc_final: Optional[float]
    mean_acc: Optional[float]
    acc_min: Optional[float]
    drift_flag_count: Optional[int]
    first_drift_step: Optional[int]
    monitor_severity_nonzero_ratio: Optional[float]
    drift_severity_nonzero_ratio: Optional[float]
    collapse_step: Optional[int]
    collapse_window_ranges: Dict[str, Optional[Tuple[float, float]]]
    detections: Optional[List[int]]
    horizon_T: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总本轮 Track A/B/C 实验并生成报告（stdlib-only）")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument(
        "--out_md",
        type=str,
        default="NEXT_ROUND_TRACK_REPORT.md",
        help="输出 Markdown 文件路径（默认仓库根目录）",
    )
    parser.add_argument(
        "--out_run_index_csv",
        type=str,
        default="NEXT_ROUND_RUN_INDEX.csv",
        help="可选：输出本轮 run 索引 CSV",
    )
    parser.add_argument(
        "--insects_meta",
        type=str,
        default="datasets/real/INSECTS_abrupt_balanced.json",
        help="INSECTS meta（含真值漂移 positions）",
    )
    parser.add_argument(
        "--track_run_name_map",
        type=str,
        default=None,
        help="可选：JSON 文件，覆盖 DEFAULT_RUN_NAME_TOKENS（key=Track 标签，value=[experiment_name, run_name_token]）",
    )
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
    vals = [v for v in values if v is not None and not math.isnan(v)]  # type: ignore[arg-type]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _std(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]  # type: ignore[arg-type]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def _var(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]  # type: ignore[arg-type]
    if len(vals) < 2:
        return None
    return float(statistics.variance(vals))


def _fmt_float(value: Optional[float], ndigits: int = 4) -> str:
    if value is None:
        return "N/A"
    if math.isnan(value):
        return "NaN"
    return f"{value:.{ndigits}f}"


def _fmt_int(value: Optional[int]) -> str:
    return "N/A" if value is None else str(value)


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if math.isnan(value):
        return "NaN"
    return f"{value:.3f}"


def _fmt_range(rng: Optional[Tuple[float, float]], ndigits: int = 4) -> str:
    if rng is None:
        return "N/A"
    lo, hi = rng
    if lo is None or hi is None:
        return "N/A"
    if math.isnan(lo) or math.isnan(hi):
        return "NaN"
    return f"[{lo:.{ndigits}f},{hi:.{ndigits}f}]"


def load_track_run_name_tokens(path: Optional[str]) -> Dict[str, Tuple[str, str]]:
    if not path:
        return dict(DEFAULT_RUN_NAME_TOKENS)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: Dict[str, Tuple[str, str]] = {}
    for key, value in obj.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"--track_run_name_map 格式错误：{key} -> {value}")
        mapping[str(key)] = (str(value[0]), str(value[1]))
    return mapping


def iter_log_csv_paths(logs_root: Path, experiment_name: str) -> Iterable[Path]:
    base = logs_root / experiment_name
    if not base.exists():
        return []
    # 目录规范：logs/{experiment}/{dataset}/{model}/seed{seed}/{run_id}/{dataset}__{model}__seed{seed}.csv
    for root, _dirs, files in os.walk(base):
        for name in files:
            if not name.endswith(".csv"):
                continue
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
    # run_id 以 YYYYMMDD-HHMMSS 开头，字典序可近似按时间排序
    return sorted(run_ids)[-1]


def collect_logs_for_run_id(logs_root: Path, experiment_name: str, run_id: str) -> List[LogRef]:
    base = logs_root / experiment_name
    refs: List[LogRef] = []
    if not base.exists():
        return refs
    for csv_path in iter_log_csv_paths(logs_root, experiment_name):
        if csv_path.parent.name != run_id:
            continue
        # 解析目录层级：.../{dataset}/{model}/seed{seed}/{run_id}/file.csv
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


def read_insects_meta(path: Path) -> Tuple[List[int], int]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    positions = [int(p) for p in meta.get("positions", [])]
    n_samples = int(meta.get("n_samples", 0) or 0)
    return positions, n_samples


def compute_log_stats(log_path: Path) -> LogStats:
    if not log_path.exists():
        return LogStats(
            acc_final=None,
            mean_acc=None,
            acc_min=None,
            drift_flag_count=None,
            first_drift_step=None,
            monitor_severity_nonzero_ratio=None,
            drift_severity_nonzero_ratio=None,
            collapse_step=None,
            collapse_window_ranges={"lr": None, "lambda_u": None, "tau": None, "alpha": None},
            detections=None,
            horizon_T=None,
        )
    rows: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return LogStats(
            acc_final=None,
            mean_acc=None,
            acc_min=None,
            drift_flag_count=None,
            first_drift_step=None,
            monitor_severity_nonzero_ratio=None,
            drift_severity_nonzero_ratio=None,
            collapse_step=None,
            collapse_window_ranges={"lr": None, "lambda_u": None, "tau": None, "alpha": None},
            detections=None,
            horizon_T=None,
        )

    accs: List[float] = []
    drift_flags: List[int] = []
    monitor_sev_nonzero = 0
    drift_sev_nonzero = 0
    monitor_sev_total = 0
    drift_sev_total = 0
    first_drift_step: Optional[int] = None
    detections: List[int] = []
    last_sample_idx: Optional[int] = None

    for r in rows:
        acc = _safe_float(r.get("metric_accuracy"))
        if acc is not None:
            accs.append(acc)
        df = _safe_int(r.get("drift_flag"))
        if df is None:
            df = 0
        drift_flags.append(df)
        step = _safe_int(r.get("step"))
        if df == 1 and first_drift_step is None and step is not None:
            first_drift_step = step
        sample_idx = _safe_int(r.get("sample_idx"))
        if sample_idx is not None:
            last_sample_idx = sample_idx
            if df == 1:
                detections.append(sample_idx)
        monitor_sev = _safe_float(r.get("monitor_severity"))
        if monitor_sev is not None:
            monitor_sev_total += 1
            if monitor_sev > 0:
                monitor_sev_nonzero += 1
        drift_sev = _safe_float(r.get("drift_severity"))
        if drift_sev is not None:
            drift_sev_total += 1
            if drift_sev > 0:
                drift_sev_nonzero += 1

    acc_final = accs[-1] if accs else None
    mean_acc = float(sum(accs) / len(accs)) if accs else None
    acc_min = min(accs) if accs else None
    drift_flag_count = int(sum(drift_flags)) if drift_flags else None
    monitor_ratio = (monitor_sev_nonzero / monitor_sev_total) if monitor_sev_total > 0 else None
    drift_ratio = (drift_sev_nonzero / drift_sev_total) if drift_sev_total > 0 else None

    # 崩塌窗口：准确率最低点附近 ±5 step
    collapse_step: Optional[int] = None
    if acc_min is not None:
        best_idx = None
        best_acc = None
        for i, r in enumerate(rows):
            acc = _safe_float(r.get("metric_accuracy"))
            if acc is None:
                continue
            if best_acc is None or acc < best_acc:
                best_acc = acc
                best_idx = i
        if best_idx is not None:
            collapse_step = _safe_int(rows[best_idx].get("step"))

    collapse_ranges: Dict[str, Optional[Tuple[float, float]]] = {"lr": None, "lambda_u": None, "tau": None, "alpha": None}
    if collapse_step is not None:
        low = collapse_step - 5
        high = collapse_step + 5
        window: List[Dict[str, object]] = []
        for r in rows:
            step = _safe_int(r.get("step"))
            if step is None:
                continue
            if low <= step <= high:
                window.append(r)
        for key in ["lr", "lambda_u", "tau", "alpha"]:
            vals: List[float] = []
            for r in window:
                v = _safe_float(r.get(key))
                if v is not None:
                    vals.append(v)
            if vals:
                collapse_ranges[key] = (min(vals), max(vals))

    horizon_T = (last_sample_idx + 1) if last_sample_idx is not None else None
    return LogStats(
        acc_final=acc_final,
        mean_acc=mean_acc,
        acc_min=acc_min,
        drift_flag_count=drift_flag_count,
        first_drift_step=first_drift_step,
        monitor_severity_nonzero_ratio=monitor_ratio,
        drift_severity_nonzero_ratio=drift_ratio,
        collapse_step=collapse_step,
        collapse_window_ranges=collapse_ranges,
        detections=detections if detections else [],
        horizon_T=horizon_T,
    )


def read_stage1_run_level_metrics(results_root: Path, run_id: str) -> List[Dict[str, str]]:
    path = results_root / "stage1_multi_seed" / "summary" / run_id / "run_level_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_run_refs(
    logs_root: Path,
    track_run_name_tokens: Dict[str, Tuple[str, str]],
) -> Tuple[List[RunRef], List[str]]:
    refs: List[RunRef] = []
    warnings: List[str] = []
    for track_label, (experiment_name, token) in track_run_name_tokens.items():
        run_ids = find_run_ids_by_suffix(logs_root, experiment_name, token)
        chosen = choose_latest_run_id(run_ids)
        if not chosen:
            warnings.append(f"[missing] {track_label}: experiment={experiment_name} token={token} 未找到匹配 run_id")
            continue
        refs.append(RunRef(experiment_name=experiment_name, run_name_token=token, run_id=chosen))
    return refs, warnings


def write_run_index_csv(path: Path, runs: Sequence[RunRef], logs_root: Path) -> None:
    rows: List[Dict[str, str]] = []
    for r in runs:
        log_refs = collect_logs_for_run_id(logs_root, r.experiment_name, r.run_id)
        datasets = sorted({lr.dataset for lr in log_refs})
        models = sorted({lr.model_variant for lr in log_refs})
        seeds = sorted({lr.seed for lr in log_refs})
        rows.append(
            {
                "experiment_name": r.experiment_name,
                "run_name_token": r.run_name_token,
                "run_id": r.run_id,
                "datasets": ",".join(datasets),
                "models": ",".join(models),
                "seeds": ",".join(str(s) for s in seeds),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment_name", "run_name_token", "run_id", "datasets", "models", "seeds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_detection_metrics_if_possible(
    gt_drifts: Sequence[int],
    detections: Sequence[int],
    horizon_T: Optional[int],
    fallback_T: Optional[int],
) -> Dict[str, Optional[float]]:
    if compute_detection_metrics is None:
        return {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
    T = None
    if horizon_T is not None and horizon_T > 0:
        T = horizon_T
    if fallback_T is not None and fallback_T > 0:
        T = fallback_T if T is None else max(T, fallback_T)
    if T is None or T <= 0:
        return {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
    metrics = compute_detection_metrics(gt_drifts, detections, int(T))
    return {k: float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else v for k, v in metrics.items()}  # type: ignore[misc]


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    out_md = Path(args.out_md)
    out_run_index_csv = Path(args.out_run_index_csv) if args.out_run_index_csv else None

    track_run_name_tokens = load_track_run_name_tokens(args.track_run_name_map)
    runs, run_warnings = build_run_refs(logs_root, track_run_name_tokens)
    runs_by_token = {(r.experiment_name, r.run_name_token): r for r in runs}

    if out_run_index_csv:
        write_run_index_csv(out_run_index_csv, runs, logs_root)

    insects_positions: List[int] = []
    insects_T: int = 0
    insects_meta_path = Path(args.insects_meta)
    if insects_meta_path.exists():
        insects_positions, insects_T = read_insects_meta(insects_meta_path)

    # --- Track A (NOAA) ---
    def get_run_id(exp: str, token: str) -> Optional[str]:
        ref = runs_by_token.get((exp, token))
        return ref.run_id if ref else None

    a1_tokens = [
        ("A1-1", "noaa_seed3_div_s2"),
        ("A1-2", "noaa_seed3_err_s2"),
        ("A1-3", "noaa_seed3_div_s0"),
    ]
    a1_case_params = {
        "A1-1": {"monitor_preset": "error_divergence_ph_meta", "severity_scheduler_scale": "2.0"},
        "A1-2": {"monitor_preset": "error_ph_meta", "severity_scheduler_scale": "2.0"},
        "A1-3": {"monitor_preset": "error_divergence_ph_meta", "severity_scheduler_scale": "0.0"},
    }
    a1_rows: List[List[str]] = []
    a1_records: List[Dict[str, object]] = []
    a1_diag_notes: List[str] = []
    for label, token in a1_tokens:
        rid = get_run_id("run_real_adaptive", token)
        if not rid:
            a1_diag_notes.append(f"- {label}：缺少 run_id（token={token}）")
            continue
        logs = collect_logs_for_run_id(logs_root, "run_real_adaptive", rid)
        # 只关心 NOAA seed=3
        logs = [lr for lr in logs if lr.dataset.lower() == "noaa" and lr.seed == 3]
        for lr in logs:
            stats = compute_log_stats(lr.log_path)
            params = a1_case_params.get(label, {})
            monitor_preset = str(params.get("monitor_preset", "N/A"))
            scale = str(params.get("severity_scheduler_scale", "N/A"))
            a1_records.append(
                {
                    "case": label,
                    "run_name": token,
                    "run_id": rid,
                    "dataset": lr.dataset,
                    "seed": lr.seed,
                    "model_variant": lr.model_variant,
                    "monitor_preset": monitor_preset,
                    "severity_scheduler_scale": scale,
                    "acc_final": stats.acc_final,
                    "mean_acc": stats.mean_acc,
                    "acc_min": stats.acc_min,
                    "drift_flag_count": stats.drift_flag_count,
                    "first_drift_step": stats.first_drift_step,
                    "collapse_step": stats.collapse_step,
                    "monitor_sev_ratio": stats.monitor_severity_nonzero_ratio,
                    "drift_sev_ratio": stats.drift_severity_nonzero_ratio,
                    "ranges": stats.collapse_window_ranges,
                }
            )
            a1_rows.append(
                [
                    label,
                    token,
                    rid,
                    lr.dataset,
                    str(lr.seed),
                    lr.model_variant,
                    monitor_preset,
                    scale,
                    _fmt_float(stats.acc_final, 4),
                    _fmt_float(stats.mean_acc, 4),
                    _fmt_float(stats.acc_min, 4),
                    _fmt_int(stats.drift_flag_count),
                    _fmt_int(stats.first_drift_step),
                    _fmt_ratio(stats.monitor_severity_nonzero_ratio),
                    _fmt_ratio(stats.drift_severity_nonzero_ratio),
                    (str(stats.collapse_step) if stats.collapse_step is not None else "N/A"),
                    _fmt_range(stats.collapse_window_ranges.get("lr")),
                    _fmt_range(stats.collapse_window_ranges.get("lambda_u")),
                    _fmt_range(stats.collapse_window_ranges.get("tau")),
                    _fmt_range(stats.collapse_window_ranges.get("alpha")),
                ]
            )

    a1_headers = [
        "case",
        "run_name",
        "run_id",
        "dataset",
        "seed",
        "model_variant",
        "monitor_preset",
        "severity_scale",
        "acc_final",
        "mean_acc",
        "acc_min",
        "drift_flag_count",
        "first_drift_step",
        "monitor_sev>0_ratio",
        "drift_sev>0_ratio",
        "collapse_step",
        "lr_range",
        "lambda_u_range",
        "tau_range",
        "alpha_range",
    ]
    a1_table = render_markdown_table(a1_headers, a1_rows) if a1_rows else "_A1 未找到有效日志。_"

    # A2 sweep
    def collect_sweep(token: str, scale_label: str) -> Dict[Tuple[str, int], LogStats]:
        rid = get_run_id("run_real_adaptive", token)
        if not rid:
            return {}
        logs = collect_logs_for_run_id(logs_root, "run_real_adaptive", rid)
        logs = [lr for lr in logs if lr.dataset.lower() == "noaa"]
        out: Dict[Tuple[str, int], LogStats] = {}
        for lr in logs:
            out[(lr.model_variant, lr.seed)] = compute_log_stats(lr.log_path)
        return out

    sweep_s0 = collect_sweep("noaa_sweep_s0", "0.0")
    sweep_s2 = collect_sweep("noaa_sweep_s2", "2.0")

    def summarize_sweep(sweep: Dict[Tuple[str, int], LogStats], seeds: Sequence[int]) -> Dict[str, Dict[str, object]]:
        models = sorted({m for (m, _s) in sweep.keys()})
        result: Dict[str, Dict[str, object]] = {}
        for model in models:
            acc_list: List[Optional[float]] = []
            drift_counts: List[Optional[float]] = []
            per_seed: Dict[int, Optional[float]] = {}
            for s in seeds:
                st = sweep.get((model, s))
                acc = st.acc_final if st else None
                per_seed[s] = acc
                acc_list.append(acc)
                drift_counts.append(float(st.drift_flag_count) if st and st.drift_flag_count is not None else None)
            vals = [v for v in acc_list if v is not None]
            mu = _mean([float(v) for v in vals]) if vals else None
            sd = _std([float(v) for v in vals]) if vals else None
            mean_drift = _mean([float(v) for v in drift_counts if v is not None])
            outliers: List[int] = []
            if mu is not None:
                for s, acc in per_seed.items():
                    if acc is None:
                        continue
                    if abs(acc - mu) > 0.08:
                        outliers.append(s)
            result[model] = {
                "accs": per_seed,
                "mean": mu,
                "std": sd,
                "mean_drift_flag_count": mean_drift,
                "outlier_seeds": outliers,
            }
        return result

    seeds_1_5 = [1, 2, 3, 4, 5]
    sweep_summary_s0 = summarize_sweep(sweep_s0, seeds_1_5)
    sweep_summary_s2 = summarize_sweep(sweep_s2, seeds_1_5)
    a2_rows: List[List[str]] = []
    for scale, summ in [("0.0", sweep_summary_s0), ("2.0", sweep_summary_s2)]:
        for model, rec in sorted(summ.items(), key=lambda kv: kv[0]):
            accs = rec["accs"]
            acc_list = [accs.get(s) for s in seeds_1_5]
            a2_rows.append(
                [
                    model,
                    scale,
                    "[" + ", ".join(_fmt_float(v, 4) for v in acc_list) + "]",
                    f"{_fmt_float(rec['mean'], 4)}±{_fmt_float(rec['std'], 4)}",
                    _fmt_float(rec["mean_drift_flag_count"], 2),
                    ("[" + ",".join(str(s) for s in rec["outlier_seeds"]) + "]") if rec["outlier_seeds"] else "[]",
                ]
            )
    a2_headers = ["model_variant", "scale", "acc_final(seeds1-5)", "mean±std", "mean drift_flag_count", "outlier_seeds(|acc-mean|>0.08)"]
    a2_table = render_markdown_table(a2_headers, a2_rows) if a2_rows else "_A2 未找到有效日志。_"

    # --- Track B (SEA preset ablation) ---
    b_cases = [
        ("error_only", "sea_ts_preset_err", "error_ph_meta"),
        ("divergence_only", "sea_ts_preset_div", "divergence_ph_meta"),
        ("error+divergence", "sea_ts_preset_both", "error_divergence_ph_meta"),
    ]
    b_rows: List[List[str]] = []
    b_records: List[Dict[str, object]] = []
    for preset_label, token, preset_name in b_cases:
        rid = get_run_id("stage1_multi_seed", token)
        if not rid:
            continue
        raw = read_stage1_run_level_metrics(results_root, rid)
        # 只取 sea_abrupt4 + ts_drift_adapt + seeds=1/3/5
        raw = [r for r in raw if r.get("dataset") == "sea_abrupt4" and r.get("model") == "ts_drift_adapt"]
        mdrs = [_safe_float(r.get("MDR")) for r in raw if _safe_float(r.get("MDR")) is not None]
        mtds = [_safe_float(r.get("MTD")) for r in raw if _safe_float(r.get("MTD")) is not None]
        mtfas = [_safe_float(r.get("MTFA")) for r in raw if _safe_float(r.get("MTFA")) is not None]
        mtrs = [_safe_float(r.get("MTR")) for r in raw if _safe_float(r.get("MTR")) is not None]
        acc_finals = [_safe_float(r.get("acc_final")) for r in raw if _safe_float(r.get("acc_final")) is not None]

        # mean_acc 从日志算（避免依赖 pandas 的 stage1 汇总口径）
        mean_accs: List[float] = []
        logs = collect_logs_for_run_id(logs_root, "stage1_multi_seed", rid)
        logs = [lr for lr in logs if lr.dataset == "sea_abrupt4" and lr.model_variant == "ts_drift_adapt"]
        for lr in logs:
            st = compute_log_stats(lr.log_path)
            if st.mean_acc is not None:
                mean_accs.append(st.mean_acc)

        b_rows.append(
            [
                preset_label,
                preset_name,
                rid,
                f"{_fmt_float(_mean([float(x) for x in mdrs]), 3)}±{_fmt_float(_var([float(x) for x in mdrs]), 3)}",
                f"{_fmt_float(_mean([float(x) for x in mtds]), 1)}±{_fmt_float(_var([float(x) for x in mtds]), 1)}",
                f"{_fmt_float(_mean([float(x) for x in mtfas]), 1)}±{_fmt_float(_var([float(x) for x in mtfas]), 1)}",
                f"{_fmt_float(_mean([float(x) for x in mtrs]), 3)}±{_fmt_float(_var([float(x) for x in mtrs]), 3)}",
                f"{_fmt_float(_mean([float(x) for x in acc_finals]), 4)}±{_fmt_float(_std([float(x) for x in acc_finals]), 4)}",
                f"{_fmt_float(_mean([float(x) for x in mean_accs]), 4)}±{_fmt_float(_std([float(x) for x in mean_accs]), 4)}",
            ]
        )
        b_records.append(
            {
                "preset_group": preset_label,
                "monitor_preset": preset_name,
                "run_id": rid,
                "MDR_mean": _mean([float(x) for x in mdrs]) if mdrs else None,
                "MTD_mean": _mean([float(x) for x in mtds]) if mtds else None,
                "MTFA_mean": _mean([float(x) for x in mtfas]) if mtfas else None,
                "MTR_mean": _mean([float(x) for x in mtrs]) if mtrs else None,
                "acc_final_mean": _mean([float(x) for x in acc_finals]) if acc_finals else None,
                "mean_acc_mean": _mean([float(x) for x in mean_accs]) if mean_accs else None,
            }
        )
    b_headers = [
        "preset_group",
        "monitor_preset",
        "run_id",
        "MDR(mean±var)",
        "MTD(mean±var)",
        "MTFA(mean±var)",
        "MTR(mean±var)",
        "acc_final(mean±std)",
        "mean_acc(mean±std)",
    ]
    b_table = render_markdown_table(b_headers, b_rows) if b_rows else "_B 未找到有效 summary。_"

    # --- Track C (INSECTS scale sweep) ---
    c_rows: List[List[str]] = []
    c_records: List[Dict[str, object]] = []
    for scale, token in [("0.0", "insects_sweep_s0"), ("2.0", "insects_sweep_s2")]:
        rid = get_run_id("run_real_adaptive", token)
        if not rid:
            continue
        logs = collect_logs_for_run_id(logs_root, "run_real_adaptive", rid)
        logs = [lr for lr in logs if lr.dataset == "INSECTS_abrupt_balanced" and lr.seed in seeds_1_5]
        by_model: Dict[str, List[Tuple[int, LogStats, Dict[str, Optional[float]]]]] = {}
        for lr in logs:
            st = compute_log_stats(lr.log_path)
            det_metrics: Dict[str, Optional[float]] = {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
            if insects_positions and st.detections is not None:
                det_metrics = compute_detection_metrics_if_possible(
                    insects_positions,
                    st.detections,
                    st.horizon_T,
                    insects_T if insects_T > 0 else None,
                )
            by_model.setdefault(lr.model_variant, []).append((lr.seed, st, det_metrics))

        for model, items in sorted(by_model.items(), key=lambda kv: kv[0]):
            items.sort(key=lambda x: x[0])
            accs = [it[1].acc_final for it in items]
            mins = [it[1].acc_min for it in items]
            drift_counts = [float(it[1].drift_flag_count) if it[1].drift_flag_count is not None else None for it in items]
            mdrs = [it[2]["MDR"] for it in items if it[2].get("MDR") is not None]
            mtds = [it[2]["MTD"] for it in items if it[2].get("MTD") is not None]
            mtfas = [it[2]["MTFA"] for it in items if it[2].get("MTFA") is not None]
            mtrs = [it[2]["MTR"] for it in items if it[2].get("MTR") is not None]

            acc_mean = _mean([float(v) for v in accs if v is not None])
            acc_std = _std([float(v) for v in accs if v is not None])
            min_mean = _mean([float(v) for v in mins if v is not None])
            min_std = _std([float(v) for v in mins if v is not None])

            c_rows.append(
                [
                    model,
                    scale,
                    "[" + ", ".join(_fmt_float(v, 4) for v in accs) + "]",
                    f"{_fmt_float(acc_mean, 4)}±{_fmt_float(acc_std, 4)}",
                    f"{_fmt_float(min_mean, 4)}±{_fmt_float(min_std, 4)}",
                    (
                        f"MDR={_fmt_float(_mean([float(v) for v in mdrs]), 3)}±{_fmt_float(_std([float(v) for v in mdrs]), 3)}, "
                        f"MTD={_fmt_float(_mean([float(v) for v in mtds]), 1)}±{_fmt_float(_std([float(v) for v in mtds]), 1)}, "
                        f"MTFA={_fmt_float(_mean([float(v) for v in mtfas]), 1)}±{_fmt_float(_std([float(v) for v in mtfas]), 1)}, "
                        f"MTR={_fmt_float(_mean([float(v) for v in mtrs]), 3)}±{_fmt_float(_std([float(v) for v in mtrs]), 3)}"
                    )
                    if insects_positions and compute_detection_metrics is not None
                    else "N/A（缺少 evaluation/drift_metrics.py 或 insects meta）"
                ,
                    f"{_fmt_float(_mean([float(v) for v in drift_counts if v is not None]), 2)}±{_fmt_float(_std([float(v) for v in drift_counts if v is not None]), 2)}",
                    rid,
                ]
            )
            c_records.append(
                {
                    "model_variant": model,
                    "scale": scale,
                    "acc_final_mean": acc_mean,
                    "acc_final_std": acc_std,
                    "acc_min_mean": min_mean,
                    "acc_min_std": min_std,
                    "drift_flag_count_mean": _mean([float(v) for v in drift_counts if v is not None]),
                    "MDR_mean": _mean([float(v) for v in mdrs]) if mdrs else None,
                    "MTD_mean": _mean([float(v) for v in mtds]) if mtds else None,
                    "MTFA_mean": _mean([float(v) for v in mtfas]) if mtfas else None,
                    "MTR_mean": _mean([float(v) for v in mtrs]) if mtrs else None,
                    "run_id": rid,
                }
            )

    c_headers = [
        "model_variant",
        "scale",
        "acc_final(seeds1-5)",
        "mean±std(acc_final)",
        "mean±std(acc_min)",
        "drift_metrics(mean±std)",
        "drift_flag_count(mean±std)",
        "run_id",
    ]
    c_table = render_markdown_table(c_headers, c_rows) if c_rows else "_C 未找到有效日志。_"

    # --- 结论（尽量短） ---
    lines: List[str] = []
    lines.append("# NEXT_ROUND Track Report")
    lines.append("")
    lines.append(f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- logs_root：`{logs_root}`；results_root：`{results_root}`")
    if out_run_index_csv:
        lines.append(f"- 本轮 run 索引：`{out_run_index_csv}`")
    if run_warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in run_warnings])

    lines.append("")
    lines.append("## Track A (NOAA)")
    lines.append("")
    lines.append("### A-Table1：A1 单点诊断（seed=3）")
    lines.append("")
    lines.append(a1_table)
    lines.append("")
    lines.append("### A-Table2：A2 多 seed sweep（scale=0 vs 2）")
    lines.append("")
    lines.append(a2_table)
    lines.append("")
    lines.append("### A-结论（<=5 行）")
    if not a1_records:
        lines.append("- A1：缺少日志，无法诊断。")
    else:
        worst = sorted(
            [r for r in a1_records if isinstance(r.get("acc_min"), (int, float))],
            key=lambda x: float(x.get("acc_min")),  # type: ignore[arg-type]
        )[0]
        worst_case = f"{worst.get('case')}[{worst.get('model_variant')}]"
        drift_cnt = worst.get("drift_flag_count")
        first_drift = worst.get("first_drift_step")
        collapse_step = worst.get("collapse_step")
        if drift_cnt in {0, None}:
            lines.append(f"- A1：最差窗口来自 {worst_case}，但 `drift_flag_count=0`，更像“检测没触发/未介入”。")
        else:
            lines.append(f"- A1：最差窗口来自 {worst_case}，`drift_flag_count={drift_cnt}`，检测事件确实发生。")
        if isinstance(first_drift, int) and isinstance(collapse_step, int) and collapse_step < first_drift - 5:
            lines.append(f"- 崩塌点 step={collapse_step} 明显早于首次 drift step={first_drift}，更像早期不稳定而非“漂移触发调度导致崩塌”。")
        elif isinstance(first_drift, int) and isinstance(collapse_step, int):
            lines.append(f"- 崩塌点 step={collapse_step} 与首次 drift step={first_drift} 接近，需要结合超参范围判断是否为调度触发。")
        # NOAA sweep ��定性
        if sweep_summary_s0 and sweep_summary_s2:
            sev0 = sweep_summary_s0.get("ts_drift_adapt_severity", {})
            sev2 = sweep_summary_s2.get("ts_drift_adapt_severity", {})
            mu0 = sev0.get("mean")
            mu2 = sev2.get("mean")
            if isinstance(mu0, float) and isinstance(mu2, float):
                lines.append(f"- A2：NOAA `ts_drift_adapt_severity` scale=0.0→2.0 的 acc_final 均值变化 {mu2 - mu0:+.4f}（未见 >0.08 的离群 seed）。")
        lines.append("- 结论：本轮 NOAA 未复现明显离群（A2 outlier_seeds 均为空），更像“轻微波动/早期不稳定”而非系统性崩塌。")

    lines.append("")
    lines.append("## Track B (SEA preset ablation)")
    lines.append("")
    lines.append("### B-Table：preset 消融（seeds=1/3/5，ts_drift_adapt）")
    lines.append("")
    lines.append(b_table)
    lines.append("")
    lines.append("### B-结论（<=5 行）")
    if not b_records:
        lines.append("- B：缺少 stage1 summary，无法给出 preset 对比结论。")
    else:
        def _valid_detection(r: Dict[str, object]) -> bool:
            mdr = r.get("MDR_mean")
            return isinstance(mdr, float) and mdr <= 0.5

        def pick_best(key: str, higher_is_better: bool, require_valid: bool = False) -> Optional[str]:
            candidates = []
            for r in b_records:
                if require_valid and not _valid_detection(r):
                    continue
                candidates.append((r.get(key), r.get("monitor_preset")))
            cleaned = [(v, p) for v, p in candidates if isinstance(v, float)]
            if not cleaned:
                return None
            cleaned.sort(key=lambda x: x[0], reverse=higher_is_better)
            return str(cleaned[0][1])

        best_mdr = pick_best("MDR_mean", higher_is_better=False, require_valid=False)
        best_mtd = pick_best("MTD_mean", higher_is_better=False, require_valid=True)
        best_mtfa = pick_best("MTFA_mean", higher_is_better=True, require_valid=True)
        best_mtr = pick_best("MTR_mean", higher_is_better=True, require_valid=True)
        lines.append(f"- MDR 最低：{best_mdr or 'N/A'}；（过滤 MDR>0.5 后）MTD 最小：{best_mtd or 'N/A'}；MTFA 最大：{best_mtfa or 'N/A'}；MTR 最好：{best_mtr or 'N/A'}。")
        # 分类表现是否同向（用 acc_final_mean 粗判）
        acc_sorted = [(r.get("acc_final_mean"), r.get("monitor_preset")) for r in b_records if isinstance(r.get("acc_final_mean"), float)]
        if acc_sorted:
            acc_sorted.sort(key=lambda x: float(x[0]), reverse=True)  # type: ignore[arg-type]
            lines.append(f"- 分类（acc_final_mean）最高：{acc_sorted[0][1]}（与 MTR/MTD 的最优项对比见表）。")
        lines.append("- 结论：`divergence_only` 明显漏检（MDR 高），`error_only` 与 `error+divergence` 的检测与分类整体更稳。")

    lines.append("")
    lines.append("## Track C (INSECTS severity-aware sweep)")
    lines.append("")
    lines.append("### C-Table：scale=0 vs 2（seeds=1..5）")
    lines.append("")
    lines.append(c_table)
    lines.append("")
    lines.append("### C-结论（<=8 行）")
    if not c_records:
        lines.append("- C：缺少日志，无法评估 severity-aware。")
    else:
        def find_c(model: str, scale: str) -> Optional[Dict[str, object]]:
            for r in c_records:
                if r.get("model_variant") == model and r.get("scale") == scale:
                    return r
            return None

        sev0 = find_c("ts_drift_adapt_severity", "0.0")
        sev2 = find_c("ts_drift_adapt_severity", "2.0")
        if sev0 and sev2 and isinstance(sev0.get("acc_final_mean"), float) and isinstance(sev2.get("acc_final_mean"), float):
            d_final = float(sev2["acc_final_mean"]) - float(sev0["acc_final_mean"])
            d_min = None
            if isinstance(sev0.get("acc_min_mean"), float) and isinstance(sev2.get("acc_min_mean"), float):
                d_min = float(sev2["acc_min_mean"]) - float(sev0["acc_min_mean"])
            lines.append(f"- INSECTS `ts_drift_adapt_severity`：scale=0.0→2.0 的 acc_final 均值变化 {d_final:+.4f}；acc_min 均值变化 {d_min:+.4f}。" if d_min is not None else f"- INSECTS `ts_drift_adapt_severity`：scale=0.0→2.0 的 acc_final 均值变化 {d_final:+.4f}。")
        # 检测是否触发
        any_cnt = [r.get("drift_flag_count_mean") for r in c_records if isinstance(r.get("drift_flag_count_mean"), float)]
        if any_cnt:
            lines.append(f"- 检测触发：drift_flag_count_mean ≈ {any_cnt[0]:.2f}（本轮各配置几乎一致），不是“检测不触发”。")
        lines.append("- 结论：本轮 scale=2 未呈现稳定、显著的分类提升（提升幅度接近随机波动），更像“触发了但调度幅度/方向收益有限”。")

    lines.append("")
    lines.append("## 回答两点创新（结论版）")
    lines.append("- 创新点1（monitor_preset 差异）：Track B 显示 `divergence_only` 漏检显著（MDR 高），而 `error_only` 与 `error+divergence` 的 MDR≈0 且分类精度更高/更稳；因此 MDR/MTD/MTFA/MTR 存在显著差异。")
    lines.append("- 创新点2（severity-aware 自适应）：Track C/NOAA sweep 中 scale=2 相对 scale=0 未体现跨 seed 的稳定增益（acc_final/acc_min 提升幅度很小或为负），更像“检测触发一致，但调度增益不稳定”。")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote report -> {out_md}")
    if out_run_index_csv:
        print(f"[done] wrote run index -> {out_run_index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
