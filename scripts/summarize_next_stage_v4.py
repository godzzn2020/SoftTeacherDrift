#!/usr/bin/env python
"""自动汇总 Track I/J/K/L 并生成 NEXT_STAGE_V4 报告与统一表。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize next stage V4 (Track I/J/K/L)")
    p.add_argument("--tracki_csv", type=str, default="scripts/TRACKI_DELAY_DIAG.csv")
    p.add_argument("--trackj_csv", type=str, default="scripts/TRACKJ_TWOSTAGE_SWEEP.csv")
    p.add_argument("--trackk_csv", type=str, default="scripts/TRACKK_GENERALIZATION.csv")
    p.add_argument("--trackl_csv", type=str, default="scripts/TRACKL_GATING_SWEEP.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V4_REPORT.md")
    p.add_argument("--out_run_index", type=str, default="scripts/NEXT_STAGE_V4_RUN_INDEX.csv")
    p.add_argument("--out_metrics_table", type=str, default="scripts/NEXT_STAGE_V4_METRICS_TABLE.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(v: object) -> Optional[int]:
    x = _safe_float(v)
    return None if x is None else int(x)


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


def percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
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


def fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "N/A"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{nd}f}"


def fmt_mu_std(mu: Optional[float], sd: Optional[float], nd: int = 4) -> str:
    if mu is None:
        return "N/A"
    if sd is None:
        return f"{fmt(mu, nd)}±N/A"
    return f"{fmt(mu, nd)}±{fmt(sd, nd)}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def group_rows(rows: List[Dict[str, str]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    out: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for r in rows:
        k = tuple(str(r.get(x, "") or "") for x in keys)
        out.setdefault(k, []).append(r)
    return out


def summarize_track_i(rows: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, object]]]:
    if not rows:
        return "_N/A_", []
    groups = group_rows(rows, ["dataset", "mode"])
    table_rows: List[List[str]] = []
    summary_items: List[Dict[str, object]] = []
    for (dataset, mode), rs in sorted(groups.items()):
        delays_cand = [_safe_float(r.get("delay_candidate")) for r in rs]
        delays_conf = [_safe_float(r.get("delay_confirmed")) for r in rs]
        miss_win = mean([1.0 if r.get("first_confirmed_step") in {None, "", "None"} else 0.0 for r in rs])
        miss_tol = mean([_safe_float(r.get("miss_tol500")) for r in rs])
        p50_c = percentile(delays_cand, 0.5)
        p90_c = percentile(delays_cand, 0.9)
        p99_c = percentile(delays_cand, 0.99)
        p50_f = percentile(delays_conf, 0.5)
        p90_f = percentile(delays_conf, 0.9)
        p99_f = percentile(delays_conf, 0.99)
        n = len(rs)
        table_rows.append(
            [
                dataset,
                mode,
                str(n),
                fmt(miss_win, 3),
                fmt(miss_tol, 3),
                fmt(p50_c, 1),
                fmt(p90_c, 1),
                fmt(p99_c, 1),
                fmt(p50_f, 1),
                fmt(p90_f, 1),
                fmt(p99_f, 1),
            ]
        )
        summary_items.append(
            {
                "track": "I",
                "dataset": dataset,
                "mode": mode,
                "n_drifts_total": n,
                "miss_rate_win": miss_win,
                "miss_rate_tol500": miss_tol,
                "delay_candidate_p50": p50_c,
                "delay_candidate_p90": p90_c,
                "delay_candidate_p99": p99_c,
                "delay_confirmed_p50": p50_f,
                "delay_confirmed_p90": p90_f,
                "delay_confirmed_p99": p99_f,
            }
        )
    headers = [
        "dataset",
        "mode",
        "n(drifts×seeds)",
        "miss_win",
        "miss_tol500",
        "cand_P50",
        "cand_P90",
        "cand_P99",
        "conf_P50",
        "conf_P90",
        "conf_P99",
    ]
    return md_table(headers, table_rows), summary_items


def summarize_track_j(rows: List[Dict[str, str]], acc_tol: float) -> Tuple[str, Dict[str, Tuple[Optional[float], Optional[int], str]], Tuple[Optional[float], Optional[int], str]]:
    if not rows:
        return "_N/A_", {}, (None, None, "N/A")
    # 聚合到 dataset×(theta,window)
    grouped: Dict[Tuple[str, float, int], List[Dict[str, str]]] = {}
    for r in rows:
        ds = r.get("dataset") or ""
        theta = _safe_float(r.get("confirm_theta"))
        window = _safe_int(r.get("confirm_window"))
        if not ds or theta is None or window is None:
            continue
        grouped.setdefault((ds, float(theta), int(window)), []).append(r)

    # 计算表格 + per-dataset 推荐
    best_acc: Dict[str, float] = {}
    agg: Dict[Tuple[str, float, int], Dict[str, Optional[float]]] = {}
    for key, rs in grouped.items():
        ds, theta, window = key
        acc_mu = mean([_safe_float(x.get("acc_final")) for x in rs])
        if acc_mu is not None:
            best_acc[ds] = max(best_acc.get(ds, float("-inf")), float(acc_mu))
        agg[key] = {
            "acc_final_mean": acc_mu,
            "acc_final_std": std([_safe_float(x.get("acc_final")) for x in rs]),
            "acc_min_mean": mean([_safe_float(x.get("acc_min")) for x in rs]),
            "acc_min_std": std([_safe_float(x.get("acc_min")) for x in rs]),
            "MDR_tol_mean": mean([_safe_float(x.get("MDR_tol")) for x in rs]),
            "MTR_win_mean": mean([_safe_float(x.get("MTR_win")) for x in rs]),
            "miss_tol_mean": mean([_safe_float(x.get("miss_rate_tol500")) for x in rs]),
            "delay_p90_mean": mean([_safe_float(x.get("delay_p90")) for x in rs]),
        }

    table_rows: List[List[str]] = []
    rec_per_dataset: Dict[str, Tuple[Optional[float], Optional[int], str]] = {}
    for ds in sorted(best_acc.keys()):
        candidates: List[Tuple[float, int, float, float, float]] = []
        for (dsi, theta, window), a in agg.items():
            if dsi != ds:
                continue
            acc_mu = a["acc_final_mean"]
            if acc_mu is None:
                continue
            if float(acc_mu) < float(best_acc[ds]) - acc_tol:
                continue
            miss = float(a["miss_tol_mean"] if a["miss_tol_mean"] is not None else 1.0)
            acc_min = float(a["acc_min_mean"] if a["acc_min_mean"] is not None else float("-inf"))
            mtr = float(a["MTR_win_mean"] if a["MTR_win_mean"] is not None else float("-inf"))
            candidates.append((theta, window, miss, acc_min, mtr))
        if not candidates:
            rec_per_dataset[ds] = (None, None, "无法推荐：无满足 acc 约束候选")
            continue
        candidates.sort(key=lambda x: (x[2], -x[3], -x[4], x[0], x[1]))
        theta, window, miss, acc_min, mtr = candidates[0]
        rec_per_dataset[ds] = (
            float(theta),
            int(window),
            f"规则：acc_final_mean≥best-{acc_tol}，先最小化 miss_tol500，再最大化 acc_min（再看 MTR_win）",
        )

    # 全局默认：同时满足所有 dataset 的 acc 约束，优先最小化平均 miss，再最大化平均 acc_min
    datasets = sorted(best_acc.keys())
    global_candidates: List[Tuple[float, int, float, float]] = []
    all_thetas = sorted({k[1] for k in agg.keys()})
    all_windows = sorted({k[2] for k in agg.keys()})
    for theta in all_thetas:
        for window in all_windows:
            ok = True
            miss_list: List[float] = []
            accmin_list: List[float] = []
            for ds in datasets:
                a = agg.get((ds, theta, window))
                if not a or a["acc_final_mean"] is None:
                    ok = False
                    break
                if float(a["acc_final_mean"]) < float(best_acc[ds]) - acc_tol:
                    ok = False
                    break
                miss_list.append(float(a["miss_tol_mean"] if a["miss_tol_mean"] is not None else 1.0))
                accmin_list.append(float(a["acc_min_mean"] if a["acc_min_mean"] is not None else float("-inf")))
            if not ok:
                continue
            global_candidates.append((float(theta), int(window), float(statistics.mean(miss_list)), float(statistics.mean(accmin_list))))
    if not global_candidates:
        global_rec = (None, None, "无法给出全局默认：无满足所有数据集 acc 约束的组合")
    else:
        global_candidates.sort(key=lambda x: (x[2], -x[3], x[0], x[1]))
        theta, window, miss_mu, accmin_mu = global_candidates[0]
        global_rec = (theta, window, f"全局默认：平均 miss_tol500 最小（{miss_mu:.3f}），且平均 acc_min 最大（{accmin_mu:.3f}）")

    # 生成展示表
    for (ds, theta, window), a in sorted(agg.items()):
        table_rows.append(
            [
                ds,
                fmt(theta, 2),
                str(window),
                fmt_mu_std(a["acc_final_mean"], a["acc_final_std"], 4),
                fmt_mu_std(a["acc_min_mean"], a["acc_min_std"], 4),
                fmt(a["miss_tol_mean"], 3),
                fmt(a["delay_p90_mean"], 1),
                fmt(a["MDR_tol_mean"], 3),
                fmt(a["MTR_win_mean"], 3),
            ]
        )
    headers = [
        "dataset",
        "theta",
        "window",
        "acc_final(mean±std)",
        "acc_min(mean±std)",
        "miss_tol500(mean)",
        "delay_P90(mean)",
        "MDR_tol(mean)",
        "MTR_win(mean)",
    ]
    return md_table(headers, table_rows), rec_per_dataset, global_rec


def summarize_track_k(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_N/A_"
    grouped = group_rows(rows, ["dataset", "mode"])
    table_rows: List[List[str]] = []
    for (dataset, mode), rs in sorted(grouped.items()):
        table_rows.append(
            [
                dataset,
                mode,
                str(len(rs)),
                fmt_mu_std(mean([_safe_float(x.get("acc_final")) for x in rs]), std([_safe_float(x.get("acc_final")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("acc_min")) for x in rs]), std([_safe_float(x.get("acc_min")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("post_mean_acc")) for x in rs]), std([_safe_float(x.get("post_mean_acc")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("post_min_acc")) for x in rs]), std([_safe_float(x.get("post_min_acc")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("recovery_time_to_pre90")) for x in rs]), std([_safe_float(x.get("recovery_time_to_pre90")) for x in rs]), 1),
                fmt_mu_std(mean([_safe_float(x.get("MDR_win")) for x in rs]), std([_safe_float(x.get("MDR_win")) for x in rs]), 3),
                fmt_mu_std(mean([_safe_float(x.get("MDR_tol")) for x in rs]), std([_safe_float(x.get("MDR_tol")) for x in rs]), 3),
            ]
        )
    headers = [
        "dataset",
        "mode",
        "n_runs",
        "acc_final(mean±std)",
        "acc_min(mean±std)",
        "post_mean@W1000(mean±std)",
        "post_min@W1000(mean±std)",
        "recovery_time_to_pre90(mean±std)",
        "MDR_win(mean±std)",
        "MDR_tol500(mean±std)",
    ]
    return md_table(headers, table_rows)


def summarize_track_l(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_N/A_"
    grouped = group_rows(rows, ["dataset", "group"])
    table_rows: List[List[str]] = []
    for (dataset, group), rs in sorted(grouped.items()):
        table_rows.append(
            [
                dataset,
                group,
                str(len(rs)),
                fmt_mu_std(mean([_safe_float(x.get("acc_final")) for x in rs]), std([_safe_float(x.get("acc_final")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("acc_min")) for x in rs]), std([_safe_float(x.get("acc_min")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("post_mean_acc")) for x in rs]), std([_safe_float(x.get("post_mean_acc")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("post_min_acc")) for x in rs]), std([_safe_float(x.get("post_min_acc")) for x in rs]), 4),
                fmt_mu_std(mean([_safe_float(x.get("recovery_time_to_pre90")) for x in rs]), std([_safe_float(x.get("recovery_time_to_pre90")) for x in rs]), 1),
            ]
        )
    headers = [
        "dataset",
        "group",
        "n_runs",
        "acc_final(mean±std)",
        "acc_min(mean±std)",
        "post_mean@W1000(mean±std)",
        "post_min@W1000(mean±std)",
        "recovery_time_to_pre90(mean±std)",
    ]
    return md_table(headers, table_rows)


def build_run_index_and_metrics(
    track_rows: Dict[str, List[Dict[str, str]]],
    out_run_index: Path,
    out_metrics: Path,
) -> None:
    # 统一为每个 run（dataset+seed+model_variant+log_path）写一行
    run_map: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for track, rows in track_rows.items():
        for r in rows:
            dataset = str(r.get("dataset", "") or "")
            seed = str(r.get("seed", "") or "")
            model = str(r.get("model_variant", "") or "")
            log_path = str(r.get("log_path", "") or "")
            if not dataset or not seed or not model or not log_path:
                continue
            key = (dataset, seed, model, log_path)
            base = run_map.get(key, {})
            base.update(
                {
                    "track": track,
                    "experiment_name": r.get("experiment_name", ""),
                    "run_id": r.get("run_id", ""),
                    "dataset": dataset,
                    "seed": int(float(seed)),
                    "model_variant": model,
                    "mode": r.get("mode", r.get("group", "")),
                    "monitor_preset": r.get("monitor_preset", ""),
                    "trigger_mode": r.get("trigger_mode", ""),
                    "trigger_threshold": _safe_float(r.get("trigger_threshold")),
                    "confirm_window": _safe_int(r.get("confirm_window")),
                    "weights": r.get("weights", ""),
                    "use_severity_v2": _safe_int(r.get("use_severity_v2")),
                    "severity_gate": r.get("severity_gate", ""),
                    "severity_gate_min_streak": _safe_int(r.get("severity_gate_min_streak")),
                    "entropy_mode": r.get("entropy_mode", ""),
                    "severity_decay": _safe_float(r.get("severity_decay")),
                    "freeze_baseline_steps": _safe_int(r.get("freeze_baseline_steps")),
                    "severity_scheduler_scale": _safe_float(r.get("severity_scheduler_scale")),
                    "log_path": log_path,
                }
            )
            run_map[key] = base

    out_run_index.parent.mkdir(parents=True, exist_ok=True)
    index_rows = list(run_map.values())
    if index_rows:
        fieldnames = list(index_rows[0].keys())
        with out_run_index.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(index_rows)

    # 统一指标表：尽量从各 track 的 CSV 直接取（同 run 可能出现多个 track，优先 K/L/J，其次 I）
    priority = {"L": 3, "K": 2, "J": 1, "I": 0}
    metrics_map: Dict[Tuple[str, str, str, str], Tuple[int, Dict[str, object]]] = {}
    for track, rows in track_rows.items():
        for r in rows:
            dataset = str(r.get("dataset", "") or "")
            seed = str(r.get("seed", "") or "")
            model = str(r.get("model_variant", "") or "")
            log_path = str(r.get("log_path", "") or "")
            if not dataset or not seed or not model or not log_path:
                continue
            key = (dataset, seed, model, log_path)
            pr = priority.get(track, 0)
            prev = metrics_map.get(key)
            if prev is not None and prev[0] > pr:
                continue
            metrics_map[key] = (
                pr,
                {
                    "track": track,
                    "run_id": r.get("run_id", ""),
                    "dataset": dataset,
                    "seed": int(float(seed)),
                    "model_variant": model,
                    "mode": r.get("mode", r.get("group", "")),
                    "log_path": log_path,
                    "acc_final": _safe_float(r.get("acc_final")),
                    "mean_acc": _safe_float(r.get("mean_acc")),
                    "acc_min": _safe_float(r.get("acc_min")),
                    "drift_flag_count": _safe_int(r.get("drift_flag_count")) or _safe_int(r.get("confirmed_count")),
                    "MDR_win": _safe_float(r.get("MDR_win")),
                    "MTD_win": _safe_float(r.get("MTD_win")),
                    "MTFA_win": _safe_float(r.get("MTFA_win")),
                    "MTR_win": _safe_float(r.get("MTR_win")),
                    "MDR_tol500": _safe_float(r.get("MDR_tol")),
                    "MTD_tol500": _safe_float(r.get("MTD_tol")),
                    "MTFA_tol500": _safe_float(r.get("MTFA_tol")),
                    "MTR_tol500": _safe_float(r.get("MTR_tol")),
                    "miss_rate_tol500": _safe_float(r.get("miss_rate_tol500")) or _safe_float(r.get("miss_tol500")),
                    "delay_p90": _safe_float(r.get("delay_p90")),
                    "post_mean_acc_W1000": _safe_float(r.get("post_mean_acc")),
                    "post_min_acc_W1000": _safe_float(r.get("post_min_acc")),
                    "recovery_time_to_pre90": _safe_float(r.get("recovery_time_to_pre90")),
                },
            )
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = [v[1] for v in metrics_map.values()]
    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with out_metrics.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(metrics_rows)


def main() -> int:
    args = parse_args()
    tracki_path = Path(args.tracki_csv)
    trackj_path = Path(args.trackj_csv)
    trackk_path = Path(args.trackk_csv)
    trackl_path = Path(args.trackl_csv)

    rows_i = read_csv(tracki_path)
    rows_j = read_csv(trackj_path)
    rows_k = read_csv(trackk_path)
    rows_l = read_csv(trackl_path)

    tracki_table, _ = summarize_track_i(rows_i)
    trackj_table, rec_j, global_rec = summarize_track_j(rows_j, float(args.acc_tolerance))
    trackk_table = summarize_track_k(rows_k)
    trackl_table = summarize_track_l(rows_l)

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    python_ver = platform.python_version()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# NEXT_STAGE V4 Report (Track I/J/K/L)")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`")
    lines.append(f"- Python：`{python_exec}` / `{python_ver}`")
    lines.append("")
    lines.append("## 最小定位（入口与关键产物路径）")
    lines.append("- 入口代码：`experiments/trackI_delay_diag.py`、`experiments/trackJ_twostage_sweep.py`、`experiments/trackK_generalization.py`、`experiments/trackL_gating_sweep.py`")
    lines.append("- 关键实现：`drift/detectors.py`、`training/loop.py`、`soft_drift/severity.py`")
    lines.append("- 汇总脚本：`scripts/summarize_next_stage_v4.py`")
    lines.append("- 既有产物（F/G/H）：`scripts/TRACKF_THRESHOLD_SWEEP.csv`、`scripts/TRACKG_TWO_STAGE.csv`、`scripts/TRACKH_SEVERITY_GATING.csv`、`scripts/NEXT_STAGE_REPORT.md`")
    lines.append("- 本阶段产物（I/J/K/L）：`scripts/TRACKI_DELAY_DIAG.csv`、`scripts/TRACKJ_TWOSTAGE_SWEEP.csv`、`scripts/TRACKK_GENERALIZATION.csv`、`scripts/TRACKL_GATING_SWEEP.csv`")
    lines.append("")
    lines.append("## 口径统一（window vs tol500）")
    lines.append("- 时间轴：本报告对合成/INSECTS 的 GT drift 均使用 `sample_idx`（与 meta.json 的 positions 对齐）。")
    lines.append("- window 口径（Lukats window-based）：只要在“当前 drift 到下一个 drift 之间”出现过一次报警，就不算漏检；因此**晚检也会被视为命中**。")
    lines.append("- tol500 口径：要求在 `drift_pos + 500` 内出现报警才算命中；因此**晚检会被计为漏检**。")
    lines.append("- 结论：当出现 `MDR_win≈0` 但 `MDR_tol500` 较高时，通常不是完全漏检，而是**检测延迟显著（P90/P99>500）**。Track I 会给出延迟分位数来解释该冲突。")
    lines.append("")
    lines.append("## Track I：延迟诊断（必须）")
    lines.append(tracki_table)
    lines.append("")
    lines.append("**Track I 结论（<=8 行）**")
    if rows_i:
        lines.append("- 若某个 mode 的 `miss_win` 低但 `miss_tol500` 高，同时 `conf_P90` 或 `conf_P99` > 500：说明主要问题是“晚检”，论文中应把 tol500 作为辅助口径用于强调实时性。")
        lines.append("- `or` 往往 candidate 最早但误报更多；`weighted`/`two_stage` 倾向降低误报但可能带来确认延迟。")
    else:
        lines.append("- TODO：未找到 Track I CSV（请先运行 experiments/trackI_delay_diag.py）。")
    lines.append("")
    lines.append("## Track J：two_stage 小扫（稳健默认）")
    lines.append(trackj_table)
    lines.append("")
    lines.append("**推荐组合（按数据集）**")
    if rec_j:
        for ds, (theta, window, reason) in rec_j.items():
            lines.append(f"- {ds}: theta={theta if theta is not None else 'N/A'}, window={window if window is not None else 'N/A'}（{reason}）")
    else:
        lines.append("- TODO：未找到 Track J CSV（请先运行 experiments/trackJ_twostage_sweep.py）。")
    lines.append("")
    lines.append("**推荐组合（全局默认）**")
    lines.append(f"- theta={global_rec[0] if global_rec[0] is not None else 'N/A'}, window={global_rec[1] if global_rec[1] is not None else 'N/A'}（{global_rec[2]}）")
    lines.append("")
    lines.append("## Track K：泛化验证（INSECTS 必须）")
    lines.append(trackk_table)
    lines.append("")
    lines.append("**Track K 结论（<=8 行）**")
    if rows_k:
        lines.append("- 重点看 `acc_min`、`post_*@W1000` 与 `recovery_time_to_pre90` 是否在 two_stage 下更稳（且 `acc_final` 不明显下降）。")
        lines.append("- 若 two_stage 的 `MDR_tol500` 下降且延迟统计改善，同时分类不掉点：可作为论文默认触发策略。")
    else:
        lines.append("- TODO：未找到 Track K CSV（请先运行 experiments/trackK_generalization.py）。")
    lines.append("")
    lines.append("## Track L：severity v2 gating 强化验证（INSECTS）")
    lines.append(trackl_table)
    lines.append("")
    lines.append("**Track L 结论（<=8 行）**")
    if rows_l:
        lines.append("- 对比 `v2` vs `v2_gate_*`：若 gating 组的 `acc_min` 与 `post_min@W1000` 更高且方差更小，同时 `acc_final` 不显著下降，可支持“confirmed drift gating 缓解负迁移”的结论。")
        lines.append("- 若 gating 过强导致 `acc_final` 下滑或恢复指标变差：说明确认条件太苛刻（可回退更小的 min_streak 或降低 trigger_threshold）。")
    else:
        lines.append("- TODO：未找到 Track L CSV（请先运行 experiments/trackL_gating_sweep.py）。")
    lines.append("")
    lines.append("## 最终建议（论文可用默认）")
    lines.append("- 主口径建议：window-based 作为主表（与既有文献一致）；tol500 作为辅表强调实时性，并用 Track I 的 delay 分位数解释冲突来源。")
    lines.append("- 默认触发建议：two_stage（candidate=OR，confirm=weighted），采用 Track J 的“全局默认”组合；并在论文方法段明确 candidate/confirm 统计（candidate_count/confirmed_count/confirm_delay）。")
    lines.append("- severity v2 建议：默认开启 confirmed drift gating（优先 `confirmed_streak`），以 Track L 的恢复指标与 acc_min 作为核心证据。")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote report -> {out_report}")

    build_run_index_and_metrics(
        {"I": rows_i, "J": rows_j, "K": rows_k, "L": rows_l},
        out_run_index=Path(args.out_run_index),
        out_metrics=Path(args.out_metrics_table),
    )
    print(f"[done] wrote run index -> {args.out_run_index}")
    print(f"[done] wrote metrics table -> {args.out_metrics_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
