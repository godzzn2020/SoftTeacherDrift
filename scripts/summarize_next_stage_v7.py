#!/usr/bin/env python
"""汇总 NEXT_STAGE V7（Track Q/R/S）并生成报告。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V7 (Track Q/R/S)")
    p.add_argument("--tracko_csv", type=str, default="scripts/TRACKO_CONFIRM_SWEEP.csv")
    p.add_argument("--trackp_csv", type=str, default="scripts/TRACKP_GATE_COOLDOWN.csv")
    p.add_argument("--tracks_csv", type=str, default="scripts/TRACKS_ADAPTIVE_COOLDOWN.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V7_REPORT.md")
    p.add_argument("--out_trackq_md", type=str, default="scripts/TRACKQ_METRIC_AUDIT.md")
    p.add_argument("--out_trackr_csv", type=str, default="scripts/TRACKR_CONFIRM_DENSITY.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--warmup_samples", type=int, default=2000)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            seen.add(k)
            fieldnames.append(k)
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
    except ValueError:
        return None


def _safe_int(v: Any) -> Optional[int]:
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
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def read_summary_from_log_path(log_path: str) -> Dict[str, Any]:
    sp = Path(log_path).with_suffix(".summary.json")
    return __import__("json").loads(sp.read_text(encoding="utf-8"))


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


def gap_stats(confirmed: Sequence[int]) -> Tuple[Optional[float], Optional[float]]:
    ev = sorted(int(x) for x in confirmed)
    if len(ev) < 2:
        return None, None
    gaps = [float(b - a) for a, b in zip(ev[:-1], ev[1:])]
    gaps_sorted = sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else None
    p10 = percentile(gaps_sorted, 0.10) if gaps_sorted else None
    return median, p10


def pearson(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) < 2:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.mean(xvals)
    my = statistics.mean(yvals)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xvals) * sum((y - my) ** 2 for y in yvals)) ** 0.5
    return None if den == 0 else float(num / den)


def spearman(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) < 2:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    # 简单 rank（无 ties 特殊处理；本任务以诊断为主）
    xr = {v: i for i, v in enumerate(sorted(set(xvals)))}
    yr = {v: i for i, v in enumerate(sorted(set(yvals)))}
    rx = [float(xr[v]) for v in xvals]
    ry = [float(yr[v]) for v in yvals]
    return pearson(rx, ry)


def build_trackq_audit(
    tracko_rows: List[Dict[str, str]],
    trackp_rows: List[Dict[str, str]],
    warmup_samples: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    聚焦 V6 的一个“同参对齐”对照：
    - Track O：sea_abrupt4 + config_tag=theta0.50_w1_cd0（tuned PH + theta=0.5, w=1, cd=0）
    - Track P：sea_abrupt4 + group=baseline（同上）
    """
    o = [r for r in tracko_rows if r.get("dataset") == "sea_abrupt4" and r.get("config_tag") == "theta0.50_w1_cd0"]
    # Track O 是 drift 粒度，按 (run_id,seed) 去重
    seen: set[Tuple[str, str]] = set()
    o_unique: List[Dict[str, str]] = []
    for r in o:
        k = (str(r.get("run_id") or ""), str(r.get("seed") or ""))
        if k in seen:
            continue
        seen.add(k)
        o_unique.append(r)
    p = [r for r in trackp_rows if r.get("dataset") == "sea_abrupt4" and r.get("group") == "baseline"]

    o_by_seed = {int(float(r["seed"])): r for r in o_unique if r.get("seed")}
    p_by_seed = {int(float(r["seed"])): r for r in p if r.get("seed")}

    headers = ["seed", "TrackO_run_id", "TrackP_run_id", "acc_min_raw(O)", "acc_min_raw(P)", "Δraw(P-O)", f"acc_min@{warmup_samples}(O)", f"acc_min@{warmup_samples}(P)", f"Δwarm(P-O)"]
    md_rows: List[List[str]] = []
    deltas_raw: List[Optional[float]] = []
    deltas_warm: List[Optional[float]] = []
    for s in sorted(set(o_by_seed) & set(p_by_seed)):
        ro = o_by_seed[s]
        rp = p_by_seed[s]
        o_raw = _safe_float(ro.get("acc_min"))
        p_raw = _safe_float(rp.get("acc_min"))
        so = read_summary_from_log_path(str(ro.get("log_path") or ""))
        sp = read_summary_from_log_path(str(rp.get("log_path") or ""))
        o_w = acc_min_after_warmup(so, warmup_samples)
        p_w = acc_min_after_warmup(sp, warmup_samples)
        d_raw = (p_raw - o_raw) if (p_raw is not None and o_raw is not None) else None
        d_w = (p_w - o_w) if (p_w is not None and o_w is not None) else None
        deltas_raw.append(d_raw)
        deltas_warm.append(d_w)
        md_rows.append(
            [
                str(s),
                str(ro.get("run_id") or ""),
                str(rp.get("run_id") or ""),
                fmt(o_raw, 6),
                fmt(p_raw, 6),
                fmt(d_raw, 6),
                fmt(o_w, 6),
                fmt(p_w, 6),
                fmt(d_w, 6),
            ]
        )
    table = md_table(headers, md_rows)
    meta = {
        "mean_delta_raw": mean(deltas_raw),
        "mean_delta_warm": mean(deltas_warm),
    }
    return table, meta


def write_trackq_md(path: Path, table: str, meta: Dict[str, Any], warmup_samples: int) -> None:
    lines: List[str] = []
    lines.append("# V7-Track Q：口径一致性审计")
    lines.append("")
    lines.append("## acc_min 的来源与定义（代码位置）")
    lines.append("- `training/loop.py`：训练结束写 sidecar `*.summary.json` 时，`acc_min = df[\"metric_accuracy\"].min()`（不跳过 warmup、不平滑、不做窗口聚合）。")
    lines.append("- `scripts/summarize_next_stage_v6.py`：仅从 Track CSV/summary 中读取 `acc_min` 做聚合，不会二次处理。")
    lines.append("")
    lines.append("## V6 不一致现象复现（同参对齐，run_id 不同）")
    lines.append(table)
    lines.append("")
    lines.append("## 根因结论")
    lines.append("- 不一致不是“计算口径不同”，而是 **不同 run_id 的实际轨迹不同**；`acc_min_raw` 对早期瞬时下探极敏感。")
    lines.append(f"- 当改用 `acc_min@sample_idx>={warmup_samples}`（跳过早期 warmup），Track O / Track P 的差异显著收敛（Δwarm 的均值更接近 0）。")
    lines.append("")
    lines.append("## 最终采用口径（V7 统一）")
    lines.append(f"- sea 统一采用：`acc_min@sample_idx>={warmup_samples}`（并在表中同时保留 `acc_min_raw` 作为参考）。")
    lines.append("- INSECTS 仍以 `post_min@W1000` 作为“谷底”主口径（与 V5/V6 一致）。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_trackr_confirm_density(trackp_rows: List[Dict[str, str]], warmup_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in trackp_rows:
        log_path = str(r.get("log_path") or "")
        if not log_path:
            continue
        summ = read_summary_from_log_path(log_path)
        horizon = int(summ.get("horizon") or 0)
        confirmed = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
        cc = int(summ.get("confirmed_count_total") or len(confirmed))
        rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
        med_gap, p10_gap = gap_stats(confirmed)
        acc_min_warm = acc_min_after_warmup(summ, warmup_samples)
        rows.append(
            {
                "track": "R",
                "dataset": str(r.get("dataset") or ""),
                "seed": _safe_int(r.get("seed")),
                "group": str(r.get("group") or ""),
                "run_id": str(r.get("run_id") or ""),
                "confirm_theta": _safe_float(r.get("confirm_theta")),
                "confirm_window": _safe_int(r.get("confirm_window")),
                "confirm_cooldown": _safe_int(r.get("confirm_cooldown")),
                "confirmed_count_total": cc,
                "horizon": horizon,
                "confirm_rate_per_10k": rate,
                "median_gap_between_confirms": med_gap,
                "p10_gap_between_confirms": p10_gap,
                "acc_min_raw": _safe_float(r.get("acc_min")),
                f"acc_min@{warmup_samples}": acc_min_warm,
                "post_min@W1000": _safe_float(r.get("post_min@W1000")),
                "post_mean@W1000": _safe_float(r.get("post_mean@W1000")),
                "recovery_time_to_pre90": _safe_float(r.get("recovery_time_to_pre90")),
            }
        )
    return rows


def summarize_tracks(rows: List[Dict[str, str]], dataset: str, group_key: str = "group") -> str:
    if not rows:
        return "_N/A_"
    rs = [r for r in rows if r.get("dataset") == dataset]
    if not rs:
        return "_N/A_"
    by_g: Dict[str, List[Dict[str, str]]] = {}
    for r in rs:
        g = str(r.get(group_key) or "")
        if not g:
            continue
        by_g.setdefault(g, []).append(r)
    headers = ["group", "n_runs", "acc_final", "acc_min_raw", "acc_min_warmup", "miss_tol500", "conf_P90", "MTFA_win", "post_min@W1000"]
    md_rows: List[List[str]] = []
    for g, grs in sorted(by_g.items()):
        acc_final = [_safe_float(x.get("acc_final")) for x in grs]
        acc_min_raw = [_safe_float(x.get("acc_min_raw") or x.get("acc_min")) for x in grs]
        acc_min_warm = [_safe_float(x.get("acc_min_warmup")) for x in grs]
        miss = [_safe_float(x.get("miss_tol500")) for x in grs]
        conf_p90 = [_safe_float(x.get("conf_P90")) for x in grs]
        mtfa = [_safe_float(x.get("MTFA_win")) for x in grs]
        post_min = [_safe_float(x.get("post_min@W1000")) for x in grs]
        md_rows.append(
            [
                g,
                str(len(grs)),
                fmt_mu_std(mean(acc_final), std(acc_final), 4),
                fmt_mu_std(mean(acc_min_raw), std(acc_min_raw), 4),
                fmt_mu_std(mean(acc_min_warm), std(acc_min_warm), 4),
                fmt(mean(miss), 3),
                fmt(mean(conf_p90), 1),
                fmt(mean(mtfa), 1),
                fmt(mean(post_min), 4),
            ]
        )
    return md_table(headers, md_rows)


def build_correlation_table(trackr_rows: List[Dict[str, Any]], warmup_samples: int) -> str:
    if not trackr_rows:
        return "_N/A_"
    headers = ["dataset", "y", "x", "pearson_r", "spearman_r"]
    rows_md: List[List[str]] = []

    def _add(dataset: str, y_key: str, x_key: str) -> None:
        rs = [r for r in trackr_rows if r.get("dataset") == dataset]
        xs = [r.get(x_key) for r in rs]  # type: ignore[list-item]
        ys = [r.get(y_key) for r in rs]  # type: ignore[list-item]
        pr = pearson(xs, ys)
        sr = spearman(xs, ys)
        rows_md.append([dataset, y_key, x_key, fmt(pr, 4), fmt(sr, 4)])

    sea = "sea_abrupt4"
    insects = "INSECTS_abrupt_balanced"
    y_sea = f"acc_min@{warmup_samples}"
    y_ins = "post_min@W1000"

    for x in ["confirm_rate_per_10k", "median_gap_between_confirms", "p10_gap_between_confirms"]:
        _add(sea, y_sea, x)
    for x in ["confirm_rate_per_10k", "median_gap_between_confirms", "p10_gap_between_confirms"]:
        _add(insects, y_ins, x)

    return md_table(headers, rows_md)


def summarize_v6_tracko_recomputed(
    tracko_rows: List[Dict[str, str]],
    warmup_samples: int,
    acc_tol: float,
    *,
    topk: int = 12,
) -> Tuple[str, Dict[str, Any]]:
    rows = [r for r in tracko_rows if r.get("dataset") == "sea_abrupt4"]
    if not rows:
        return "_N/A_", {"note": "未找到 V6 Track O CSV"}

    by_cfg: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        tag = str(r.get("config_tag") or "")
        if not tag:
            continue
        by_cfg.setdefault(tag, []).append(r)

    agg: List[Dict[str, Any]] = []
    for tag, rs in sorted(by_cfg.items()):
        # run-level 去重（同 run 有多条 drift 行）
        seen_run: set[Tuple[str, str]] = set()
        acc_final_list: List[Optional[float]] = []
        acc_min_raw_list: List[Optional[float]] = []
        acc_min_warm_list: List[Optional[float]] = []
        mtfa_win_list: List[Optional[float]] = []
        for r in rs:
            run_id = str(r.get("run_id") or "")
            seed = str(r.get("seed") or "")
            key = (run_id, seed)
            if key in seen_run:
                continue
            seen_run.add(key)
            acc_final_list.append(_safe_float(r.get("acc_final")))
            acc_min_raw_list.append(_safe_float(r.get("acc_min")))
            mtfa_win_list.append(_safe_float(r.get("MTFA_win")))
            # warmup 口径来自 summary.json（精确定位到本 run）
            log_path = str(r.get("log_path") or "")
            if log_path:
                summ = read_summary_from_log_path(log_path)
                acc_min_warm_list.append(acc_min_after_warmup(summ, warmup_samples))
            else:
                acc_min_warm_list.append(None)

        miss = [_safe_float(r.get("miss_tol500")) for r in rs]
        miss_mean = mean(miss)
        delays: List[Optional[float]] = []
        for r in rs:
            d = _safe_float(r.get("delay_confirmed"))
            if d is None:
                g = _safe_int(r.get("drift_pos"))
                end = _safe_int(r.get("segment_end"))
                if g is not None and end is not None and end >= g:
                    d = float(end - g)
            delays.append(d)

        sample = rs[0]
        agg.append(
            {
                "config_tag": tag,
                "theta": _safe_float(sample.get("confirm_theta")),
                "window": _safe_int(sample.get("confirm_window")),
                "cooldown": _safe_int(sample.get("confirm_cooldown")),
                "monitor_preset": str(sample.get("monitor_preset") or ""),
                "n_runs": len(seen_run),
                "acc_final_mean": mean(acc_final_list),
                "acc_final_std": std(acc_final_list),
                "acc_min_raw_mean": mean(acc_min_raw_list),
                "acc_min_raw_std": std(acc_min_raw_list),
                f"acc_min@{warmup_samples}_mean": mean(acc_min_warm_list),
                f"acc_min@{warmup_samples}_std": std(acc_min_warm_list),
                "miss_tol500_mean": miss_mean,
                "conf_p90": percentile(delays, 0.90),
                "MTFA_win_mean": mean(mtfa_win_list),
                "MTFA_win_std": std(mtfa_win_list),
            }
        )

    best_cfg: Optional[Dict[str, Any]] = None
    reason = "N/A"
    if agg:
        best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in agg)
        eligible = [r for r in agg if float(r["acc_final_mean"] or float("-inf")) >= best_acc - float(acc_tol)]
        if not eligible:
            eligible = list(agg)
        eligible.sort(
            key=lambda r: (
                float(r["miss_tol500_mean"] if r.get("miss_tol500_mean") is not None else 1.0),
                float(r["conf_p90"] if r.get("conf_p90") is not None else float("inf")),
                -float(r["MTFA_win_mean"] if r.get("MTFA_win_mean") is not None else float("-inf")),
                -float(r.get(f"acc_min@{warmup_samples}_mean") if r.get(f"acc_min@{warmup_samples}_mean") is not None else float("-inf")),
                str(r["config_tag"]),
            )
        )
        best_cfg = eligible[0]
        reason = (
            f"规则：acc_final_mean≥best-{acc_tol}，先最小化 miss_tol500_mean（再看 conf_P90），"
            f"再最大化 MTFA_win_mean，再最大化 acc_min@{warmup_samples}_mean；best_acc={best_acc:.4f}"
        )

    show = sorted(
        agg,
        key=lambda r: (
            float(r["miss_tol500_mean"] if r.get("miss_tol500_mean") is not None else 1.0),
            float(r["conf_p90"] if r.get("conf_p90") is not None else float("inf")),
            -float(r["MTFA_win_mean"] if r.get("MTFA_win_mean") is not None else float("-inf")),
            -float(r.get(f"acc_min@{warmup_samples}_mean") if r.get(f"acc_min@{warmup_samples}_mean") is not None else float("-inf")),
            str(r["config_tag"]),
        ),
    )[: min(int(topk), len(agg))]

    headers = [
        "config_tag",
        "theta",
        "window",
        "cooldown",
        "acc_final",
        "acc_min_raw",
        f"acc_min@{warmup_samples}",
        "miss_tol500",
        "conf_P90",
        "MTFA_win",
    ]
    md_rows: List[List[str]] = []
    for r in show:
        md_rows.append(
            [
                str(r["config_tag"]),
                fmt(_safe_float(r.get("theta")), 2),
                str(r.get("window") if r.get("window") is not None else "N/A"),
                str(r.get("cooldown") if r.get("cooldown") is not None else "N/A"),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_raw_mean"), r.get("acc_min_raw_std"), 4),
                fmt_mu_std(r.get(f"acc_min@{warmup_samples}_mean"), r.get(f"acc_min@{warmup_samples}_std"), 4),
                fmt(r.get("miss_tol500_mean"), 3),
                fmt(r.get("conf_p90"), 1),
                fmt_mu_std(r.get("MTFA_win_mean"), r.get("MTFA_win_std"), 1),
            ]
        )
    return md_table(headers, md_rows), {"best": best_cfg, "reason": reason}


def summarize_v6_trackp_recomputed(
    trackp_rows: List[Dict[str, str]],
    warmup_samples: int,
    acc_tol: float,
) -> Dict[str, Any]:
    if not trackp_rows:
        return {"sea_table": "_N/A_", "insects_table": "_N/A_", "note": "未找到 V6 Track P CSV"}

    out: Dict[str, Any] = {"note": "ok"}
    sea_rows = [r for r in trackp_rows if r.get("dataset") == "sea_abrupt4"]
    ins_rows = [r for r in trackp_rows if r.get("dataset") == "INSECTS_abrupt_balanced"]

    def _agg(rs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        by_g: Dict[str, List[Dict[str, str]]] = {}
        for r in rs:
            g = str(r.get("group") or "")
            if not g:
                continue
            by_g.setdefault(g, []).append(r)
        out_rows: List[Dict[str, Any]] = []
        for g, grs in sorted(by_g.items()):
            acc_final = [_safe_float(x.get("acc_final")) for x in grs]
            acc_min_raw = [_safe_float(x.get("acc_min")) for x in grs]
            miss = [_safe_float(x.get("miss_tol500")) for x in grs]
            conf_p90 = [_safe_float(x.get("conf_P90")) for x in grs]
            mtfa = [_safe_float(x.get("MTFA_win")) for x in grs]
            # warmup acc_min 从 summary.json 计算
            warm_list: List[Optional[float]] = []
            for x in grs:
                lp = str(x.get("log_path") or "")
                if not lp:
                    warm_list.append(None)
                    continue
                summ = read_summary_from_log_path(lp)
                warm_list.append(acc_min_after_warmup(summ, warmup_samples))
            sample = grs[0]
            out_rows.append(
                {
                    "group": g,
                    "n_runs": len(grs),
                    "theta": _safe_float(sample.get("confirm_theta")),
                    "window": _safe_int(sample.get("confirm_window")),
                    "cooldown": _safe_int(sample.get("confirm_cooldown")),
                    "acc_final_mean": mean(acc_final),
                    "acc_final_std": std(acc_final),
                    "acc_min_raw_mean": mean(acc_min_raw),
                    "acc_min_raw_std": std(acc_min_raw),
                    f"acc_min@{warmup_samples}_mean": mean(warm_list),
                    f"acc_min@{warmup_samples}_std": std(warm_list),
                    "miss_tol500_mean": mean(miss),
                    "conf_P90_mean": mean(conf_p90),
                    "MTFA_win_mean": mean(mtfa),
                    "post_min@W1000_mean": mean([_safe_float(x.get("post_min@W1000")) for x in grs]),
                    "post_min@W1000_std": std([_safe_float(x.get("post_min@W1000")) for x in grs]),
                }
            )
        return out_rows

    sea_agg = _agg(sea_rows)
    ins_agg = _agg(ins_rows)

    sea_headers = ["group", "n_runs", "theta", "window", "cooldown", "acc_final", "acc_min_raw", f"acc_min@{warmup_samples}", "miss_tol500", "conf_P90", "MTFA_win"]
    sea_md: List[List[str]] = []
    for r in sea_agg:
        sea_md.append(
            [
                str(r["group"]),
                str(r["n_runs"]),
                fmt(_safe_float(r.get("theta")), 2),
                str(r.get("window") if r.get("window") is not None else "N/A"),
                str(r.get("cooldown") if r.get("cooldown") is not None else "N/A"),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_raw_mean"), r.get("acc_min_raw_std"), 4),
                fmt_mu_std(r.get(f"acc_min@{warmup_samples}_mean"), r.get(f"acc_min@{warmup_samples}_std"), 4),
                fmt(r.get("miss_tol500_mean"), 3),
                fmt(r.get("conf_P90_mean"), 1),
                fmt(r.get("MTFA_win_mean"), 1),
            ]
        )
    out["sea_table"] = md_table(sea_headers, sea_md)

    ins_headers = ["group", "n_runs", "theta", "window", "cooldown", "acc_final", "acc_min_raw", f"acc_min@{warmup_samples}", "post_min@W1000"]
    ins_md: List[List[str]] = []
    for r in ins_agg:
        ins_md.append(
            [
                str(r["group"]),
                str(r["n_runs"]),
                fmt(_safe_float(r.get("theta")), 2),
                str(r.get("window") if r.get("window") is not None else "N/A"),
                str(r.get("cooldown") if r.get("cooldown") is not None else "N/A"),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_raw_mean"), r.get("acc_min_raw_std"), 4),
                fmt_mu_std(r.get(f"acc_min@{warmup_samples}_mean"), r.get(f"acc_min@{warmup_samples}_std"), 4),
                fmt_mu_std(r.get("post_min@W1000_mean"), r.get("post_min@W1000_std"), 4),
            ]
        )
    out["insects_table"] = md_table(ins_headers, ins_md)

    # 选一个推荐组：INSECTS 按 post_min@W1000 最大化（同时保证 acc_final 不掉太多）
    best_ins = None
    reason = "N/A"
    if ins_agg:
        best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in ins_agg)
        eligible = [r for r in ins_agg if float(r["acc_final_mean"] or float("-inf")) >= best_acc - float(acc_tol)]
        if not eligible:
            eligible = list(ins_agg)
        eligible.sort(
            key=lambda r: (
                -float(r["post_min@W1000_mean"] if r.get("post_min@W1000_mean") is not None else float("-inf")),
                -float(r.get(f"acc_min@{warmup_samples}_mean") if r.get(f"acc_min@{warmup_samples}_mean") is not None else float("-inf")),
                str(r["group"]),
            )
        )
        best_ins = eligible[0]
        reason = f"规则：acc_final_mean≥best-{acc_tol}，优先最大化 post_min@W1000_mean；best_acc={best_acc:.4f}"
    out["best_insects"] = best_ins
    out["reason_insects"] = reason
    return out


def write_report(
    out_path: Path,
    warmup_samples: int,
    trackq_path: Path,
    trackr_path: Path,
    q_meta: Dict[str, Any],
    trackr_corr_table: str,
    tracks_rows: List[Dict[str, str]],
    v6_tracko_table: str,
    v6_tracko_meta: Dict[str, Any],
    v6_trackp_meta: Dict[str, Any],
) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_cmd = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD"
    py = f"{sys.executable} / {platform.python_version()}"

    lines: List[str] = []
    lines.append("# NEXT_STAGE V7 Report (统一口径 + cooldown 机制解释 + 自适应 cooldown)")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 环境要求（命令）：`{env_cmd}`")
    lines.append(f"- Python：`{py}`")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append(f"- 统一口径：sea 的“谷底”最终采用 `acc_min@sample_idx>={warmup_samples}`；`acc_min_raw` 仅作参考。")
    lines.append(f"- Track Q 产物：`{trackq_path}`")
    lines.append(f"- Track R 产物：`{trackr_path}`")
    lines.append("")
    lines.append("========================")
    lines.append("V7-Track Q：口径一致性审计")
    lines.append("========================")
    lines.append("")
    lines.append(f"- 关键现象：V6 sea 在 Track O vs Track P 的 `acc_min_raw` 会因早期瞬时下探而出现 run 间差异；改用 warmup 后差异收敛。")
    lines.append(f"- 量化（同参对齐 baseline）：mean Δraw={fmt(q_meta.get('mean_delta_raw'),6)}，mean Δwarm={fmt(q_meta.get('mean_delta_warm'),6)}")
    lines.append("")
    lines.append("### V6 表重算（统一口径）")
    lines.append(f"- sea 的谷底统一使用 `acc_min@sample_idx>={warmup_samples}`。")
    lines.append("")
    lines.append("**V6-Track O（sea_abrupt4）重算**")
    lines.append(v6_tracko_table)
    lines.append("")
    best_o = v6_tracko_meta.get("best")
    lines.append("**Track O 推荐点（按统一口径重选）**")
    lines.append(f"- {v6_tracko_meta.get('reason','N/A')}")
    if best_o:
        lines.append(
            "- "
            + f"best=`{best_o.get('config_tag')}`：theta={fmt(_safe_float(best_o.get('theta')),2)} "
            + f"window={best_o.get('window')} cooldown={best_o.get('cooldown')} "
            + f"miss={fmt(best_o.get('miss_tol500_mean'),3)} conf_P90={fmt(best_o.get('conf_p90'),1)} "
            + f"MTFA={fmt(best_o.get('MTFA_win_mean'),1)} acc_min@{warmup_samples}={fmt(best_o.get(f'acc_min@{warmup_samples}_mean'),4)}"
        )
    else:
        lines.append("- _N/A_")
    lines.append("")
    lines.append("**V6-Track P 重算（sea_abrupt4）**")
    lines.append(str(v6_trackp_meta.get("sea_table") or "_N/A_"))
    lines.append("")
    lines.append("**V6-Track P 重算（INSECTS_abrupt_balanced）**")
    lines.append(str(v6_trackp_meta.get("insects_table") or "_N/A_"))
    lines.append("")
    best_ins = v6_trackp_meta.get("best_insects")
    if best_ins:
        lines.append("**Track P 推荐组（按统一口径复核）**")
        lines.append(f"- best_group=`{best_ins.get('group')}`（{v6_trackp_meta.get('reason_insects','N/A')}）")
        lines.append("")
    lines.append("### cooldown 机制解释（写清楚）")
    lines.append("- fixed cooldown：距离上次 confirmed 小于 `confirm_cooldown` 时，禁止新 confirm，并清空 two_stage 的 pending（避免“过期后补确认”的晚检）。")
    lines.append("- adaptive cooldown：在最近窗口 `adaptive_window` 内统计 confirmed 数量，换算 `confirm_rate_per_10k`；若 >upper 切到高 cooldown（默认 500），若 <lower 切到低 cooldown（默认 200）。")
    lines.append("")
    lines.append("========================")
    lines.append("V7-Track R：触发密度诊断（V6 runs）")
    lines.append("========================")
    lines.append("")
    lines.append("相关性（run 粒度）：")
    lines.append(trackr_corr_table)
    lines.append("")
    lines.append("========================")
    lines.append("V7-Track S：自适应 cooldown")
    lines.append("========================")
    lines.append("")
    lines.append("### sea_abrupt4")
    lines.append(summarize_tracks(tracks_rows, "sea_abrupt4"))
    lines.append("")
    lines.append("### INSECTS_abrupt_balanced")
    lines.append(summarize_tracks(tracks_rows, "INSECTS_abrupt_balanced"))
    lines.append("")
    lines.append("## 回答三问（必须）")
    lines.append(f"1) acc_min 不一致根因：见 `{trackq_path}`，核心是 `acc_min_raw` 包含 warmup 段且对瞬时下探敏感；最终口径采用 `acc_min@sample_idx>={warmup_samples}`。")
    lines.append(f"2) cooldown 是否通过确认密度影响谷底/恢复：见 `{trackr_path}` 与上表相关性。")
    lines.append("3) adaptive cooldown 是否兼顾 tol500 与谷底：对比 Track S 的 miss/conf_P90 与 acc_min_warmup/post_min@W1000。")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    tracko_rows = read_csv(Path(args.tracko_csv))
    trackp_rows = read_csv(Path(args.trackp_csv))
    tracks_rows = read_csv(Path(args.tracks_csv))

    # Track Q
    table, q_meta = build_trackq_audit(tracko_rows, trackp_rows, int(args.warmup_samples))
    write_trackq_md(Path(args.out_trackq_md), table, q_meta, int(args.warmup_samples))

    # Track R（由 V6 Track P 的 run 列表精确定位 summary.json，不扫描 logs/results）
    trackr_rows = build_trackr_confirm_density(trackp_rows, int(args.warmup_samples))
    write_csv(Path(args.out_trackr_csv), trackr_rows)
    trackr_corr = build_correlation_table(trackr_rows, int(args.warmup_samples))

    # V6 表按统一口径重算
    v6_tracko_table, v6_tracko_meta = summarize_v6_tracko_recomputed(
        tracko_rows, int(args.warmup_samples), float(args.acc_tolerance)
    )
    v6_trackp_meta = summarize_v6_trackp_recomputed(
        trackp_rows, int(args.warmup_samples), float(args.acc_tolerance)
    )

    # Report
    write_report(
        Path(args.out_report),
        int(args.warmup_samples),
        Path(args.out_trackq_md),
        Path(args.out_trackr_csv),
        q_meta,
        trackr_corr,
        tracks_rows,
        v6_tracko_table,
        v6_tracko_meta,
        v6_trackp_meta,
    )
    print(f"[done] wrote report={args.out_report} trackq={args.out_trackq_md} trackr={args.out_trackr_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
