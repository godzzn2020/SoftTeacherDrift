#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 汇总入口：
- 先复用 V14 的 summarize 逻辑生成 report/run_index/metrics_table
- 再生成“主报告（瘦身）”与“全量报告（大表）”
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V15 (Permutation-test Confirm + vote_score)")
    p.add_argument("--trackal_csv", type=str, default="scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv")
    p.add_argument("--trackam_csv", type=str, default="scripts/TRACKAM_PERM_DIAG_V15.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V15_REPORT.md")
    p.add_argument("--out_report_full", type=str, default="scripts/NEXT_STAGE_V15_REPORT_FULL.md")
    p.add_argument("--out_run_index", type=str, default="scripts/NEXT_STAGE_V15_RUN_INDEX.csv")
    p.add_argument("--out_metrics_table", type=str, default="scripts/NEXT_STAGE_V15_METRICS_TABLE.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=20, help="主报告顶部 Top-K hard-ok 表行数")
    return p.parse_args()


def _safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a"}:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _is_zero(v: Optional[float], tol: float = 1e-12) -> bool:
    return v is not None and abs(v) <= tol


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_best_phase(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    # (group,dataset) 可能出现 quick/full；优先 full
    def rank(r: Dict[str, str]) -> int:
        return 0 if str(r.get("phase") or "") == "full" else 1

    best: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in sorted(rows, key=rank):
        g = str(r.get("group") or "")
        d = str(r.get("dataset") or "")
        if not g or not d:
            continue
        best.setdefault((g, d), r)
    return best


def _nd_rate(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> Optional[float]:
    sea = _safe_float(by_group.get(g, {}).get("sea_nodrift", {}).get("confirm_rate_per_10k_mean"))
    sine = _safe_float(by_group.get(g, {}).get("sine_nodrift", {}).get("confirm_rate_per_10k_mean"))
    if sea is None or sine is None:
        return None
    return (sea + sine) / 2.0


def _nd_mtfa(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> Optional[float]:
    sea = _safe_float(by_group.get(g, {}).get("sea_nodrift", {}).get("MTFA_win_mean"))
    sine = _safe_float(by_group.get(g, {}).get("sine_nodrift", {}).get("MTFA_win_mean"))
    if sea is None or sine is None:
        return None
    return (sea + sine) / 2.0


def _drift_acc_final(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> Optional[float]:
    sea = _safe_float(by_group.get(g, {}).get("sea_abrupt4", {}).get("acc_final_mean"))
    sine = _safe_float(by_group.get(g, {}).get("sine_abrupt4", {}).get("acc_final_mean"))
    if sea is None or sine is None:
        return None
    return (sea + sine) / 2.0


def _hard_ok(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> bool:
    sea = by_group.get(g, {}).get("sea_abrupt4", {})
    sine = by_group.get(g, {}).get("sine_abrupt4", {})
    sea_miss = _safe_float(sea.get("miss_tol500_mean"))
    sine_miss = _safe_float(sine.get("miss_tol500_mean"))
    sea_p90 = _safe_float(sea.get("conf_P90_mean"))
    sine_p90 = _safe_float(sine.get("conf_P90_mean"))
    return _is_zero(sea_miss) and _is_zero(sine_miss) and (sea_p90 is not None and sea_p90 < 500.0) and (sine_p90 is not None and sine_p90 < 500.0)


def _violations(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> List[str]:
    sea = by_group.get(g, {}).get("sea_abrupt4", {})
    sine = by_group.get(g, {}).get("sine_abrupt4", {})
    sea_miss = _safe_float(sea.get("miss_tol500_mean"))
    sine_miss = _safe_float(sine.get("miss_tol500_mean"))
    sea_p90 = _safe_float(sea.get("conf_P90_mean"))
    sine_p90 = _safe_float(sine.get("conf_P90_mean"))
    out: List[str] = []
    if not _is_zero(sea_miss) or not _is_zero(sine_miss):
        out.append(f"miss!=0（sea={sea_miss} sine={sine_miss}）")
    if (sea_p90 is None or sea_p90 >= 500.0) or (sine_p90 is None or sine_p90 >= 500.0):
        out.append(f"confP90>=500（sea={sea_p90} sine={sine_p90}）")
    return out or ["N/A"]


def _parse_v14_baseline_nd(v14_report: Path) -> Optional[float]:
    if not v14_report.exists():
        return None
    lines = v14_report.read_text(encoding="utf-8", errors="replace").splitlines()
    header: Optional[List[str]] = None
    for i, line in enumerate(lines):
        if line.startswith("|") and "group" in line and "no_drift_rate" in line:
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            # 下一行是 --- 分隔行；从后续开始找数据行
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                cols = [c.strip() for c in row.strip().strip("|").split("|")]
                if not cols or cols[0] != "A_weighted_n5":
                    continue
                try:
                    idx = header.index("no_drift_rate")
                except ValueError:
                    return None
                if idx >= len(cols):
                    return None
                return _safe_float(cols[idx])
    return None


def _postprocess_report_header_text(text: str, *, title: str, summarize_cmd: str) -> str:
    out = str(text)
    out = out.replace("# NEXT_STAGE V14 Report（Permutation-test Confirm）", title, 1)
    out = out.replace("python scripts/summarize_next_stage_v14.py", summarize_cmd)
    return out


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{nd}f}"


def _build_slim_report(
    *,
    trackal_csv: Path,
    trackam_csv: Path,
    out_report: Path,
    out_report_full: Path,
    topk: int,
) -> None:
    rows = _read_csv(trackal_csv)
    if not rows:
        raise FileNotFoundError(f"空或缺失：{trackal_csv}")

    best = _pick_best_phase(rows)
    by_group: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (g, d), r in best.items():
        by_group.setdefault(g, {})[d] = r

    baseline_group = "A_weighted_n5" if "A_weighted_n5" in by_group else (sorted(by_group.keys())[0] if by_group else "")
    baseline_nd = _nd_rate(by_group, baseline_group) if baseline_group else None

    perm_groups = [g for g in by_group.keys() if str(by_group[g].get("sea_abrupt4", {}).get("confirm_rule") or "") == "perm_test"]
    hard_perm = [g for g in perm_groups if _hard_ok(by_group, g)]
    hard_perm.sort(key=lambda g: (_nd_rate(by_group, g) if _nd_rate(by_group, g) is not None else float("inf"), -(float(_nd_mtfa(by_group, g) or -1e18)), g))

    hard_and_better: List[str] = []
    if baseline_nd is not None:
        for g in hard_perm:
            nd = _nd_rate(by_group, g)
            if nd is None:
                continue
            if nd < baseline_nd:
                hard_and_better.append(g)

    best_hard = hard_perm[0] if hard_perm else None
    best_hard_nd = _nd_rate(by_group, best_hard) if best_hard else None
    best_hard_mtfa = _nd_mtfa(by_group, best_hard) if best_hard else None
    best_hard_acc = _drift_acc_final(by_group, best_hard) if best_hard else None

    best_better = hard_and_better[0] if hard_and_better else None
    best_better_nd = _nd_rate(by_group, best_better) if best_better else None
    best_better_mtfa = _nd_mtfa(by_group, best_better) if best_better else None
    best_better_acc = _drift_acc_final(by_group, best_better) if best_better else None

    def meta(g: str, key: str) -> str:
        sea = by_group.get(g, {}).get("sea_abrupt4", {})
        v = sea.get(key)
        return str(v) if v is not None else ""

    top_rows: List[List[str]] = []
    for g in hard_perm[: max(0, int(topk))]:
        sea = by_group.get(g, {}).get("sea_abrupt4", {})
        sine = by_group.get(g, {}).get("sine_abrupt4", {})
        top_rows.append(
            [
                g,
                meta(g, "perm_stat"),
                meta(g, "perm_alpha"),
                meta(g, "perm_pre_n"),
                meta(g, "perm_post_n"),
                _fmt(_safe_float(sea.get("miss_tol500_mean")), 3),
                _fmt(_safe_float(sine.get("miss_tol500_mean")), 3),
                _fmt(_safe_float(sea.get("conf_P90_mean")), 1),
                _fmt(_safe_float(sine.get("conf_P90_mean")), 1),
                _fmt(_nd_rate(by_group, g), 3),
                _fmt(_nd_mtfa(by_group, g), 1),
                _fmt(_drift_acc_final(by_group, g), 4),
            ]
        )

    title = "# NEXT_STAGE V15.1 Report（TopK + 验收前置）"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    v14_baseline_nd = _parse_v14_baseline_nd(Path("scripts/NEXT_STAGE_V14_REPORT.md"))

    lines: List[str] = []
    lines.append(title)
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- TrackAL：`{trackal_csv}`")
    lines.append(f"- TrackAM：`{trackam_csv}`")
    lines.append(f"- 全量报告：`{out_report_full}`（含全表）")
    lines.append("")
    lines.append("## 说明（口径）")
    lines.append("- V15 在 V14 pipeline 上新增 `vote_score` 的 `perm_test` 统计量分支，并支持更小的 `post_n` 与分片并行。")
    lines.append("- 本节验收：只检查是否存在满足硬约束（sea/sine `miss==0` 且 `confP90<500`）且 no-drift 低于 baseline 的 `perm_test` 配置。")
    if v14_baseline_nd is not None:
        lines.append(f"- 参考：历史 V14 报告中 `A_weighted_n5` no_drift_rate={v14_baseline_nd:.3f}（仅供对照，不作为本次 baseline）。")
    if baseline_nd is not None:
        lines.append(f"- 本次 baseline：`{baseline_group}` no_drift_rate={baseline_nd:.3f}（取自本次 TrackAL 聚合/metrics_table 口径）。")
    lines.append("")
    lines.append("## 验收结论（前置）")
    if best_better and baseline_nd is not None and best_better_nd is not None:
        delta = best_better_nd - baseline_nd
        lines.append(
            f"- ✅ 存在满足硬约束且优于 baseline 的 perm_test：`{best_better}` "
            f"(no_drift_rate={best_better_nd:.3f}, Δ={delta:+.3f}; MTFA={_fmt(best_better_mtfa,1)}; drift_acc={_fmt(best_better_acc,4)})"
        )
    else:
        lines.append("- ❌ 未发现满足硬约束且优于 baseline 的 perm_test（网格内）")
        if best_hard and best_hard_nd is not None and baseline_nd is not None:
            delta = best_hard_nd - baseline_nd
            lines.append(
                f"- 硬约束内最优候选：`{best_hard}` "
                f"(no_drift_rate={best_hard_nd:.3f}, Δ={delta:+.3f}; MTFA={_fmt(best_hard_mtfa,1)}; drift_acc={_fmt(best_hard_acc,4)})"
            )
    lines.append("")
    lines.append(f"## Top-K Hard-OK candidates（K={int(topk)}）")
    headers = [
        "group",
        "perm_stat",
        "perm_alpha",
        "perm_pre_n",
        "perm_post_n",
        "sea_miss",
        "sine_miss",
        "sea_confP90",
        "sine_confP90",
        "no_drift_rate",
        "no_drift_MTFA",
        "drift_acc_final",
    ]
    lines.append(_md_table(headers, top_rows))
    lines.append("")
    lines.append("## 产物")
    lines.append(f"- `{out_report}`")
    lines.append(f"- `{out_report_full}`")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    v14_summarizer = Path(__file__).resolve().parent / "summarize_next_stage_v14.py"
    if not v14_summarizer.exists():
        print(f"[error] missing: {v14_summarizer}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(v14_summarizer),
        "--trackal_csv",
        str(args.trackal_csv),
        "--trackam_csv",
        str(args.trackam_csv),
        "--out_report",
        str(args.out_report_full),
        "--out_run_index",
        str(args.out_run_index),
        "--out_metrics_table",
        str(args.out_metrics_table),
        "--acc_tolerance",
        str(args.acc_tolerance),
    ]
    subprocess.check_call(cmd)
    # 修正 full 报告头部，避免标题/命令误导
    out_full = Path(args.out_report_full)
    if out_full.exists():
        full_text = out_full.read_text(encoding="utf-8", errors="replace")
        full_text = _postprocess_report_header_text(
            full_text,
            title="# NEXT_STAGE V15.1 Report FULL（Permutation-test Confirm + vote_score）",
            summarize_cmd="python scripts/summarize_next_stage_v15.py",
        )
        out_full.write_text(full_text, encoding="utf-8")

    _build_slim_report(
        trackal_csv=Path(args.trackal_csv),
        trackam_csv=Path(args.trackam_csv),
        out_report=Path(args.out_report),
        out_report_full=Path(args.out_report_full),
        topk=int(args.topk),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
