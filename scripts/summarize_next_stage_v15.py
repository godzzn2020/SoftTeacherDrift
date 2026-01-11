#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 汇总入口：
- 先复用 V14 的 summarize 逻辑生成 report/run_index/metrics_table
- 再追加 V15 验收问答段落（不影响 V14 逻辑）
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
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V15 (Permutation-test Confirm + vote_score)")
    p.add_argument("--trackal_csv", type=str, default="scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv")
    p.add_argument("--trackam_csv", type=str, default="scripts/TRACKAM_PERM_DIAG_V15.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V15_REPORT.md")
    p.add_argument("--out_run_index", type=str, default="scripts/NEXT_STAGE_V15_RUN_INDEX.csv")
    p.add_argument("--out_metrics_table", type=str, default="scripts/NEXT_STAGE_V15_METRICS_TABLE.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
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


def _parse_declared_winner(report_text: str) -> Optional[str]:
    m = re.search(r"\\*\\*winner\\*\\*\\s*\\n\\s*-\\s*`([^`]+)`", report_text)
    return m.group(1).strip() if m else None


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


def _postprocess_report_header(out_report: Path) -> Optional[str]:
    if not out_report.exists():
        return None
    text = out_report.read_text(encoding="utf-8", errors="replace")
    text = text.replace(
        "# NEXT_STAGE V14 Report（Permutation-test Confirm）",
        "# NEXT_STAGE V15 Report（Permutation-test Confirm + vote_score）",
        1,
    )
    text = text.replace("python scripts/summarize_next_stage_v14.py", "python scripts/summarize_next_stage_v15.py")
    out_report.write_text(text, encoding="utf-8")
    return text


def _append_acceptance_section(trackal_csv: Path, out_report: Path) -> None:
    rows = _read_csv(trackal_csv)
    if not rows or not out_report.exists():
        return

    best = _pick_best_phase(rows)
    by_group: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (g, d), r in best.items():
        by_group.setdefault(g, {})[d] = r

    baseline = "A_weighted_n5" if "A_weighted_n5" in by_group else None
    baseline_nd = _nd_rate(by_group, baseline) if baseline else None

    perm_groups = [g for g in by_group.keys() if str(by_group[g].get("sea_abrupt4", {}).get("confirm_rule") or "") == "perm_test"]
    hard_and_better: List[Tuple[str, float]] = []
    for g in perm_groups:
        if not _hard_ok(by_group, g):
            continue
        nd = _nd_rate(by_group, g)
        if nd is None or baseline_nd is None:
            continue
        if nd < baseline_nd:
            hard_and_better.append((g, float(nd)))
    hard_and_better.sort(key=lambda x: (x[1], x[0]))

    best_perm: Optional[str] = None
    best_perm_nd: Optional[float] = None
    for g in sorted(perm_groups):
        nd = _nd_rate(by_group, g)
        if nd is None:
            continue
        if best_perm_nd is None or nd < best_perm_nd:
            best_perm = g
            best_perm_nd = float(nd)

    report_text = _postprocess_report_header(out_report) or out_report.read_text(encoding="utf-8", errors="replace")
    declared_winner = _parse_declared_winner(report_text)
    v14_baseline_nd = _parse_v14_baseline_nd(Path("scripts/NEXT_STAGE_V14_REPORT.md"))

    lines: List[str] = []
    lines.append("")
    lines.append("## V15 验收回答")
    lines.append(f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if declared_winner:
        lines.append(f"- 规则 recompute winner：`{declared_winner}`（仅作对照；V15 约束为“不推翻 V14 winner”）")
    if v14_baseline_nd is not None:
        lines.append(f"- V14 winner 基线 no-drift（来自 `scripts/NEXT_STAGE_V14_REPORT.md`）：{v14_baseline_nd:.3f}")
    if baseline_nd is not None:
        lines.append(f"- 本次 V15 重跑基线（`{baseline}`）：confirm_rate_per_10k={baseline_nd:.3f}")

    if hard_and_better:
        g0, nd0 = hard_and_better[0]
        vios = "; ".join(_violations(by_group, g0))
        lines.append(f"- 存在 perm_test 同时满足硬约束且降低 no-drift：`{g0}`（no_drift_rate={nd0:.3f}；违约检查：{vios}）")
    else:
        lines.append("- 未发现 perm_test 同时满足硬约束且降低 no-drift（网格内）")
        if best_perm is not None and best_perm_nd is not None:
            vios = "; ".join(_violations(by_group, best_perm))
            lines.append(f"- no-drift 最低的 perm_test：`{best_perm}`（no_drift_rate={best_perm_nd:.3f}），违约：{vios}")

    out_report.write_text(report_text + "\n".join(lines) + "\n", encoding="utf-8")


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
        str(args.out_report),
        "--out_run_index",
        str(args.out_run_index),
        "--out_metrics_table",
        str(args.out_metrics_table),
        "--acc_tolerance",
        str(args.acc_tolerance),
    ]
    subprocess.check_call(cmd)
    _append_acceptance_section(Path(args.trackal_csv), Path(args.out_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
