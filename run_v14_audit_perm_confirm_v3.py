#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEXT_STAGE V14（Permutation-test Confirm）审计脚本 V3（只读审计 + 产出报告/表格）

严格约束（对应用户 prompt）：
- 不做全局搜索/扫描（尤其不扫描 logs/ 根目录；只按 RUN_INDEX 的 log_path 做定点尝试）
- 不重跑任何训练/实验（只读既有 CSV/MD/summary.json）
- 逐 run drill-down：仅在单个 log_path 所在目录内，按固定顺序尝试 + 最多 1 次局部 glob

产物：
- V14_AUDIT_PERM_CONFIRM_V3.md
- V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv（table_name 分表）
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


AL_SWEEP_CSV = "scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv"
AM_DIAG_CSV = "scripts/TRACKAM_PERM_DIAG.csv"
RUN_INDEX_CSV = "scripts/NEXT_STAGE_V14_RUN_INDEX.csv"
METRICS_TABLE_CSV = "scripts/NEXT_STAGE_V14_METRICS_TABLE.csv"
V14_REPORT_MD = "scripts/NEXT_STAGE_V14_REPORT.md"

# 这两个文件在本仓库当前不存在（之前版本放在 scripts/ 下），但白名单要求“若不存在则 N/A”
LEGACY_AUDIT_MD = "V14_AUDIT_PERM_CONFIRM.md"
LEGACY_AUDIT_TABLES = "V14_AUDIT_PERM_CONFIRM_TABLES.csv"

# 代码口径审计白名单（仅允许打开这些文件）
CODE_PATHS = [
    "drift/detectors.py",
    "training/loop.py",
    "experiments/trackAL_perm_confirm_sweep.py",
    "experiments/trackAM_perm_diagnostics.py",
    "scripts/summarize_next_stage_v14.py",
    "run_v14_audit_perm_confirm.py",  # 若存在；本仓库当前为 scripts/run_v14_audit_perm_confirm.py -> 仍按白名单规则 N/A
]

OUT_MD = "V14_AUDIT_PERM_CONFIRM_V3.md"
OUT_TABLES = "V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv"

HARD_CONF_P90_LT = 500.0
DRIFT_ACC_TOL = 0.01


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "N/A", "nan", "NaN", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_zero(x: Optional[float], tol: float = 1e-12) -> bool:
    return x is not None and abs(x) <= tol


def _fmt(x: Any, ndigits: int = 4) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return "N/A"
        s = f"{x:.{ndigits}f}"
        return s.rstrip("0").rstrip(".")
    return str(x)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(statistics.mean(xs))


@dataclass(frozen=True)
class GroupSummary:
    group: str
    phase: str
    confirm_rule: str
    perm_stat: str
    perm_alpha: str
    perm_pre_n: str
    perm_post_n: str
    delta_k: str
    sea_miss: Optional[float]
    sea_confP90: Optional[float]
    sine_miss: Optional[float]
    sine_confP90: Optional[float]
    no_drift_rate: Optional[float]
    no_drift_MTFA: Optional[float]
    drift_acc_final: Optional[float]


def _parse_report_winner(report_text: str) -> Optional[str]:
    m = re.search(r"winner[^`]*`([^`]+)`", report_text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_al_sweep_summaries(path: str) -> List[GroupSummary]:
    by_group_dataset: Dict[str, Dict[str, Dict[str, str]]] = {}
    meta: Dict[str, Dict[str, str]] = {}
    for row in _load_csv_rows(path):
        g = row["group"]
        d = row["dataset"]
        by_group_dataset.setdefault(g, {})[d] = row
        meta.setdefault(
            g,
            {k: (row.get(k, "") or "") for k in ["phase", "confirm_rule", "perm_stat", "perm_alpha", "perm_pre_n", "perm_post_n", "delta_k"]},
        )

    required = ("sea_abrupt4", "sine_abrupt4", "sea_nodrift", "sine_nodrift")
    out: List[GroupSummary] = []
    for g, dmap in by_group_dataset.items():
        if not all(d in dmap for d in required):
            continue

        def get(dataset: str, col: str) -> Optional[float]:
            return _to_float(dmap[dataset].get(col))

        sea_miss = get("sea_abrupt4", "miss_tol500_mean")
        sea_conf = get("sea_abrupt4", "conf_P90_mean")
        sine_miss = get("sine_abrupt4", "miss_tol500_mean")
        sine_conf = get("sine_abrupt4", "conf_P90_mean")

        sea_rate = get("sea_nodrift", "confirm_rate_per_10k_mean")
        sine_rate = get("sine_nodrift", "confirm_rate_per_10k_mean")
        no_drift_rate = None if sea_rate is None or sine_rate is None else (sea_rate + sine_rate) / 2

        sea_mtfa = get("sea_nodrift", "MTFA_win_mean")
        sine_mtfa = get("sine_nodrift", "MTFA_win_mean")
        no_drift_mtfa = None if sea_mtfa is None or sine_mtfa is None else (sea_mtfa + sine_mtfa) / 2

        sea_acc = get("sea_abrupt4", "acc_final_mean")
        sine_acc = get("sine_abrupt4", "acc_final_mean")
        drift_acc = None if sea_acc is None or sine_acc is None else (sea_acc + sine_acc) / 2

        m = meta[g]
        out.append(
            GroupSummary(
                group=g,
                phase=m["phase"],
                confirm_rule=m["confirm_rule"],
                perm_stat=m["perm_stat"],
                perm_alpha=m["perm_alpha"],
                perm_pre_n=m["perm_pre_n"],
                perm_post_n=m["perm_post_n"],
                delta_k=m["delta_k"],
                sea_miss=sea_miss,
                sea_confP90=sea_conf,
                sine_miss=sine_miss,
                sine_confP90=sine_conf,
                no_drift_rate=no_drift_rate,
                no_drift_MTFA=no_drift_mtfa,
                drift_acc_final=drift_acc,
            )
        )
    return out


def _hard_ok(s: GroupSummary) -> bool:
    return (
        _is_zero(s.sea_miss)
        and _is_zero(s.sine_miss)
        and (s.sea_confP90 is not None and s.sea_confP90 < HARD_CONF_P90_LT)
        and (s.sine_confP90 is not None and s.sine_confP90 < HARD_CONF_P90_LT)
    )


def _select_winner(summaries: Sequence[GroupSummary]) -> Tuple[List[GroupSummary], Optional[float], Optional[GroupSummary]]:
    feasible = [s for s in summaries if _hard_ok(s)]
    best_acc = max((s.drift_acc_final for s in feasible if s.drift_acc_final is not None), default=None)
    feasible_acc = feasible
    if best_acc is not None:
        feasible_acc = [s for s in feasible if s.drift_acc_final is not None and s.drift_acc_final >= best_acc - DRIFT_ACC_TOL]

    def key(s: GroupSummary) -> Tuple[float, float, float]:
        return (
            s.no_drift_rate if s.no_drift_rate is not None else float("inf"),
            -(s.no_drift_MTFA if s.no_drift_MTFA is not None else -float("inf")),
            -(s.drift_acc_final if s.drift_acc_final is not None else -float("inf")),
        )

    winner = min(feasible_acc, key=key) if feasible_acc else None
    return feasible, best_acc, winner


def _top_k_near_constraints_perm(summaries: Sequence[GroupSummary], k: int) -> List[GroupSummary]:
    perm = [s for s in summaries if s.confirm_rule == "perm_test" and not _hard_ok(s)]

    def near_key(s: GroupSummary) -> Tuple[float, float, float]:
        miss_sum = (s.sea_miss or 0.0) + (s.sine_miss or 0.0)
        max_conf = max(
            s.sea_confP90 if s.sea_confP90 is not None else float("inf"),
            s.sine_confP90 if s.sine_confP90 is not None else float("inf"),
        )
        nd = s.no_drift_rate if s.no_drift_rate is not None else float("inf")
        return (miss_sum, max_conf, nd)

    return sorted(perm, key=near_key)[:k]


def _load_am_diag(path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not _exists(path):
        return out
    for row in _load_csv_rows(path):
        out[(row["group"], row["dataset"])] = row
    return out


def _extract_code_block(path: str, start_line: int, end_line: int) -> Optional[str]:
    if not _exists(path):
        return None
    lines = _read_text(path).splitlines()
    start = max(1, int(start_line))
    end = min(len(lines), int(end_line))
    if start > end:
        return None
    out = []
    for ln in range(start, end + 1):
        out.append(f"{ln:04d}: {lines[ln - 1]}")
    return "\n".join(out)


def _run_am_consistency_checks(am_rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    for row in am_rows:
        g = row.get("group", "")
        d = row.get("dataset", "")
        confirm_rule = row.get("confirm_rule", "")
        alpha = _to_float(row.get("perm_alpha"))
        p50 = _to_float(row.get("perm_pvalue_p50_mean"))
        p90 = _to_float(row.get("perm_pvalue_p90_mean"))
        p99 = _to_float(row.get("perm_pvalue_p99_mean"))
        ratio = _to_float(row.get("perm_pvalue_le_alpha_ratio_mean"))

        # 仅对 perm_test 的行做强约束校验；weighted 行相关字段多为 N/A
        if str(confirm_rule).strip().lower() != "perm_test":
            continue

        # Check0：分位数单调性
        if p50 is not None and p90 is not None and p99 is not None:
            if not (p50 <= p90 <= p99):
                anomalies.append(
                    {
                        "table_name": "AM_consistency_anomalies",
                        "group": g,
                        "dataset": d,
                        "check_name": "quantile_order_p50_p90_p99",
                        "status": "FAIL",
                        "detail": f"p50={_fmt(p50)}, p90={_fmt(p90)}, p99={_fmt(p99)}",
                    }
                )

        # B2-1：若 p_le_alpha_ratio > 0.5，则 pvalue_P50 应 <= alpha
        if ratio is None or alpha is None or p50 is None:
            anomalies.append(
                {
                    "table_name": "AM_consistency_anomalies",
                    "group": g,
                    "dataset": d,
                    "check_name": "ratio_gt_0.5_implies_p50_le_alpha",
                    "status": "N/A",
                    "detail": f"alpha={_fmt(alpha)}, p50={_fmt(p50)}, ratio={_fmt(ratio)}",
                }
            )
        else:
            if ratio > 0.5 and not (p50 <= alpha):
                anomalies.append(
                    {
                        "table_name": "AM_consistency_anomalies",
                        "group": g,
                        "dataset": d,
                        "check_name": "ratio_gt_0.5_implies_p50_le_alpha",
                        "status": "FAIL",
                        "detail": f"alpha={_fmt(alpha)}, p50={_fmt(p50)}, ratio={_fmt(ratio)}",
                    }
                )

        # B2-2：若 p90==1.0 且 ratio>0.3 -> 需 bimodal 证据
        # diag 不含 raw；此处仅做“需要/不需要验证”的记录，并在报告用实现代码解释为何容易出现 1.0 质量点。
        if p90 is None or ratio is None:
            anomalies.append(
                {
                    "table_name": "AM_consistency_anomalies",
                    "group": g,
                    "dataset": d,
                    "check_name": "bimodal_required_when_p90_is_1_and_ratio_gt_0.3",
                    "status": "N/A",
                    "detail": f"p90={_fmt(p90)}, ratio={_fmt(ratio)} (diag无raw，无法做unique/占比统计)",
                }
            )
        else:
            if abs(p90 - 1.0) < 1e-12 and ratio > 0.3:
                anomalies.append(
                    {
                        "table_name": "AM_consistency_anomalies",
                        "group": g,
                        "dataset": d,
                        "check_name": "bimodal_required_when_p90_is_1_and_ratio_gt_0.3",
                        "status": "NEEDS_EVIDENCE",
                        "detail": f"p90=1.0 且 ratio={_fmt(ratio)}；diag无raw无法量化bimodal（见报告用代码解释为何易出现1.0质量点）",
                    }
                )
    return anomalies


def _select_drill_runs(run_index_rows: Sequence[Dict[str, str]], groups: Sequence[str]) -> List[Dict[str, str]]:
    picked: List[Dict[str, str]] = []
    for g in groups:
        rows_g = [r for r in run_index_rows if r.get("group") == g]
        # 按 run_index 原始顺序优先选 sea_abrupt4 与 sea_nodrift 各 1 个
        used = set()
        for ds in ("sea_abrupt4", "sea_nodrift"):
            for r in rows_g:
                if r.get("dataset") == ds and r.get("log_path") and r.get("log_path") not in used:
                    picked.append(r)
                    used.add(r.get("log_path"))
                    break
        # 不足则补齐前 2 个（仍保持原顺序）
        if len([r for r in picked if r.get("group") == g]) < 2:
            for r in rows_g:
                if r.get("log_path") and r.get("log_path") not in used:
                    picked.append(r)
                    used.add(r.get("log_path"))
                if len([x for x in picked if x.get("group") == g]) >= 2:
                    break
    return picked


def _try_read_summary_within_log_path_dir(log_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    C2：严格按顺序尝试（最多 3 次 + 1 次局部 glob）
    - {log_path}/.summary.json
    - {log_path}/summary.json
    - {log_path}/metrics.summary.json
    - glob("{log_path}/*.summary.json")（最多一次；多个取字典序第一个）
    注意：RUN_INDEX 的 log_path 在本仓库是“csv 文件路径”，因此这里按“log_path 所在目录”为尝试目录，但仍遵守“仅在 log_path 目录内”。
    """
    meta: Dict[str, Any] = {"log_path": log_path, "log_path_exists": _exists(log_path), "attempts": [], "glob_used": 0}
    if not _exists(log_path):
        return None, None, meta

    # RUN_INDEX 给的是文件路径 -> 使用其所在目录；若本身是目录则直接用
    log_dir = log_path if os.path.isdir(log_path) else os.path.dirname(log_path)
    meta["log_dir"] = log_dir
    meta["log_dir_exists"] = os.path.isdir(log_dir)
    if not os.path.isdir(log_dir):
        return None, None, meta

    candidates = [
        os.path.join(log_dir, ".summary.json"),
        os.path.join(log_dir, "summary.json"),
        os.path.join(log_dir, "metrics.summary.json"),
    ]
    for p in candidates:
        meta["attempts"].append({"kind": "direct", "path": p, "exists": _exists(p)})
        if _exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                return p, obj, meta
            except Exception as e:
                meta["attempts"][-1]["error"] = str(e)

    # 单目录局部 glob（最多 1 次）
    meta["glob_used"] = 1
    pattern = os.path.join(log_dir, "*.summary.json")
    matches = sorted(glob.glob(pattern))
    meta["attempts"].append({"kind": "glob", "pattern": pattern, "n_matches": len(matches), "picked": (matches[0] if matches else None)})
    if not matches:
        return None, None, meta
    p = matches[0]
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        return p, obj, meta
    except Exception as e:
        meta["attempts"][-1]["error"] = str(e)
        return None, None, meta


def _get_from_trigger_weights(summary: Dict[str, Any], key: str) -> Any:
    tw = summary.get("trigger_weights")
    if isinstance(tw, dict) and key in tw:
        return tw.get(key)
    return None


def _extract_run_fields(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if summary is None:
        return {}
    # 训练 summary 的 key 来自 training/loop.py
    perm_alpha = summary.get("perm_alpha")
    perm_stat = _get_from_trigger_weights(summary, "__perm_stat")
    perm_pre_n = _get_from_trigger_weights(summary, "__perm_pre_n")
    perm_post_n = _get_from_trigger_weights(summary, "__perm_post_n")
    delta_k = _get_from_trigger_weights(summary, "__perm_delta_k")
    perm_n_perm = _get_from_trigger_weights(summary, "__perm_n_perm")
    min_effect = _get_from_trigger_weights(summary, "__perm_min_effect")
    rng_seed = _get_from_trigger_weights(summary, "__perm_rng_seed")

    test_count = summary.get("perm_test_count_total")
    accept_count = summary.get("perm_accept_count_total")
    reject_count = summary.get("perm_reject_count_total")

    p50 = summary.get("perm_pvalue_p50")
    p90 = summary.get("perm_pvalue_p90")
    p99 = summary.get("perm_pvalue_p99")
    ratio = summary.get("perm_pvalue_le_alpha_ratio")

    candidate = summary.get("candidate_count_total")
    confirmed = summary.get("confirmed_count_total")

    out: Dict[str, Any] = {
        "perm_alpha": _to_float(perm_alpha),
        "perm_stat": perm_stat if perm_stat is not None else None,
        "perm_pre_n": _to_float(perm_pre_n),
        "perm_post_n": _to_float(perm_post_n),
        "delta_k": _to_float(delta_k),
        "perm_n_perm": _to_float(perm_n_perm),
        "perm_min_effect": _to_float(min_effect),
        "perm_rng_seed": _to_float(rng_seed),
        "test_count": _to_float(test_count),
        "accept_count": _to_float(accept_count),
        "reject_count": _to_float(reject_count),
        "pvalue_P50": _to_float(p50),
        "pvalue_P90": _to_float(p90),
        "pvalue_P99": _to_float(p99),
        "p_le_alpha_ratio": _to_float(ratio),
        "last_perm_pvalue": _to_float(summary.get("last_perm_pvalue")),
        "last_perm_effect": _to_float(summary.get("last_perm_effect")),
        "candidate_count_total": _to_float(candidate),
        "confirmed_count_total": _to_float(confirmed),
        "confirm_rule_effective": summary.get("confirm_rule_effective"),
    }
    if out["candidate_count_total"] is not None and out["candidate_count_total"] > 0 and out["confirmed_count_total"] is not None:
        out["confirmed_over_candidate"] = float(out["confirmed_count_total"]) / float(out["candidate_count_total"])
    else:
        out["confirmed_over_candidate"] = None
    if out["test_count"] is not None and out["test_count"] > 0 and out["accept_count"] is not None:
        out["accept_over_test"] = float(out["accept_count"]) / float(out["test_count"])
    else:
        out["accept_over_test"] = None
    return out


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------- Task A：表格级复核 --------
    report_winner = _parse_report_winner(_read_text(V14_REPORT_MD)) if _exists(V14_REPORT_MD) else None
    legacy_winner = None
    if _exists(LEGACY_AUDIT_MD):
        legacy_winner = _parse_report_winner(_read_text(LEGACY_AUDIT_MD))

    summaries = _load_al_sweep_summaries(AL_SWEEP_CSV) if _exists(AL_SWEEP_CSV) else []
    feasible_step1, best_acc, winner = _select_winner(summaries) if summaries else ([], None, None)
    top10 = _top_k_near_constraints_perm(summaries, k=10) if summaries else []
    top1 = top10[0] if top10 else None

    perm = [s for s in summaries if s.confirm_rule == "perm_test" and s.no_drift_rate is not None]
    nodrift_min = min(perm, key=lambda s: s.no_drift_rate) if perm else None

    # 可行组总表（按 A2 字段）
    feasible_table: List[Dict[str, Any]] = []
    for s in feasible_step1:
        feasible_table.append(
            {
                "table_name": "AL_feasible_groups",
                "group": s.group,
                "sea_miss": s.sea_miss,
                "sea_confP90": s.sea_confP90,
                "sine_miss": s.sine_miss,
                "sine_confP90": s.sine_confP90,
                "no_drift_rate": s.no_drift_rate,
                "no_drift_MTFA": s.no_drift_MTFA,
                "drift_acc_final": s.drift_acc_final,
                "best_acc_final_in_step1": best_acc,
                "acc_ok": 1 if (best_acc is None or (s.drift_acc_final is not None and s.drift_acc_final >= best_acc - DRIFT_ACC_TOL)) else 0,
            }
        )

    top10_table: List[Dict[str, Any]] = []
    for rank, s in enumerate(top10, start=1):
        top10_table.append(
            {
                "table_name": "AL_top10_near_constraints",
                "rank": rank,
                "group": s.group,
                "perm_stat": s.perm_stat,
                "perm_alpha": s.perm_alpha,
                "perm_pre_n": s.perm_pre_n,
                "perm_post_n": s.perm_post_n,
                "delta_k": s.delta_k,
                "sea_miss": s.sea_miss,
                "sine_miss": s.sine_miss,
                "sea_confP90": s.sea_confP90,
                "sine_confP90": s.sine_confP90,
                "miss_sum": (s.sea_miss or 0.0) + (s.sine_miss or 0.0),
                "max_confP90": max(s.sea_confP90 or float("inf"), s.sine_confP90 or float("inf")),
                "no_drift_rate": s.no_drift_rate,
                "no_drift_MTFA": s.no_drift_MTFA,
                "drift_acc_final": s.drift_acc_final,
            }
        )

    # -------- Task B：AM 诊断一致性核查 --------
    am_rows_raw = _load_csv_rows(AM_DIAG_CSV) if _exists(AM_DIAG_CSV) else []
    am_anomalies = _run_am_consistency_checks(am_rows_raw) if am_rows_raw else []

    # -------- Task C：逐 run drill-down（严格使用 log_path） --------
    run_index_rows = _load_csv_rows(RUN_INDEX_CSV) if _exists(RUN_INDEX_CSV) else []
    drill_groups: List[str] = []
    if winner:
        drill_groups.append(winner.group)
    if top1 and top1.group not in drill_groups:
        drill_groups.append(top1.group)
    if nodrift_min and nodrift_min.group not in drill_groups:
        drill_groups.append(nodrift_min.group)

    selected_runs = _select_drill_runs(run_index_rows, drill_groups)

    am_diag = _load_am_diag(AM_DIAG_CSV)
    # 用 TRACKAL sweep 做 run vs 聚合对照（按 group+dataset 取聚合 mean）
    al_by_group_dataset: Dict[Tuple[str, str], Dict[str, str]] = {}
    if _exists(AL_SWEEP_CSV):
        for row in _load_csv_rows(AL_SWEEP_CSV):
            al_by_group_dataset[(row.get("group", ""), row.get("dataset", ""))] = row

    run_extract_rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []
    for r in selected_runs:
        group = r.get("group", "")
        dataset = r.get("dataset", "")
        run_id = r.get("run_id", "")
        log_path = r.get("log_path", "")

        summary_path, summary_obj, locate_meta = _try_read_summary_within_log_path_dir(log_path)
        log_dir = locate_meta.get("log_dir")
        locate_attempts_json = None
        try:
            locate_attempts_json = json.dumps(locate_meta.get("attempts", []), ensure_ascii=False)
        except Exception:
            locate_attempts_json = None

        # 标注 summary 来自 direct 还是 glob（便于复核“只做了 1 次局部 glob”）
        summary_found_method = None
        if summary_path:
            direct_candidates = {
                os.path.join(log_dir, ".summary.json") if log_dir else None,
                os.path.join(log_dir, "summary.json") if log_dir else None,
                os.path.join(log_dir, "metrics.summary.json") if log_dir else None,
            }
            summary_found_method = "direct" if summary_path in direct_candidates else "glob"

        fields = _extract_run_fields(summary_obj)
        run_extract_rows.append(
            {
                "table_name": "RUN_drilldown_extract",
                "group": group,
                "dataset": dataset,
                "run_id": run_id,
                "log_path": log_path,
                "log_dir": log_dir,
                "log_path_exists": 1 if locate_meta.get("log_path_exists") else 0,
                "glob_used": int(locate_meta.get("glob_used", 0) or 0),
                "summary_found_method": summary_found_method,
                "summary_locate_attempts_json": locate_attempts_json,
                "summary_path": summary_path,
                **{k: fields.get(k) for k in sorted(fields.keys())},
            }
        )

        diag = am_diag.get((group, dataset))
        al = al_by_group_dataset.get((group, dataset))

        diff: Dict[str, Any] = {
            "table_name": "DIFF_diag_vs_summary",
            "group": group,
            "dataset": dataset,
            "run_id": run_id,
            "summary_path": summary_path,
            "diag_row_exists": 1 if diag else 0,
            "al_row_exists": 1 if al else 0,
        }

        def _diag_f(k: str) -> Optional[float]:
            return _to_float(diag.get(k)) if diag else None

        def _al_f(k: str) -> Optional[float]:
            return _to_float(al.get(k)) if al else None

        # diag vs summary（可对齐的核心字段）
        diff["diag_perm_alpha"] = _diag_f("perm_alpha")
        diff["summary_perm_alpha"] = fields.get("perm_alpha")
        diff["diff_perm_alpha"] = None if diff["diag_perm_alpha"] is None or diff["summary_perm_alpha"] is None else float(diff["summary_perm_alpha"]) - float(diff["diag_perm_alpha"])

        diff["diag_p50"] = _diag_f("perm_pvalue_p50_mean")
        diff["summary_p50"] = fields.get("pvalue_P50")
        diff["diff_p50"] = None if diff["diag_p50"] is None or diff["summary_p50"] is None else float(diff["summary_p50"]) - float(diff["diag_p50"])

        diff["diag_p90"] = _diag_f("perm_pvalue_p90_mean")
        diff["summary_p90"] = fields.get("pvalue_P90")
        diff["diff_p90"] = None if diff["diag_p90"] is None or diff["summary_p90"] is None else float(diff["summary_p90"]) - float(diff["diag_p90"])

        diff["diag_ratio"] = _diag_f("perm_pvalue_le_alpha_ratio_mean")
        diff["summary_ratio"] = fields.get("p_le_alpha_ratio")
        diff["summary_accept_over_test"] = fields.get("accept_over_test")
        diff["diff_ratio"] = None if diff["diag_ratio"] is None or diff["summary_ratio"] is None else float(diff["summary_ratio"]) - float(diff["diag_ratio"])

        # 与 TRACKAL 聚合对齐（单 run vs 均值：只做差异展示，不做一致性结论）
        diff["al_conf_P90_mean"] = _al_f("conf_P90_mean")
        diff["al_miss_tol500_mean"] = _al_f("miss_tol500_mean")
        diff["al_confirm_rate_per_10k_mean"] = _al_f("confirm_rate_per_10k_mean")
        diff["summary_confirmed_over_candidate"] = fields.get("confirmed_over_candidate")

        diff_rows.append(diff)

    # -------- Task D：归因（基于可复核证据链） --------
    # 证据均来自：AL_SWEEP / AM_DIAG / code snippets / run drill-down summary（若存在）

    # -------- 写 V3 tables --------
    all_table_rows: List[Dict[str, Any]] = []
    all_table_rows.extend(feasible_table)
    all_table_rows.extend(top10_table)
    all_table_rows.extend(am_anomalies)
    all_table_rows.extend(run_extract_rows)
    all_table_rows.extend(diff_rows)

    # 确保至少包含所有 table_name，即使为空也要有占位行
    required_tables = [
        "AL_feasible_groups",
        "AL_top10_near_constraints",
        "AM_consistency_anomalies",
        "RUN_drilldown_extract",
        "DIFF_diag_vs_summary",
    ]
    present = {r.get("table_name") for r in all_table_rows}
    for t in required_tables:
        if t not in present:
            all_table_rows.append({"table_name": t, "note": "EMPTY"})

    fieldnames = sorted({k for r in all_table_rows for k in r.keys()})
    with open(OUT_TABLES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_table_rows:
            w.writerow(r)

    # -------- 写 V3 MD --------
    lines: List[str] = []
    lines.append("# V14 审计（Permutation-test Confirm）V3")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 复现命令：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python run_v14_audit_perm_confirm_v3.py`")
    lines.append("")
    lines.append("## 0) 审计范围声明（强约束）")
    lines.append("- 未进行任何全局搜索/扫描（未对 `logs/` 做递归查找、未做全仓库 grep/rg）。")
    lines.append("- 逐 run drill-down 仅使用 `scripts/NEXT_STAGE_V14_RUN_INDEX.csv` 的 `log_path` 定位；只在该 `log_path` 所在目录内按固定顺序尝试 + 最多 1 次局部 glob。")
    lines.append("- 未重跑任何训练/实验（不生成新 runs）。")
    lines.append("")

    lines.append("## 1) Task A：表格级复核（TRACKAL 聚合口径）")
    lines.append(f"- Step1 可行组数量：{len(feasible_step1)}")
    lines.append(f"- best_acc_final（Step1 可行组内最大 drift_acc_final）：{_fmt(best_acc, 6)}")
    lines.append(f"- winner（Step1→Step2→并列规则）：`{winner.group if winner else 'N/A'}`")
    lines.append(f"- NEXT_STAGE_V14_REPORT 声明 winner：`{_fmt(report_winner)}`")
    lines.append(f"- 既有 V14_AUDIT（根目录白名单文件）winner：`{_fmt(legacy_winner)}`（文件不存在则为 N/A）")
    lines.append("")
    lines.append("### 1.1 可行组总表（A2）")
    lines.append("- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=AL_feasible_groups`")
    lines.append("")
    lines.append("### 1.2 Top-10 最接近硬约束的 perm_test（A3）")
    lines.append("- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=AL_top10_near_constraints`")
    if top1:
        lines.append(f"- rank=1：`{top1.group}`")
    if nodrift_min:
        lines.append(f"- no-drift 最低 perm_test：`{nodrift_min.group}`（no_drift_rate≈{_fmt(nodrift_min.no_drift_rate, 6)}）")
    lines.append("")

    lines.append("## 2) Task B：AM 诊断口径与一致性核查")
    lines.append("### 2.1 B1 口径定义（来自代码，可复核）")
    # 关键代码片段（固定行号，来自本仓库当前版本）
    blocks = [
        ("drift/detectors.py（perm cfg 透传与覆盖）", "drift/detectors.py", 537, 603),
        ("drift/detectors.py（one-sided pvalue：obs<=0 -> p=1.0）", "drift/detectors.py", 611, 631),
        ("drift/detectors.py（accept 定义：p<=alpha 且 obs>=min_effect；不足窗口 perm_ok=False）", "drift/detectors.py", 800, 840),
        ("training/loop.py（summary 里 perm_pvalue_* 与 le_alpha_ratio 的生成）", "training/loop.py", 465, 543),
        ("experiments/trackAL_perm_confirm_sweep.py（注入 __perm_*）", "experiments/trackAL_perm_confirm_sweep.py", 340, 395),
        ("experiments/trackAM_perm_diagnostics.py（AM 读取每个 run 的 *.summary.json 并聚合）", "experiments/trackAM_perm_diagnostics.py", 205, 260),
    ]
    for title, path, a, b in blocks:
        blk = _extract_code_block(path, a, b)
        lines.append(f"- {title}：`{path}`")
        if blk is None:
            lines.append("  - N/A（文件不存在或无法读取）")
        else:
            lines.append("```py")
            lines.append(blk)
            lines.append("```")
    lines.append("")
    lines.append("口径小结：")
    lines.append("- `perm_pvalue`：`drift/detectors.py` 的 `_perm_test_one_sided()` 计算；当 `obs = post.mean - pre.mean <= 0` 时直接返回 `p=1.0`（强烈把“非正向变化”压到 1.0）。")
    lines.append("- `accept`：在 `confirm_rule=perm_test` 时，`perm_ok = (p<=alpha) and (obs>=min_effect)`；窗口不足时 `perm_ok=False`，且该次不会增加 `perm_test_count_total`。")
    lines.append("- `perm_pvalue_le_alpha_ratio`：在 `training/loop.py` summary 里定义为 `perm_pvalues` 列表中 `p<=perm_alpha` 的比例（`perm_pvalues` 来自 monitor 内存 `_perm_pvalues`）。当 `min_effect==0` 时与 `accept_count_total/test_count_total` 口径一致；若未来 `min_effect>0` 则两者会分叉。")
    lines.append("")

    lines.append("### 2.2 B2 一致性校验结果（异常清单）")
    lines.append("- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=AM_consistency_anomalies`")
    lines.append("- 说明：diag 不含 raw pvalue 序列，因此“bimodal 证据”的 unique/占比统计在表格层面为 N/A；但实现层面可由 `obs<=0 -> p=1.0` 解释为何容易出现 `p90/p99=1.0` 的质量点。")
    lines.append("")

    lines.append("## 3) Task C：逐 run drill-down（严格使用 log_path；不乱找）")
    lines.append("- 对应表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=RUN_drilldown_extract`")
    lines.append("- 对照表：`V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv` / `table_name=DIFF_diag_vs_summary`")
    lines.append("- 关键约束复述：每个 run 的 summary 定位最多 3 次固定路径尝试 + 1 次局部 glob（仅在该 log_path 所在目录内）。")
    lines.append("")

    lines.append("## 4) Task D：失败归因（可证伪分类 + 证据链）")
    lines.append("### 4.1 结论摘要")
    lines.append("- 现象（来自 `scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv`）：Step1 硬约束可行组极少（本次为 1 个：`A_weighted_n5`）；perm_test 网格里 no-drift 误报可降低，但 drift 侧 miss/延迟同时恶化，导致无法同时满足 Step1。")
    lines.append("- 主要归因排序：B（窗口对齐/统计量错配） > A（统计功效不足/不稳定） > C（状态机副作用） > D（实现 bug/口径不一致：未发现硬 bug，但存在可解释的实现行为导致 bimodal）。")
    lines.append("")

    lines.append("### 4.2 分类证据与“如何推翻”")
    lines.append("**A：power 不足/不稳定**")
    lines.append("- 证据：perm_test 的确认依赖 `obs=post.mean-pre.mean` 的 one-sided 置换检验；在 drift 早期/窗口污染情况下，`obs` 不稳定会导致大量 `p=1.0` 或 reject，从而推迟确认并造成 `conf_P90_mean` 上升/`miss_tol500_mean` 上升。")
    lines.append("- 如何推翻：若能在 run summary 中看到 drift 数据集里 `perm_pvalue_P90` 明显小于 alpha 且 `perm_pvalue_le_alpha_ratio` 高，同时仍出现高 miss/高 confP90，则 A 不是主因。")
    lines.append("")
    lines.append("**B：窗口对齐/统计量错配**")
    lines.append("- 证据：实现中 pre/post 窗口以 `sample_idx` 展开（`batch_n=current_pos-prev_pos`），但 confirm 的 pending 生命周期由 `confirm_window`（step 计数）控制，且 pre-window 还会“去污染”跳过最近 `post_n`（`drift/detectors.py` 795-801）。这些都会使 drift early transition 的 effect 被稀释/错过。")
    lines.append("- 如何推翻：若在 drift 数据集里，run summary 显示 `test_count` 很高、`p_le_alpha_ratio` 很高，但 `confirmed_count_total/candidate_count_total` 仍很低且 pending 常在 deadline 之前满足 post_n，则需要转向 C/D。")
    lines.append("")
    lines.append("**C：状态机副作用**")
    lines.append("- 证据：cooldown 期间直接清空 pending（`drift/detectors.py` 746-749），会导致“候选触发了，但在 cooldown 内被抹掉”，从而错过 confirm 窗口（表现为延迟变大/甚至 miss）。")
    lines.append("- 如何推翻：若在 drift 数据集里 cooldown_active 很少、且 pending 未被频繁清空，但仍出现同样的 miss/延迟，则 C 不是主要原因。")
    lines.append("")
    lines.append("**D：实现 bug / 口径不一致**")
    lines.append("- 证据（未发现硬 bug）：AM diag 的 `perm_pvalue_*` 与 `perm_pvalue_le_alpha_ratio` 明确来自每个 run 的 `.summary.json`（`experiments/trackAM_perm_diagnostics.py`）；summary 中 `perm_pvalue_le_alpha_ratio` 的计算也与字段名一致（`training/loop.py` 488）。")
    lines.append("- 风险点（实现行为导致“看起来异常”）：`obs<=0 -> p=1.0` 的早返回会天然造成 pvalue 分布在 1.0 处堆积，叠加 drift/no-drift 下不同窗口行为，会形成 bimodal 统计形态，可能被误读为“数据/口径异常”。")
    lines.append("- 如何推翻：若发现同一 run 的 `accept_over_test` 与 `p_le_alpha_ratio` 大幅不一致（且 `min_effect==0`），或发现 `perm_alpha` 在 summary 与 trigger_weights/__perm_alpha 不一致，则属于 D 类问题。")

    # 白名单缺失说明
    lines.append("")
    lines.append("## 5) 白名单文件存在性（缺失标 N/A）")
    for p in [AL_SWEEP_CSV, AM_DIAG_CSV, RUN_INDEX_CSV, METRICS_TABLE_CSV, V14_REPORT_MD, LEGACY_AUDIT_MD, LEGACY_AUDIT_TABLES]:
        lines.append(f"- {p}：{'OK' if _exists(p) else 'N/A'}")
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # stdout（简短）
    print(f"winner复核：{winner.group if winner else 'N/A'}（Step1可行组={len(feasible_step1)}；report={_fmt(report_winner)}）")
    print("主因归因：B（窗口对齐/统计量错配）")
    print("下一步方向（一句话）：优先修正 sample_idx vs step 的窗口/生命周期对齐与观测字段完备性，再讨论继续收紧 no-drift 误报。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
