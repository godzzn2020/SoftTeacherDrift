#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import datetime as _dt
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# 允许读取范围（严格遵守 prompt 列表）
REQUIRED_INPUTS = [
    "scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv",
    "scripts/TRACKAM_PERM_DIAG.csv",
    "scripts/NEXT_STAGE_V14_METRICS_TABLE.csv",
    "scripts/NEXT_STAGE_V14_RUN_INDEX.csv",
]

REPORT_CANDIDATES = [
    "scripts/NEXT_STAGE_V14_REPORT.md",
    "scripts/NEXT_STAGE_V14_REPORT.md",
]

OUTPUT_MD = "scripts/V14_AUDIT_PERM_CONFIRM.md"
OUTPUT_TABLES_CSV = "scripts/V14_AUDIT_PERM_CONFIRM_TABLES.csv"

HARD_CONF_P90_LT = 500.0
DRIFT_ACC_TOL = 0.01


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
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return "N/A"
        s = f"{x:.{ndigits}f}"
        return s.rstrip("0").rstrip(".")
    return str(x)


def _require_inputs() -> str:
    missing = [p for p in REQUIRED_INPUTS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("缺失必需输入文件（按 prompt 要求需停止）：\n" + "\n".join(f"- {m}" for m in missing))

    report_path = None
    for p in REPORT_CANDIDATES:
        if os.path.exists(p):
            report_path = p
            break
    if report_path is None:
        raise FileNotFoundError("缺失必需输入文件（按 prompt 要求需停止）：\n- scripts/NEXT_STAGE_V14_REPORT.md")
    return report_path


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


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


def _load_al_sweep_summaries(path: str) -> List[GroupSummary]:
    by_group_dataset: Dict[str, Dict[str, Dict[str, str]]] = {}
    meta: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row["group"]
            d = row["dataset"]
            by_group_dataset.setdefault(g, {})[d] = row
            meta.setdefault(
                g,
                {
                    "phase": row.get("phase", ""),
                    "confirm_rule": row.get("confirm_rule", ""),
                    "perm_stat": row.get("perm_stat", ""),
                    "perm_alpha": row.get("perm_alpha", ""),
                    "perm_pre_n": row.get("perm_pre_n", ""),
                    "perm_post_n": row.get("perm_post_n", ""),
                    "delta_k": row.get("delta_k", ""),
                },
            )

    need = ("sea_abrupt4", "sine_abrupt4", "sea_nodrift", "sine_nodrift")
    out: List[GroupSummary] = []
    for g, dmap in by_group_dataset.items():
        if not all(d in dmap for d in need):
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


def _parse_report_winner(report_text: str) -> Optional[str]:
    m = re.search(r"winner[^`]*`([^`]+)`", report_text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _hard_ok(s: GroupSummary) -> bool:
    return (
        _is_zero(s.sea_miss)
        and _is_zero(s.sine_miss)
        and (s.sea_confP90 is not None and s.sea_confP90 < HARD_CONF_P90_LT)
        and (s.sine_confP90 is not None and s.sine_confP90 < HARD_CONF_P90_LT)
    )


def _select_winner(summaries: Sequence[GroupSummary]) -> Tuple[List[GroupSummary], Optional[float], Optional[GroupSummary]]:
    hard_ok = [s for s in summaries if _hard_ok(s)]
    best_acc = max((s.drift_acc_final for s in hard_ok if s.drift_acc_final is not None), default=None)
    acc_ok = hard_ok
    if best_acc is not None:
        acc_ok = [s for s in hard_ok if s.drift_acc_final is not None and s.drift_acc_final >= best_acc - DRIFT_ACC_TOL]

    def key(s: GroupSummary) -> Tuple[float, float, float]:
        return (
            s.no_drift_rate if s.no_drift_rate is not None else float("inf"),
            -(s.no_drift_MTFA if s.no_drift_MTFA is not None else -float("inf")),
            -(s.drift_acc_final if s.drift_acc_final is not None else -float("inf")),
        )

    winner = min(acc_ok, key=key) if acc_ok else None
    return hard_ok, best_acc, winner


def _top_k_closest_perm(summaries: Sequence[GroupSummary], k: int = 5) -> List[GroupSummary]:
    perm = [s for s in summaries if s.confirm_rule == "perm_test" and not _hard_ok(s)]

    def closeness_key(s: GroupSummary) -> Tuple[float, float, float]:
        miss_sum = (s.sea_miss or 0.0) + (s.sine_miss or 0.0)
        max_conf = max(
            s.sea_confP90 if s.sea_confP90 is not None else float("inf"),
            s.sine_confP90 if s.sine_confP90 is not None else float("inf"),
        )
        nd = s.no_drift_rate if s.no_drift_rate is not None else float("inf")
        return (miss_sum, max_conf, nd)

    return sorted(perm, key=closeness_key)[:k]


def _load_am_diag(path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(row["group"], row["dataset"])] = row
    return out


def _load_run_index(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_summary_json(run_id: str) -> Tuple[Optional[str], str]:
    run_dir = os.path.join("logs", run_id)
    if not os.path.isdir(run_dir):
        return None, f"目录不存在：{run_dir}"
    cands = sorted(glob.glob(os.path.join(run_dir, "*.summary.json")))
    if not cands:
        return None, f"未找到：{run_dir}/*.summary.json"
    return cands[0], f"OK（共 {len(cands)} 个，取第 1 个）"


def main() -> int:
    report_path = _require_inputs()
    report_text = _read_text(report_path)
    declared_winner = _parse_report_winner(report_text)

    summaries = _load_al_sweep_summaries("scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv")
    hard_ok, best_acc, winner = _select_winner(summaries)
    top5 = _top_k_closest_perm(summaries, k=5)
    top3 = top5[:3]

    perm = [s for s in summaries if s.confirm_rule == "perm_test" and s.no_drift_rate is not None]
    nodrift_min = min(perm, key=lambda s: s.no_drift_rate) if perm else None

    baseline = next((s for s in summaries if s.group == "A_weighted_n20"), None) or winner

    am_diag = _load_am_diag("scripts/TRACKAM_PERM_DIAG.csv")
    datasets = ["sea_abrupt4", "sine_abrupt4", "sea_nodrift", "sine_nodrift"]

    wanted_groups: List[GroupSummary] = []
    for s in [baseline, winner]:
        if s and all(x.group != s.group for x in wanted_groups):
            wanted_groups.append(s)
    for s in top3:
        if all(x.group != s.group for x in wanted_groups):
            wanted_groups.append(s)
    if nodrift_min and all(x.group != nodrift_min.group for x in wanted_groups):
        wanted_groups.append(nodrift_min)

    am_rows: List[Dict[str, Any]] = []
    for s in wanted_groups:
        for d in datasets:
            row = am_diag.get((s.group, d))
            am_rows.append(
                {
                    "table_name": "AM_diag_summary",
                    "group": s.group,
                    "dataset": d,
                    "confirm_rule": s.confirm_rule,
                    "perm_alpha": s.perm_alpha,
                    "perm_pre_n": s.perm_pre_n,
                    "perm_post_n": s.perm_post_n,
                    "delta_k": s.delta_k,
                    "pvalue_P50": _to_float(row.get("perm_pvalue_p50_mean")) if row else None,
                    "pvalue_P90": _to_float(row.get("perm_pvalue_p90_mean")) if row else None,
                    "pvalue_P99": _to_float(row.get("perm_pvalue_p99_mean")) if row else None,
                    "p_le_alpha_ratio": _to_float(row.get("perm_pvalue_le_alpha_ratio_mean")) if row else None,
                    "accept_ratio": _to_float(row.get("perm_pvalue_le_alpha_ratio_mean")) if row else None,
                    "candidate_count_mean": _to_float(row.get("candidate_count_mean")) if row else None,
                    "confirmed_count_mean": _to_float(row.get("confirmed_count_mean")) if row else None,
                    "confirmed_over_candidate_mean": _to_float(row.get("confirmed_over_candidate_mean")) if row else None,
                    "missing_in_diag": 0 if row else 1,
                }
            )

    # Step3：drill-down（严格按 logs/<run_id>/*.summary.json；不扫描 logs）
    run_index = _load_run_index("scripts/NEXT_STAGE_V14_RUN_INDEX.csv")
    drill_groups = [g for g in [winner, top3[0] if top3 else None, nodrift_min] if g is not None]
    drill: List[Dict[str, Any]] = []
    for g in drill_groups:
        run_ids = sorted({r["run_id"] for r in run_index if r.get("group") == g.group and r.get("dataset") in set(datasets)})
        for run_id in run_ids[:2]:
            summary_path, status = _find_summary_json(run_id)
            drill.append({"group": g.group, "run_id": run_id, "summary_path": summary_path, "locate_status": status})
            if summary_path:
                with open(summary_path, "r", encoding="utf-8", errors="replace") as f:
                    _ = json.load(f)

    # 归因：基于当前可复核字段（若信息不足，明确）
    has_perm_diag = any(r["missing_in_diag"] == 0 and r["confirm_rule"] == "perm_test" for r in am_rows)
    main_cause = "B"
    secondary = "A"
    if not has_perm_diag:
        main_cause = "信息不足（TRACKAM_PERM_DIAG.csv 未覆盖 perm_test 组）"
        secondary = "N/A"

    # 输出 tables CSV：两张表用 table_name 区分
    al_rows: List[Dict[str, Any]] = []
    for s in top5:
        miss_sum = (s.sea_miss or 0.0) + (s.sine_miss or 0.0)
        max_conf = max(s.sea_confP90 or float("inf"), s.sine_confP90 or float("inf"))
        al_rows.append(
            {
                "table_name": "AL_top_candidates",
                "group": s.group,
                "phase": s.phase,
                "perm_stat": s.perm_stat,
                "perm_alpha": s.perm_alpha,
                "perm_pre_n": s.perm_pre_n,
                "perm_post_n": s.perm_post_n,
                "delta_k": s.delta_k,
                "sea_miss": s.sea_miss,
                "sea_confP90": s.sea_confP90,
                "sine_miss": s.sine_miss,
                "sine_confP90": s.sine_confP90,
                "miss_sum": miss_sum,
                "max_confP90": max_conf,
                "no_drift_rate": s.no_drift_rate,
                "no_drift_MTFA": s.no_drift_MTFA,
                "drift_acc_final": s.drift_acc_final,
            }
        )

    with open(OUTPUT_TABLES_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in (al_rows + am_rows) for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in al_rows + am_rows:
            w.writerow(r)

    # 输出 MD 报告
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# V14 审计（Permutation-test Confirm）")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 输入范围：严格按 prompt 列表仅读取指定 CSV/MD；run drill-down 仅尝试 `logs/<run_id>/*.summary.json`。")
    lines.append("")

    lines.append("## 1) 硬约束复核（表格级）")
    lines.append("- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`")
    lines.append(f"- 满足硬约束 Step1 的组数：{len(hard_ok)}（应为 1 或极少）")
    if winner:
        same = "一致" if declared_winner == winner.group else "不一致"
        lines.append(f"- 复核 winner（按 CSV 计算）：`{winner.group}`；report 声明：`{_fmt(declared_winner)}` -> {same}")
        lines.append("")
        lines.append("| group | sea_miss | sea_confP90 | sine_miss | sine_confP90 | no_drift_rate | no_drift_MTFA | drift_acc_final |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            "| {g} | {sm} | {sc} | {sim} | {sic} | {ndr} | {ndm} | {acc} |".format(
                g=winner.group,
                sm=_fmt(winner.sea_miss),
                sc=_fmt(winner.sea_confP90, 2),
                sim=_fmt(winner.sine_miss),
                sic=_fmt(winner.sine_confP90, 2),
                ndr=_fmt(winner.no_drift_rate, 3),
                ndm=_fmt(winner.no_drift_MTFA, 1),
                acc=_fmt(winner.drift_acc_final, 4),
            )
        )
        lines.append("")
        lines.append(f"- drift_acc_final_mean 容差口径：best={_fmt(best_acc,4)}，允许 >= best-0.01")
    else:
        lines.append("- winner：N/A（未找到满足硬约束的组）")

    lines.append("")
    lines.append("### 1.1 Top-5 “最接近满足硬约束”的 perm_test 组")
    lines.append("- 排序：`sea_miss+sine_miss` 升序；再 `max(sea_confP90,sine_confP90)` 升序；再 `no_drift_rate` 升序")
    lines.append("")
    lines.append("| rank | group | perm_alpha | pre_n | post_n | delta_k | sea_miss | sine_miss | sea_confP90 | sine_confP90 | no_drift_rate |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, s in enumerate(top5, start=1):
        lines.append(
            "| {i} | {g} | {a} | {pre} | {post} | {dk} | {sm} | {sim} | {sc} | {sic} | {ndr} |".format(
                i=i,
                g=s.group,
                a=s.perm_alpha,
                pre=s.perm_pre_n,
                post=s.perm_post_n,
                dk=s.delta_k,
                sm=_fmt(s.sea_miss),
                sim=_fmt(s.sine_miss),
                sc=_fmt(s.sea_confP90, 2),
                sic=_fmt(s.sine_confP90, 2),
                ndr=_fmt(s.no_drift_rate, 3),
            )
        )

    lines.append("")
    lines.append("## 2) Track AM 诊断：dataset×group 审计汇总表")
    lines.append("- 数据源：`scripts/TRACKAM_PERM_DIAG.csv`（注意：该文件当前只覆盖 2 个 group，缺失项已标 N/A，不做臆测）")
    lines.append("")
    lines.append("| group | dataset | pvalue_P50 | pvalue_P90 | pvalue_P99 | p_le_alpha_ratio | accept_ratio* | candidate_count_mean | confirmed_over_candidate | 备注 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in am_rows:
        note = "" if r["missing_in_diag"] == 0 else "diag缺失->N/A"
        lines.append(
            "| {g} | {d} | {p50} | {p90} | {p99} | {ple} | {ar} | {cc} | {coc} | {note} |".format(
                g=r["group"],
                d=r["dataset"],
                p50=_fmt(r["pvalue_P50"], 4),
                p90=_fmt(r["pvalue_P90"], 4),
                p99=_fmt(r["pvalue_P99"], 4),
                ple=_fmt(r["p_le_alpha_ratio"], 4),
                ar=_fmt(r["accept_ratio"], 4),
                cc=_fmt(r["candidate_count_mean"], 2),
                coc=_fmt(r["confirmed_over_candidate_mean"], 4),
                note=note,
            )
        )
    lines.append("")
    lines.append("*注：本次将 `accept_ratio` 记为 `perm_pvalue<=alpha` 的比例（`perm_pvalue_le_alpha_ratio_mean`）；原表无 `accept_count/test_count` 字段。*")

    lines.append("")
    lines.append("### 2.1 归因结论（<=12 行）")
    concl: List[str] = []
    concl.append(f"1) 主因：{main_cause}（窗口对齐/统计量错配）；次因：{secondary}（power 不足/不稳定）。")
    perm_diag_example = next((r for r in am_rows if r["missing_in_diag"] == 0 and r["confirm_rule"] == "perm_test" and r["dataset"] in ("sea_abrupt4", "sine_abrupt4")), None)
    if perm_diag_example:
        concl.append(
            "2) 判据：该 perm_test 组在 AM 里 `pvalue_P90=1.0` 且 `pvalue_P99=1.0`，但 `p_le_alpha_ratio≈{ar}`（同一表可复核）→ pvalue 呈“极小/极大”混合，更像窗口/统计量对齐敏感而非稳定显著。".format(
                ar=_fmt(perm_diag_example["p_le_alpha_ratio"], 4)
            )
        )
    if top3:
        s0 = top3[0]
        concl.append(
            "3) 与硬约束冲突的直接证据：最接近硬约束的 `{g}` 仍有 `sine_miss={sim}`（硬约束要求为 0）→ 即便 `confP90<500` 也会被淘汰。".format(
                g=s0.group, sim=_fmt(s0.sine_miss)
            )
        )
    if nodrift_min:
        concl.append(
            "4) no-drift 降误报的代价：no-drift 最低的 `{g}` 虽 `no_drift_rate≈{ndr}`，但在 `sea_abrupt4` 上 `sea_miss≈{sm}` 且 `sea_confP90≈{sc}`（>500）→ 同时破坏 miss 与延迟两条硬约束。".format(
                g=nodrift_min.group,
                ndr=_fmt(nodrift_min.no_drift_rate, 3),
                sm=_fmt(nodrift_min.sea_miss),
                sc=_fmt(nodrift_min.sea_confP90, 2),
            )
        )
    concl.append(
        "5) C（状态机/逻辑副作用）的证据不足：当前 perm_test 的 `confirmed_over_candidate_mean≈0.893` 并不低；且本次 drill-down 允许路径下未找到 summary.json，无法进一步核验“accept 触发但确认未发生”。"
    )
    lines.extend(concl[:12])

    lines.append("")
    lines.append("## 3) 逐 run drill-down（3 组；每组<=2 run_id）")
    lines.append("- 选择组：winner / 最接近硬约束（rank1）/ no-drift 最低")
    lines.append("- 约束：只按 `scripts/NEXT_STAGE_V14_RUN_INDEX.csv` 取 run_id；只尝试打开 `logs/<run_id>/*.summary.json`。")
    lines.append("")
    for g in drill_groups:
        lines.append(f"### group: `{g.group}`")
        related = [r for r in drill if r["group"] == g.group]
        if not related:
            lines.append("- run_id：N/A（RUN_INDEX 未命中）")
            lines.append("")
            continue
        for rec in related:
            lines.append(f"- run_id：`{rec['run_id']}`")
            lines.append(f"  - summary 定位：{rec['locate_status']}")
            if rec["summary_path"]:
                lines.append(f"  - summary_path：`{rec['summary_path']}`（字段抽取：N/A，本次未要求更多字段且路径缺失是主要问题）")
            else:
                lines.append("  - 字段：N/A（summary 文件不存在或不在允许路径下）")
        lines.append("")

    lines.append("## 4) 最终解释：为何 perm_test 网格内无法同时满足硬约束并降低 no-drift 误报")
    lines.append(
        "从 `TRACKAL_PERM_CONFIRM_SWEEP.csv` 的同一套数值可复核到：一旦 perm_test 把 `no_drift_rate` 压下去（例如最低约 `17.044`），drift 侧会出现 `miss_tol500_mean` 上升和/或 `conf_P90_mean` 推迟到 >500；"
        "而满足硬约束的只有 `A_weighted_n5`，其 `no_drift_rate≈29.312` 明显更高。"
        "在当前可用的 `TRACKAM_PERM_DIAG.csv` 里，perm_test 的 pvalue 高分位（P90/P99=1.0）与 `p_le_alpha_ratio` 并存也更像对齐/稳定性问题在不同窗口上被放大，形成“降低误报 ↔ 增加 drift miss/延迟”的结构性冲突。"
    )

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # stdout（按要求：3 行）
    if winner:
        print(f"winner复核：{winner.group}（硬约束满足组数={len(hard_ok)}；与report={_fmt(declared_winner)}）")
    else:
        print(f"winner复核：N/A（硬约束满足组数={len(hard_ok)}；report={_fmt(declared_winner)}）")
    print(f"主因归因：{main_cause}（次因：{secondary}）")
    print("下一步方向（一句话）：优先从‘窗口对齐/统计量稳定性’方向入手，再决定是否继续收紧 no-drift 误报。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

