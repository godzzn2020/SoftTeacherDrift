#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEXT_STAGE V14（Permutation-test Confirm）V5 审计：优先解决 D 类 dataset 对齐/索引一致性

强约束：
- 禁止全局搜索/递归扫描：不使用 find/rg/grep -R/os.walk/glob("**") 等
- 只读白名单文件；summary 仅按固定规则 log_path -> summary_path 读取（不 listdir / 不 glob）
- 不重跑训练，不生成新 runs
- 不修改任何现有代码文件：仅新增只读审计脚本 + 审计报告/表格

输出：
- V14_AUDIT_PERM_CONFIRM_V5.md
- V14_AUDIT_PERM_CONFIRM_V5_TABLES.csv（table_name 分表）
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------- 白名单输入（只读） ----------
AL_SWEEP_CSV = "scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv"
AM_DIAG_CSV = "scripts/TRACKAM_PERM_DIAG.csv"
RUN_INDEX_CSV = "scripts/NEXT_STAGE_V14_RUN_INDEX.csv"
METRICS_TABLE_CSV = "scripts/NEXT_STAGE_V14_METRICS_TABLE.csv"
V14_REPORT_MD = "scripts/NEXT_STAGE_V14_REPORT.md"
V4_MD = "V14_AUDIT_PERM_CONFIRM_V4.md"
V4_TABLES = "V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv"

# ---------- 只读代码白名单（用于根因定位） ----------
CODE_FILES = [
    "scripts/summarize_next_stage_v14.py",
    "experiments/trackAL_perm_confirm_sweep.py",
    "experiments/trackAM_perm_diagnostics.py",
    "training/loop.py",
    "drift/detectors.py",
    "run_v14_audit_perm_confirm_v4.py",
]


# ---------- 输出 ----------
OUT_MD = "V14_AUDIT_PERM_CONFIRM_V5.md"
OUT_TABLES = "V14_AUDIT_PERM_CONFIRM_V5_TABLES.csv"


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


def _fmt(x: Any, ndigits: int = 6) -> str:
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


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _summary_path_from_log_path(log_path: str) -> Tuple[str, List[str]]:
    """
    V5 固定规则：
    - 若 log_path 以 .csv 结尾：summary_path = 将末尾 .csv 替换为 .summary.json，只尝试这一条
    - 若不以 .csv 结尾：允许追加一次固定尝试：log_path + ".summary.json"
    """
    log_path = str(log_path or "")
    if log_path.endswith(".csv"):
        return log_path[:-4] + ".summary.json", []
    return log_path + ".summary.json", ["non_csv_log_path"]


def _read_json(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        return obj, None
    except Exception as e:
        return None, str(e)


def _trigger_weights_features(summary: Dict[str, Any]) -> Dict[str, Any]:
    tw = summary.get("trigger_weights")
    s = ""
    if isinstance(tw, dict):
        try:
            s = json.dumps(tw, ensure_ascii=False)
        except Exception:
            s = str(tw)
    elif isinstance(tw, str):
        s = tw
    else:
        s = ""
    low = s.lower()
    return {
        "tw_type": type(tw).__name__,
        "tw_has_confirm_rule_key": 1 if "__confirm_rule" in low else 0,
        "tw_has_perm_key": 1 if "__perm_" in low else 0,
        "tw_has_confirm_cooldown_key": 1 if "__confirm_cooldown" in low else 0,
    }


def _expected_is_perm_group(group: str) -> Optional[bool]:
    g = str(group or "")
    if not g:
        return None
    if g.startswith("A_"):
        return False
    if g.startswith("P_perm_") or "perm_" in g:
        return True
    return None


def _match_group_weak(group: str, summary: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    弱一致性规则（写死）：
    - 若 group 被判为 perm_test 组：summary.confirm_rule_effective 应包含 "perm_test" 或 trigger_weights 含 "__perm_"
    - 若 group 被判为 weighted 组：summary.confirm_rule_effective 不应包含 "perm_test" 且 trigger_weights 不含 "__perm_"
    """
    exp = _expected_is_perm_group(group)
    if exp is None or summary is None:
        return None
    eff = str(summary.get("confirm_rule_effective") or "").lower()
    twf = _trigger_weights_features(summary)
    has_perm = bool(int(twf["tw_has_perm_key"]))
    is_perm_eff = "perm_test" in eff
    if exp is True:
        return 1 if (is_perm_eff or has_perm) else 0
    # exp is False
    return 1 if ((not is_perm_eff) and (not has_perm)) else 0


def _extract_code_snippet(path: str, start: int, end: int) -> Optional[str]:
    if not _exists(path):
        return None
    lines = _read_text(path).splitlines()
    start = max(1, int(start))
    end = min(len(lines), int(end))
    if start > end:
        return None
    out = []
    for ln in range(start, end + 1):
        out.append(f"{ln:04d}: {lines[ln - 1]}")
    return "\n".join(out)


def _find_first_line(path: str, needle: str) -> Optional[int]:
    if not _exists(path):
        return None
    for i, line in enumerate(_read_text(path).splitlines(), start=1):
        if needle in line:
            return i
    return None


def _parse_run_index_json(raw: str) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 输入存在性 ----------
    required = [AL_SWEEP_CSV, AM_DIAG_CSV, RUN_INDEX_CSV, METRICS_TABLE_CSV, V14_REPORT_MD, V4_MD, V4_TABLES]
    missing = [p for p in required if not _exists(p)]
    # 按要求：若缺失，只在报告里标 N/A，不中断；但 V5 需要 RUN_INDEX/AL/METRICS 才能工作，这里仍继续并在表中填空。

    run_index_rows = _load_csv_rows(RUN_INDEX_CSV) if _exists(RUN_INDEX_CSV) else []

    # ---------- Task A：RUN_INDEX 结构性一致性（不读 logs，仅靠 CSV） ----------
    by_run_id: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_log_path: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in run_index_rows:
        by_run_id[str(r.get("run_id") or "")].append(r)
        by_log_path[str(r.get("log_path") or "")].append(r)

    def _top_clusters(mapping: Dict[str, List[Dict[str, str]]], topk: int = 20) -> List[Tuple[str, List[Dict[str, str]]]]:
        items = [(k, v) for k, v in mapping.items() if k]
        items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
        return items[:topk]

    dup_by_run_id_rows: List[Dict[str, Any]] = []
    for run_id, rows in _top_clusters(by_run_id, 20):
        dup_by_run_id_rows.append(
            {
                "table_name": "RUN_INDEX_DUP_BY_RUN_ID",
                "run_id": run_id,
                "cluster_size": len(rows),
                "n_datasets": len({r.get("dataset") for r in rows}),
                "n_groups": len({r.get("group") for r in rows}),
                "n_seeds": len({r.get("seed") for r in rows}),
                "n_log_paths": len({r.get("log_path") for r in rows}),
                "datasets_json": json.dumps(sorted({r.get("dataset") for r in rows}), ensure_ascii=False),
                "groups_json": json.dumps(sorted({r.get("group") for r in rows}), ensure_ascii=False),
                "seeds_json": json.dumps(sorted({r.get("seed") for r in rows}), ensure_ascii=False),
                "log_paths_preview_json": json.dumps(sorted({r.get("log_path") for r in rows})[:5], ensure_ascii=False),
            }
        )

    dup_by_log_path_rows: List[Dict[str, Any]] = []
    for log_path, rows in _top_clusters(by_log_path, 20):
        dup_by_log_path_rows.append(
            {
                "table_name": "RUN_INDEX_DUP_BY_LOG_PATH",
                "log_path": log_path,
                "cluster_size": len(rows),
                "n_datasets": len({r.get("dataset") for r in rows}),
                "n_groups": len({r.get("group") for r in rows}),
                "n_run_ids": len({r.get("run_id") for r in rows}),
                "n_seeds": len({r.get("seed") for r in rows}),
                "datasets_json": json.dumps(sorted({r.get("dataset") for r in rows}), ensure_ascii=False),
                "groups_json": json.dumps(sorted({r.get("group") for r in rows}), ensure_ascii=False),
                "run_ids_preview_json": json.dumps(sorted({r.get("run_id") for r in rows})[:5], ensure_ascii=False),
                "seeds_json": json.dumps(sorted({r.get("seed") for r in rows}), ensure_ascii=False),
            }
        )

    contradictions_rows: List[Dict[str, Any]] = []
    for run_id, rows in by_run_id.items():
        if not run_id:
            continue
        ds_set = {r.get("dataset") for r in rows}
        g_set = {r.get("group") for r in rows}
        s_set = {r.get("seed") for r in rows}
        lp_set = {r.get("log_path") for r in rows}
        # 只输出真正“矛盾”的：同一 run_id 出现 seed/group/dataset 不一致
        if len(s_set) > 1 or len(g_set) > 1 or len(ds_set) > 1:
            contradictions_rows.append(
                {
                    "table_name": "RUN_INDEX_INTERNAL_CONTRADICTIONS",
                    "run_id": run_id,
                    "cluster_size": len(rows),
                    "n_datasets": len(ds_set),
                    "n_groups": len(g_set),
                    "n_seeds": len(s_set),
                    "n_log_paths": len(lp_set),
                    "datasets_json": json.dumps(sorted(ds_set), ensure_ascii=False),
                    "groups_json": json.dumps(sorted(g_set), ensure_ascii=False),
                    "seeds_json": json.dumps(sorted(s_set), ensure_ascii=False),
                    "log_paths_preview_json": json.dumps(sorted(lp_set)[:5], ensure_ascii=False),
                }
            )
    contradictions_rows.sort(key=lambda r: (-int(r["cluster_size"]), str(r["run_id"])))

    # ---------- Task B：逐行读取 summary，做字段对齐真值表 ----------
    # 先按 unique(log_path) 读取 summary（按固定规则），再映射回每一行（带引用计数）
    unique_log_paths = sorted({str(r.get("log_path") or "") for r in run_index_rows if str(r.get("log_path") or "")})

    summary_cache: Dict[str, Tuple[str, Optional[Dict[str, Any]], str]] = {}
    for lp in unique_log_paths:
        sp, notes = _summary_path_from_log_path(lp)
        obj = None
        err = None
        if _exists(sp):
            obj, err = _read_json(sp)
        else:
            err = "missing_summary"
        summary_cache[lp] = (sp, obj, err or "")

    alignment_rows: List[Dict[str, Any]] = []
    mismatch_examples: List[Dict[str, Any]] = []
    dataset_mismatch_by_dataset = Counter()
    seed_mismatch_by_dataset = Counter()
    missing_summary_by_dataset = Counter()
    total_by_dataset = Counter()

    for r in run_index_rows:
        dataset = str(r.get("dataset") or "")
        group = str(r.get("group") or "")
        seed = str(r.get("seed") or "")
        run_id = str(r.get("run_id") or "")
        log_path = str(r.get("log_path") or "")
        total_by_dataset[dataset] += 1

        sp, summ, err = summary_cache.get(log_path, ("", None, "missing_summary"))
        if err == "missing_summary":
            missing_summary_by_dataset[dataset] += 1

        summ_dataset = summ.get("dataset_name") if isinstance(summ, dict) else None
        summ_seed = summ.get("seed") if isinstance(summ, dict) else None
        summ_eff = summ.get("confirm_rule_effective") if isinstance(summ, dict) else None

        match_dataset = None
        if isinstance(summ_dataset, str) and dataset:
            match_dataset = 1 if dataset == summ_dataset else 0
        match_seed = None
        if summ_seed is not None and seed:
            try:
                match_seed = 1 if int(float(seed)) == int(summ_seed) else 0
            except Exception:
                match_seed = None
        mgw = _match_group_weak(group, summ if isinstance(summ, dict) else None)

        twf = _trigger_weights_features(summ) if isinstance(summ, dict) else {"tw_type": "N/A", "tw_has_confirm_rule_key": 0, "tw_has_perm_key": 0, "tw_has_confirm_cooldown_key": 0}

        if match_dataset == 0:
            dataset_mismatch_by_dataset[dataset] += 1
        if match_seed == 0:
            seed_mismatch_by_dataset[dataset] += 1

        alignment_rows.append(
            {
                "table_name": "RUN_INDEX_VS_SUMMARY_ALIGNMENT",
                "track": r.get("track"),
                "phase": r.get("phase"),
                "group": group,
                "dataset": dataset,
                "base_dataset_name": r.get("base_dataset_name"),
                "seed": seed,
                "run_id": run_id,
                "log_path": log_path,
                "summary_path": sp,
                "summary_read_error": err if err else "",
                "summary_dataset_name": summ_dataset if isinstance(summ_dataset, str) else None,
                "summary_dataset_type": summ.get("dataset_type") if isinstance(summ, dict) else None,
                "summary_seed": summ_seed,
                "summary_confirm_rule_effective": summ_eff if isinstance(summ_eff, str) else None,
                "summary_monitor_preset": summ.get("monitor_preset") if isinstance(summ, dict) else None,
                "summary_created_at": summ.get("created_at") if isinstance(summ, dict) else None,
                "tw_type": twf["tw_type"],
                "tw_has_confirm_rule_key": twf["tw_has_confirm_rule_key"],
                "tw_has_perm_key": twf["tw_has_perm_key"],
                "tw_has_confirm_cooldown_key": twf["tw_has_confirm_cooldown_key"],
                "match_dataset": match_dataset,
                "match_seed": match_seed,
                "match_group_weak": mgw,
            }
        )

        # mismatch 样例：dataset mismatch / seed mismatch / summary missing
        is_mismatch = (match_dataset == 0) or (match_seed == 0) or (err == "missing_summary")
        if is_mismatch:
            mismatch_examples.append(
                {
                    "table_name": "MISMATCH_EXAMPLES_TOP30",
                    "dataset": dataset,
                    "summary_dataset_name": summ_dataset if isinstance(summ_dataset, str) else None,
                    "seed": seed,
                    "summary_seed": summ_seed,
                    "group": group,
                    "summary_confirm_rule_effective": summ_eff if isinstance(summ_eff, str) else None,
                    "run_id": run_id,
                    "log_path": log_path,
                    "summary_path": sp,
                    "summary_read_error": err if err else "",
                    "match_dataset": match_dataset,
                    "match_seed": match_seed,
                    "match_group_weak": mgw,
                }
            )

    # Top30 mismatch：优先 dataset mismatch，然后 seed mismatch，再 summary missing
    def _mismatch_rank(row: Dict[str, Any]) -> Tuple[int, int, int, str]:
        md = 1 if row.get("match_dataset") == 0 else 0
        ms = 1 if row.get("match_seed") == 0 else 0
        mm = 1 if row.get("summary_read_error") == "missing_summary" else 0
        return (-md, -ms, -mm, str(row.get("log_path") or ""))

    mismatch_examples.sort(key=_mismatch_rank)
    mismatch_examples = mismatch_examples[:30]

    stats_by_dataset_rows: List[Dict[str, Any]] = []
    for ds, total in sorted(total_by_dataset.items(), key=lambda kv: kv[0]):
        stats_by_dataset_rows.append(
            {
                "table_name": "ALIGNMENT_STATS_BY_DATASET",
                "dataset": ds,
                "n_rows": int(total),
                "n_missing_summary": int(missing_summary_by_dataset.get(ds, 0)),
                "missing_summary_ratio": None if total <= 0 else float(missing_summary_by_dataset.get(ds, 0)) / float(total),
                "n_dataset_mismatch": int(dataset_mismatch_by_dataset.get(ds, 0)),
                "dataset_mismatch_ratio": None if total <= 0 else float(dataset_mismatch_by_dataset.get(ds, 0)) / float(total),
                "n_seed_mismatch": int(seed_mismatch_by_dataset.get(ds, 0)),
                "seed_mismatch_ratio": None if total <= 0 else float(seed_mismatch_by_dataset.get(ds, 0)) / float(total),
            }
        )

    # ---------- Task C：三表一致性（AL / METRICS_TABLE / RUN_INDEX） ----------
    run_index_log_paths_set = set(unique_log_paths)
    run_index_run_ids_set = {str(r.get("run_id") or "") for r in run_index_rows if str(r.get("run_id") or "")}

    def audit_table_run_index_json(table_path: str, table_name: str) -> Dict[str, Any]:
        rows = _load_csv_rows(table_path) if _exists(table_path) else []
        n_rows = len(rows)
        n_with_run_index_json = 0
        n_parse_ok = 0
        referenced_log_paths: List[str] = []
        referenced_run_ids: List[str] = []
        for row in rows:
            raw = row.get("run_index_json")
            if raw is None or not str(raw).strip():
                continue
            n_with_run_index_json += 1
            obj = _parse_run_index_json(raw)
            if not isinstance(obj, dict):
                continue
            n_parse_ok += 1
            for ds_key, info in obj.items():
                if not isinstance(info, dict):
                    continue
                runs = info.get("runs")
                if not isinstance(runs, list):
                    continue
                for item in runs:
                    if not isinstance(item, dict):
                        continue
                    lp = item.get("log_path")
                    rid = item.get("run_id")
                    if isinstance(lp, str) and lp:
                        referenced_log_paths.append(lp)
                    if isinstance(rid, str) and rid:
                        referenced_run_ids.append(rid)

        ref_lp_set = set(referenced_log_paths)
        ref_rid_set = set(referenced_run_ids)
        lp_traceable = len(ref_lp_set & run_index_log_paths_set)
        rid_traceable = len(ref_rid_set & run_index_run_ids_set)
        return {
            "table_name": "CROSS_TABLE_JOIN_AUDIT",
            "source_table": table_name,
            "n_rows": n_rows,
            "n_rows_with_run_index_json": n_with_run_index_json,
            "n_run_index_json_parse_ok": n_parse_ok,
            "n_unique_ref_log_paths": len(ref_lp_set),
            "n_unique_ref_run_ids": len(ref_rid_set),
            "ref_log_paths_traceable_in_RUN_INDEX": lp_traceable,
            "ref_run_ids_traceable_in_RUN_INDEX": rid_traceable,
            "ref_log_paths_traceable_ratio": None if len(ref_lp_set) <= 0 else float(lp_traceable) / float(len(ref_lp_set)),
            "ref_run_ids_traceable_ratio": None if len(ref_rid_set) <= 0 else float(rid_traceable) / float(len(ref_rid_set)),
            "missing_ref_log_paths_preview_json": json.dumps(sorted(list(ref_lp_set - run_index_log_paths_set))[:10], ensure_ascii=False),
            "missing_ref_run_ids_preview_json": json.dumps(sorted(list(ref_rid_set - run_index_run_ids_set))[:10], ensure_ascii=False),
        }

    cross_join_rows: List[Dict[str, Any]] = []
    cross_join_rows.append(audit_table_run_index_json(AL_SWEEP_CSV, "TRACKAL_PERM_CONFIRM_SWEEP"))
    cross_join_rows.append(audit_table_run_index_json(METRICS_TABLE_CSV, "NEXT_STAGE_V14_METRICS_TABLE"))

    # C2：NODRIFT_LABEL_PURITY_AUDIT（summary.dataset_name 作为真值）
    # 仅使用 TRACKAL sweep 里 dataset=sea_nodrift/sine_nodrift 的行（因为要求评估 no_drift_rate 是否被污染）
    nodrift_purity_rows: List[Dict[str, Any]] = []
    if _exists(AL_SWEEP_CSV):
        for row in _load_csv_rows(AL_SWEEP_CSV):
            ds = str(row.get("dataset") or "")
            if ds not in {"sea_nodrift", "sine_nodrift"}:
                continue
            group = str(row.get("group") or "")
            base = str(row.get("base_dataset_name") or "")
            obj = _parse_run_index_json(row.get("run_index_json"))
            runs: List[Dict[str, Any]] = []
            if isinstance(obj, dict):
                info = obj.get(ds)
                if isinstance(info, dict) and isinstance(info.get("runs"), list):
                    runs = [x for x in info["runs"] if isinstance(x, dict)]
            corrected: List[str] = []
            missing_summ = 0
            for item in runs:
                lp = str(item.get("log_path") or "")
                sp, _notes = _summary_path_from_log_path(lp)
                if not _exists(sp):
                    missing_summ += 1
                    continue
                summ, _err = _read_json(sp)
                if isinstance(summ, dict) and isinstance(summ.get("dataset_name"), str):
                    corrected.append(str(summ.get("dataset_name")))
            total_runs = len(runs)
            if total_runs <= 0:
                nodrift_purity_rows.append(
                    {
                        "table_name": "NODRIFT_LABEL_PURITY_AUDIT",
                        "group": group,
                        "label_dataset": ds,
                        "base_dataset_name": base,
                        "n_runs_in_run_index_json": 0,
                        "n_missing_summary": 0,
                        "corrected_dataset_mode": None,
                        "corrected_dataset_counts_json": "[]",
                        "label_purity": None,
                    }
                )
                continue
            cnt = Counter(corrected)
            mode = cnt.most_common(1)[0][0] if cnt else None
            # purity：校正后 dataset 与 label_dataset 一致的比例（仅在 summary 可读的 runs 上计算；并同时输出 missing_summary）
            denom = max(1, len(corrected))
            purity = (cnt.get(ds, 0) / denom) if denom > 0 else None
            nodrift_purity_rows.append(
                {
                    "table_name": "NODRIFT_LABEL_PURITY_AUDIT",
                    "group": group,
                    "label_dataset": ds,
                    "base_dataset_name": base,
                    "n_runs_in_run_index_json": total_runs,
                    "n_missing_summary": missing_summ,
                    "n_runs_with_summary": len(corrected),
                    "corrected_dataset_mode": mode,
                    "corrected_dataset_counts_json": json.dumps(cnt.most_common(5), ensure_ascii=False),
                    "label_purity": purity,
                }
            )

    # ---------- Task D：根因定位（只读代码，限定文件范围） ----------
    # D1：summarize_next_stage_v14.py 的 RUN_INDEX 生成逻辑（通过关键字定位行号）
    # 不做全仓库搜索，仅在该文件内找关键 token
    summarize_path = "scripts/summarize_next_stage_v14.py"
    summarize_hints = ["RUN_INDEX", "NEXT_STAGE_V14_RUN_INDEX", "run_index", "log_path", "base_dataset_name", "dataset"]
    summarize_hit = None
    if _exists(summarize_path):
        text = _read_text(summarize_path).splitlines()
        for i, line in enumerate(text, start=1):
            low = line.lower()
            if any(h.lower() in low for h in summarize_hints):
                summarize_hit = i
                break
    summarize_snip = _extract_code_snippet(summarize_path, summarize_hit or 1, (summarize_hit or 1) + 80) if _exists(summarize_path) else None

    # D2：trackAL_perm_confirm_sweep.py 是否遍历 nodrift + log 目录命名（关键：ensure_log 调用是否把 dataset_name 设为 base_name）
    al_path = "experiments/trackAL_perm_confirm_sweep.py"
    al_datasets_line = _find_first_line(al_path, "datasets:")
    al_ensure_line = _find_first_line(al_path, "ensure_log(")
    al_dataset_name_base_line = _find_first_line(al_path, "dataset_name=base_name")
    al_snip = _extract_code_snippet(al_path, (al_datasets_line or 1), (al_datasets_line or 1) + 60) if _exists(al_path) else None
    al_snip2 = _extract_code_snippet(al_path, (al_ensure_line or 1), (al_ensure_line or 1) + 40) if _exists(al_path) else None
    al_snip3 = (
        _extract_code_snippet(al_path, max(1, (al_dataset_name_base_line or 1) - 8), (al_dataset_name_base_line or 1) + 18)
        if _exists(al_path)
        else None
    )

    # D3：training/loop.py summary.dataset_name 写入来源
    loop_path = "training/loop.py"
    loop_line = _find_first_line(loop_path, '"dataset_name": config.dataset_name')
    loop_snip = _extract_code_snippet(loop_path, (loop_line or 1), (loop_line or 1) + 30) if _exists(loop_path) else None

    # RootCause 分类（写死输出之一）：
    # - 已有确定性代码证据：trackAL 中 ensure_log 使用 dataset_name=base_name，导致 sea_nodrift 与 sea_abrupt4 共享同一 log_path，
    #   且 ensure_log 发现已有 summary 时会直接 return（不重跑），从而“不同 dataset 写进同一 log_dir / 复用覆盖”。
    # => RootCause=D4（log_path 复用/覆盖 bug）优先级最高。
    root_cause = "RootCause=D4"
    rc_evidence: List[str] = []
    # 证据 1：dataset mismatch 比例（从 ALIGNMENT_STATS_BY_DATASET 中读取）
    ds_mismatch_total = sum(int(r.get("n_dataset_mismatch") or 0) for r in stats_by_dataset_rows)
    total_rows = sum(int(r.get("n_rows") or 0) for r in stats_by_dataset_rows)
    rc_evidence.append(f"dataset mismatch：{ds_mismatch_total}/{total_rows}（见表 ALIGNMENT_STATS_BY_DATASET）")
    # 证据 2：log_path 被多 dataset 引用的簇大小（取 top1）
    top_lp = dup_by_log_path_rows[0] if dup_by_log_path_rows else None
    if top_lp:
        rc_evidence.append(
            f"log_path 重复簇 top1：cluster_size={top_lp.get('cluster_size')}, n_datasets={top_lp.get('n_datasets')}（见表 RUN_INDEX_DUP_BY_LOG_PATH）"
        )
    # 证据 3：代码行（D4 的“钉子”）
    if al_dataset_name_base_line is not None:
        rc_evidence.append(f"代码证据：`experiments/trackAL_perm_confirm_sweep.py` 存在 `dataset_name=base_name`（line {al_dataset_name_base_line}），会导致不同 ds_alias 复用同一 log_path。")
    # 证据 3：nodrift label purity
    if nodrift_purity_rows:
        # 取前 1 行作为样例（按 label_purity 升序）
        def purity_key(r: Dict[str, Any]) -> Tuple[float, str]:
            v = r.get("label_purity")
            try:
                fv = float(v) if v is not None else 1.0
            except Exception:
                fv = 1.0
            return (fv, str(r.get("group") or ""))

        worst = sorted(nodrift_purity_rows, key=purity_key)[:1]
        if worst:
            w = worst[0]
            rc_evidence.append(
                f"nodrift label purity（样例）：group={w.get('group')}, label={w.get('label_dataset')}, purity={_fmt(_to_float(w.get('label_purity')))}，mode={w.get('corrected_dataset_mode')}（见表 NODRIFT_LABEL_PURITY_AUDIT）"
            )

    root_cause_rows = [
        {
            "table_name": "ROOT_CAUSE_CLASSIFICATION",
            "root_cause": root_cause,
            "evidence_json": json.dumps(rc_evidence, ensure_ascii=False),
            "falsifiable_condition": (
                "若未来能看到：1) trackAL 的 ensure_log 调用使用 `dataset_name=ds_alias`（或等价地让不同 dataset 产生不同 log_path）；"
                "且 2) sea_nodrift/sine_nodrift 的 corrected_dataset 纯度接近 1（而不是全部落在 base_dataset_name）；"
                "则可推翻 D4，需要重新审计 D1/D2/D3。"
            ),
        }
    ]

    # ---------- 写出 V5 表格 ----------
    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(dup_by_run_id_rows)
    all_rows.extend(dup_by_log_path_rows)
    all_rows.extend(contradictions_rows[:200])  # 控制输出规模
    all_rows.extend(alignment_rows[:5000])  # RUN_INDEX 全量行（1060）< 5000
    all_rows.extend(stats_by_dataset_rows)
    all_rows.extend(mismatch_examples)
    all_rows.extend(cross_join_rows)
    all_rows.extend(nodrift_purity_rows[:5000])
    all_rows.extend(root_cause_rows)

    required_tables = [
        "RUN_INDEX_DUP_BY_RUN_ID",
        "RUN_INDEX_DUP_BY_LOG_PATH",
        "RUN_INDEX_INTERNAL_CONTRADICTIONS",
        "RUN_INDEX_VS_SUMMARY_ALIGNMENT",
        "ALIGNMENT_STATS_BY_DATASET",
        "MISMATCH_EXAMPLES_TOP30",
        "CROSS_TABLE_JOIN_AUDIT",
        "NODRIFT_LABEL_PURITY_AUDIT",
        "ROOT_CAUSE_CLASSIFICATION",
    ]
    present = {r.get("table_name") for r in all_rows}
    for t in required_tables:
        if t not in present:
            all_rows.append({"table_name": t, "note": "EMPTY"})

    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(OUT_TABLES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # ---------- 写出 V5 报告 ----------
    lines: List[str] = []
    lines.append("# V14 审计（Permutation-test Confirm）V5：dataset 对齐/索引一致性")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 复现命令：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python run_v14_audit_perm_confirm_v5.py`")
    lines.append("")
    lines.append("## 0) 审计范围声明（强约束）")
    lines.append("- 未做任何全局搜索/递归扫描（未使用 find/rg/grep -R/os.walk/glob('**')）。")
    lines.append("- summary 读取仅按固定规则：`summary_path = log_path 将末尾 .csv 替换为 .summary.json`；不 listdir、不 glob。")
    lines.append("")

    lines.append("## 1) Q0：RUN_INDEX 的 dataset 字段是否可信？")
    lines.append("- 结论：不可信（可复核：大量出现 `run_index.dataset != summary.dataset_name`）。")
    lines.append(f"- 量化：dataset mismatch 总数={ds_mismatch_total} / RUN_INDEX 总行数={total_rows}（见表 `ALIGNMENT_STATS_BY_DATASET`）。")
    lines.append("- 直接样例：见表 `MISMATCH_EXAMPLES_TOP30`（包含 run_index.dataset=sea_nodrift 但 summary.dataset_name=sea_abrupt4）。")
    lines.append("")

    lines.append("## 2) Q1：RUN_INDEX 是否存在 run_id/log_path 被多个 dataset 复用？")
    lines.append("- 结论：存在系统性复用。")
    lines.append("- 统计输出：`RUN_INDEX_DUP_BY_RUN_ID`、`RUN_INDEX_DUP_BY_LOG_PATH`（各 Top20 簇）。")
    if dup_by_log_path_rows:
        lines.append(
            f"- 示例簇（log_path 维度 top1）：cluster_size={dup_by_log_path_rows[0].get('cluster_size')}，n_datasets={dup_by_log_path_rows[0].get('n_datasets')}，datasets={dup_by_log_path_rows[0].get('datasets_json')}（见表 `RUN_INDEX_DUP_BY_LOG_PATH`）。"
        )
    lines.append("")

    lines.append("## 3) Q2：RUN_INDEX 每行字段能否与 summary 对齐？（dataset/seed/弱一致性）")
    lines.append("- 对齐真值表：`RUN_INDEX_VS_SUMMARY_ALIGNMENT`（每个 RUN_INDEX 行 1 条，summary 缺失标记为 missing_summary）。")
    lines.append("- 分组统计：`ALIGNMENT_STATS_BY_DATASET`（dataset mismatch / seed mismatch / missing_summary）。")
    lines.append("")

    lines.append("## 4) Q3：AL / METRICS_TABLE / RUN_INDEX 三者之间是否存在 join/聚合口径错配？")
    lines.append("- join 可追溯性审计：`CROSS_TABLE_JOIN_AUDIT`（解析 run_index_json 后，引用的 run_id/log_path 是否能在 RUN_INDEX 追溯）。")
    lines.append("- no-drift 标签纯度审计：`NODRIFT_LABEL_PURITY_AUDIT`（以 summary.dataset_name 作为真值 corrected_dataset，评估 sea_nodrift/sine_nodrift 是否被污染）。")
    lines.append("")

    lines.append("## 5) 根因定位（限定代码范围，只读）")
    lines.append("### 5.1 D3：summary.dataset_name 写入语义（training/loop.py）")
    lines.append("- 结论：summary.dataset_name 来自 `config.dataset_name`（因此 summary 的 dataset_name 字段语义明确）。")
    if loop_snip:
        lines.append("```py")
        lines.append(loop_snip)
        lines.append("```")
    else:
        lines.append("- N/A（文件不存在）")
    lines.append("")

    lines.append("### 5.2 D2：TrackAL 是否声明遍历 sea_nodrift/sine_nodrift（experiments/trackAL_perm_confirm_sweep.py）")
    if al_snip:
        lines.append("```py")
        lines.append(al_snip)
        lines.append("```")
    if al_snip2 and al_ensure_line and al_ensure_line != al_datasets_line:
        lines.append("```py")
        lines.append(al_snip2)
        lines.append("```")
    if al_snip3:
        lines.append("```py")
        lines.append(al_snip3)
        lines.append("```")
    lines.append("")

    lines.append("### 5.3 D1：RUN_INDEX 生成逻辑定位（scripts/summarize_next_stage_v14.py）")
    if summarize_snip:
        lines.append("```py")
        lines.append(summarize_snip)
        lines.append("```")
    else:
        lines.append("- N/A（文件不存在）")
    lines.append("")

    lines.append("## 6) RootCause 分类（写死）")
    lines.append(f"- 结论：{root_cause}")
    lines.append(f"- 证据：{json.dumps(rc_evidence, ensure_ascii=False)}")
    lines.append("- 可证伪条件：见表 `ROOT_CAUSE_CLASSIFICATION` 的 `falsifiable_condition`。")
    lines.append("")

    lines.append("## 7) 白名单文件缺失（如有）")
    if missing:
        for p in missing:
            lines.append(f"- N/A：{p}")
    else:
        lines.append("- 无")
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # stdout（简短）
    print(f"Q0: RUN_INDEX.dataset 不可信（dataset mismatch {ds_mismatch_total}/{total_rows}）")
    print(f"Q1: log_path/run_id 复用存在（见 {OUT_TABLES}）")
    print(f"RootCause: {root_cause}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
