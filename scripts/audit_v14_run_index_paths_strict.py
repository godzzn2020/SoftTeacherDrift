#!/usr/bin/env python
"""
NEXT_STAGE V14 严格路径一致性审计（fail-fast）

输入：
- scripts/NEXT_STAGE_V14_RUN_INDEX.csv

输出：
- scripts/V14_STRICT_PATH_AUDIT.md
- scripts/V14_STRICT_PATH_AUDIT.csv

强约束：
- 不做任何递归扫描（不 os.walk / 不 find / 不 glob("**") / 不 listdir）。
- summary 读取仅按固定规则：log_path 末尾 .csv -> .summary.json（若非 .csv 结尾，允许追加一次 ".summary.json"）。
- 任一规则违反必须 sys.exit(2)（但会先落盘报告，便于复核）。
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


RUN_INDEX_CSV = Path("scripts/NEXT_STAGE_V14_RUN_INDEX.csv")
OUT_MD = Path("scripts/V14_STRICT_PATH_AUDIT.md")
OUT_CSV = Path("scripts/V14_STRICT_PATH_AUDIT.csv")


def _norm_path(p: str) -> str:
    return str(p or "").replace("\\", "/")


def _summary_path_from_log_path(log_path: str) -> Tuple[str, str]:
    lp = str(log_path or "")
    if lp.endswith(".csv"):
        return (lp[:-4] + ".summary.json", "replace_.csv")
    return (lp + ".summary.json", "append_.summary.json")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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
            fieldnames.append(k)
            seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str


def main() -> int:
    if not RUN_INDEX_CSV.exists():
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(f"# V14 STRICT PATH AUDIT\n\nRUN_INDEX 缺失：`{RUN_INDEX_CSV}`\n", encoding="utf-8")
        _write_csv(OUT_CSV, [{"check": "FAIL", "violation": "missing_run_index", "path": str(RUN_INDEX_CSV)}])
        return 2

    rows_in: List[Dict[str, str]] = []
    with RUN_INDEX_CSV.open("r", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))

    # 复用检查（R3）
    datasets_by_log_path: Dict[str, set[str]] = defaultdict(set)
    datasets_by_run_id: Dict[str, set[str]] = defaultdict(set)

    # 逐行规则（R1/R2）
    out_rows: List[Dict[str, Any]] = []
    violations_examples: List[Dict[str, Any]] = []
    stats_by_dataset: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, r in enumerate(rows_in, start=1):
        dataset = str(r.get("dataset") or "").strip()
        run_id = str(r.get("run_id") or "").strip()
        log_path = str(r.get("log_path") or "").strip()
        track = str(r.get("track") or "").strip()
        phase = str(r.get("phase") or "").strip()
        group = str(r.get("group") or "").strip()
        seed = str(r.get("seed") or "").strip()

        norm_log_path = _norm_path(log_path)

        stats_by_dataset[dataset]["n_rows"] += 1
        if not dataset:
            stats_by_dataset["__MISSING_DATASET__"]["n_rows"] += 1

        if log_path and dataset:
            datasets_by_log_path[log_path].add(dataset)
        if run_id and dataset:
            datasets_by_run_id[run_id].add(dataset)

        summary_path, summary_rule = _summary_path_from_log_path(log_path)
        summary_exists = Path(summary_path).exists()

        vios: List[Violation] = []

        # R1: log_path 必须包含 /{dataset}/
        has_dataset_in_path = False
        if dataset:
            has_dataset_in_path = f"/{dataset}/" in norm_log_path
            if not has_dataset_in_path:
                vios.append(Violation("R1_log_path_missing_dataset_segment", f"log_path 不包含 '/{dataset}/'"))
        else:
            vios.append(Violation("R1_missing_dataset", "dataset 为空"))

        # R1: summary 必须存在
        if not summary_exists:
            vios.append(Violation("R1_missing_summary", f"summary 不存在：{summary_path} (rule={summary_rule})"))

        summary_dataset_name: Optional[str] = None
        if summary_exists:
            try:
                summ = _read_json(Path(summary_path))
                summary_dataset_name = str(summ.get("dataset_name") or "").strip() or None
            except Exception as e:
                vios.append(Violation("R1_summary_read_error", f"summary 读取失败：{type(e).__name__}: {e}"))

        # R1: summary.dataset_name == RUN_INDEX.dataset
        if dataset and summary_exists:
            if summary_dataset_name != dataset:
                vios.append(
                    Violation(
                        "R1_summary_dataset_name_mismatch",
                        f"summary.dataset_name={summary_dataset_name!r} != run_index.dataset={dataset!r}",
                    )
                )

        # R2: nodrift 加严
        is_nodrift = dataset in {"sea_nodrift", "sine_nodrift"}
        has_forbidden_abrupt = False
        if is_nodrift:
            forbidden = ["/sea_abrupt4/", "/sine_abrupt4/"]
            has_forbidden_abrupt = any(seg in norm_log_path for seg in forbidden)
            if not has_dataset_in_path:
                vios.append(Violation("R2_nodrift_path_not_self", f"nodrift log_path 未落在自身目录：{dataset}"))
            if has_forbidden_abrupt:
                vios.append(Violation("R2_nodrift_path_contains_abrupt4", "nodrift log_path 含 abrupt4 目录片段"))

        if vios:
            stats_by_dataset[dataset]["n_violations"] += 1
        if any(v.code == "R1_missing_summary" for v in vios):
            stats_by_dataset[dataset]["n_missing_summary"] += 1
        if any(v.code == "R1_summary_dataset_name_mismatch" for v in vios):
            stats_by_dataset[dataset]["n_dataset_mismatch"] += 1
        if any(v.code == "R1_log_path_missing_dataset_segment" for v in vios):
            stats_by_dataset[dataset]["n_path_mismatch"] += 1

        out_row: Dict[str, Any] = {
            "check": "FAIL" if vios else "PASS",
            "row_idx": int(i),
            "track": track or "N/A",
            "phase": phase or "N/A",
            "dataset": dataset or "N/A",
            "seed": seed or "N/A",
            "group": group or "N/A",
            "run_id": run_id or "N/A",
            "log_path": log_path or "N/A",
            "log_path_contains_dataset_segment": int(bool(has_dataset_in_path)),
            "summary_path": summary_path,
            "summary_path_rule": summary_rule,
            "summary_exists": int(bool(summary_exists)),
            "summary_dataset_name": summary_dataset_name or "N/A",
            "is_nodrift": int(bool(is_nodrift)),
            "nodrift_has_forbidden_abrupt4_segment": int(bool(has_forbidden_abrupt)),
            "violations": ";".join([v.code for v in vios]) if vios else "",
            "violation_details": " | ".join([v.detail for v in vios]) if vios else "",
        }
        out_rows.append(out_row)
        if vios and len(violations_examples) < 50:
            violations_examples.append(out_row)

    # R3: 复用检查（log_path / run_id）
    dup_log_path_clusters: List[Tuple[str, List[str]]] = []
    for lp, dss in datasets_by_log_path.items():
        if len(dss) > 1:
            dup_log_path_clusters.append((lp, sorted(dss)))

    dup_run_id_clusters: List[Tuple[str, List[str]]] = []
    for rid, dss in datasets_by_run_id.items():
        if len(dss) > 1:
            dup_run_id_clusters.append((rid, sorted(dss)))

    dup_log_path_clusters.sort(key=lambda x: (-len(x[1]), x[0]))
    dup_run_id_clusters.sort(key=lambda x: (-len(x[1]), x[0]))

    has_dup_violation = bool(dup_log_path_clusters or dup_run_id_clusters)
    if has_dup_violation:
        # 将 R3 违反也体现在每行上（便于 CSV 复核），并纳入 fail-fast
        log_path_bad = {lp for lp, _ in dup_log_path_clusters}
        run_id_bad = {rid for rid, _ in dup_run_id_clusters}
        for r in out_rows:
            if r.get("log_path") in log_path_bad:
                r["check"] = "FAIL"
                r["violations"] = (str(r.get("violations") or "") + ";R3_dup_log_path").strip(";")
            if r.get("run_id") in run_id_bad:
                r["check"] = "FAIL"
                r["violations"] = (str(r.get("violations") or "") + ";R3_dup_run_id").strip(";")

    total_rows = len(out_rows)
    total_fail = sum(1 for r in out_rows if str(r.get("check")) == "FAIL")

    # 输出 CSV（全量行）
    _write_csv(OUT_CSV, out_rows)

    # 输出 MD（摘要 + Top-50）
    md_lines: List[str] = []
    md_lines.append("# V14 STRICT PATH AUDIT（fail-fast）")
    md_lines.append("")
    md_lines.append("## 范围声明（强约束）")
    md_lines.append("- 未做任何递归扫描（不 os.walk/find/glob/listdir）。")
    md_lines.append("- summary 读取仅按固定规则：`log_path` 末尾 `.csv -> .summary.json`（否则仅追加一次 `.summary.json`）。")
    md_lines.append("")
    md_lines.append("## 总览")
    md_lines.append(f"- RUN_INDEX：`{RUN_INDEX_CSV}`")
    md_lines.append(f"- 总行数：{total_rows}")
    md_lines.append(f"- 违反行数：{total_fail}")
    md_lines.append(f"- 输出：`{OUT_MD}`、`{OUT_CSV}`")
    md_lines.append("")

    # 按 dataset 统计
    stat_rows: List[List[str]] = []
    for ds in sorted(stats_by_dataset.keys()):
        s = stats_by_dataset[ds]
        stat_rows.append(
            [
                ds or "N/A",
                str(int(s.get("n_rows") or 0)),
                str(int(s.get("n_violations") or 0)),
                str(int(s.get("n_missing_summary") or 0)),
                str(int(s.get("n_dataset_mismatch") or 0)),
                str(int(s.get("n_path_mismatch") or 0)),
            ]
        )
    md_lines.append("## 按 dataset 违反统计")
    md_lines.append(
        _md_table(
            ["dataset", "n_rows", "n_violations", "n_missing_summary", "n_dataset_mismatch", "n_path_mismatch"],
            stat_rows,
        )
    )
    md_lines.append("")

    # R3 dup 簇统计（Top-20）
    md_lines.append("## R3 复用检查（Top-20）")
    dup_lp_rows = [[str(len(dss)), json.dumps(dss, ensure_ascii=False), lp] for lp, dss in dup_log_path_clusters[:20]]
    dup_rid_rows = [[str(len(dss)), json.dumps(dss, ensure_ascii=False), rid] for rid, dss in dup_run_id_clusters[:20]]
    md_lines.append("### dup_by_log_path")
    md_lines.append(_md_table(["n_datasets", "datasets", "log_path"], dup_lp_rows))
    md_lines.append("")
    md_lines.append("### dup_by_run_id")
    md_lines.append(_md_table(["n_datasets", "datasets", "run_id"], dup_rid_rows))
    md_lines.append("")

    # Top-50 违反样例
    md_lines.append("## Top-50 违反样例")
    ex_rows: List[List[str]] = []
    for r in violations_examples[:50]:
        ex_rows.append(
            [
                str(r.get("row_idx")),
                str(r.get("dataset")),
                str(r.get("run_id")),
                str(r.get("summary_dataset_name")),
                str(r.get("log_path")),
                str(r.get("violations")),
            ]
        )
    md_lines.append(_md_table(["row_idx", "dataset", "run_id", "summary.dataset_name", "log_path", "violations"], ex_rows))
    md_lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if total_fail > 0 or has_dup_violation:
        print(f"[FAIL] strict path audit violations={total_fail} dup_violation={int(has_dup_violation)}")
        return 2

    print("[PASS] strict path audit OK (0 violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

