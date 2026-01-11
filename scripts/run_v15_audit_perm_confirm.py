#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 审计（Permutation-test Confirm + vote_score）：
- 仅基于 summarize 产物做一致性/可复现性检查（不重跑训练）
- 重点：RUN_INDEX -> summary.json 可达性、重复/缺失、log_root 分布
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V15 audit for perm_confirm")
    p.add_argument("--run_index_csv", type=str, default="scripts/NEXT_STAGE_V15_RUN_INDEX.csv")
    p.add_argument("--report_md", type=str, default="V15_AUDIT_PERM_CONFIRM.md")
    p.add_argument("--tables_csv", type=str, default="V15_AUDIT_PERM_CONFIRM_TABLES.csv")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tables(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    run_index_csv = Path(args.run_index_csv)
    report_md = Path(args.report_md)
    tables_csv = Path(args.tables_csv)

    if not run_index_csv.exists():
        raise FileNotFoundError(f"缺失：{run_index_csv}")

    rows = read_csv(run_index_csv)
    total = len(rows)

    missing_log = 0
    missing_summary = 0
    dup_key = 0
    dup_log_path = 0

    key_seen: set[Tuple[str, str, str, str, str]] = set()
    log_seen: set[str] = set()
    log_roots: Counter[str] = Counter()
    missing_summary_rows: List[Dict[str, str]] = []

    for r in rows:
        track = str(r.get("track") or "")
        phase = str(r.get("phase") or "")
        group = str(r.get("group") or "")
        dataset = str(r.get("dataset") or "")
        seed = str(r.get("seed") or "")
        log_path = str(r.get("log_path") or "")

        k = (track, phase, group, dataset, seed)
        if k in key_seen:
            dup_key += 1
        key_seen.add(k)

        if log_path in log_seen:
            dup_log_path += 1
        log_seen.add(log_path)

        lp = Path(log_path) if log_path else None
        if lp and lp.parts:
            log_roots[lp.parts[0]] += 1

        if not log_path or not Path(log_path).exists():
            missing_log += 1
            continue
        sp = Path(log_path).with_suffix(".summary.json")
        if not sp.exists():
            missing_summary += 1
            missing_summary_rows.append(
                {
                    "table_name": "MISSING_SUMMARY",
                    "track": track,
                    "phase": phase,
                    "group": group,
                    "dataset": dataset,
                    "seed": seed,
                    "log_path": log_path,
                    "summary_path": str(sp),
                }
            )

    table_rows: List[Dict[str, str]] = []
    table_rows.append({"table_name": "SUMMARY", "key": "run_index_rows", "value": str(total)})
    table_rows.append({"table_name": "SUMMARY", "key": "missing_log_path", "value": str(missing_log)})
    table_rows.append({"table_name": "SUMMARY", "key": "missing_summary_json", "value": str(missing_summary)})
    table_rows.append({"table_name": "SUMMARY", "key": "dup_key(track,phase,group,dataset,seed)", "value": str(dup_key)})
    table_rows.append({"table_name": "SUMMARY", "key": "dup_log_path", "value": str(dup_log_path)})
    for root, cnt in sorted(log_roots.items(), key=lambda x: (-x[1], x[0])):
        table_rows.append({"table_name": "LOG_ROOT_COUNTS", "log_root": root, "count": str(cnt)})
    table_rows.extend(missing_summary_rows)
    write_tables(tables_csv, table_rows)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines: List[str] = []
    md_lines.append("# V15 审计（Permutation-test Confirm）")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{now}")
    md_lines.append(f"- RUN_INDEX：`{run_index_csv}` rows={total}")
    md_lines.append("")
    md_lines.append("## 检查摘要")
    md_lines.append(f"- missing_log_path={missing_log}")
    md_lines.append(f"- missing_summary_json={missing_summary}")
    md_lines.append(f"- dup_key(track,phase,group,dataset,seed)={dup_key}")
    md_lines.append(f"- dup_log_path={dup_log_path}")
    md_lines.append("")
    md_lines.append("## log_root 分布")
    if log_roots:
        for root, cnt in sorted(log_roots.items(), key=lambda x: (-x[1], x[0])):
            md_lines.append(f"- {root}: {cnt}")
    else:
        md_lines.append("- N/A")
    md_lines.append("")
    md_lines.append("## 产物")
    md_lines.append(f"- `{report_md}`")
    md_lines.append(f"- `{tables_csv}`")

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {report_md}")
    print(f"[ok] wrote: {tables_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

