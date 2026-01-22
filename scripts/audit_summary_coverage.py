#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit summary.json coverage from TrackAL CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit summary.json coverage for TrackAL runs")
    p.add_argument(
        "--trackal_csv",
        type=str,
        default="scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv",
        help="TrackAL aggregated CSV with run_index_json",
    )
    p.add_argument("--topn", type=int, default=20, help="Top-N groups by missing rate")
    return p.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_trackal_csv(raw_path: str) -> Path:
    default_path = Path("scripts/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv")
    if Path(raw_path) != default_path:
        return Path(raw_path)
    if default_path.exists():
        return default_path
    candidates = [
        Path("artifacts/v15p3/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv"),
        Path("artifacts/v15/tables/TRACKAL_PERM_CONFIRM_STABILITY_V15P3.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[info] resolved trackal_csv -> {cand}")
            return cand
    return default_path


def main() -> int:
    args = parse_args()
    trackal_path = _resolve_trackal_csv(args.trackal_csv)
    rows = _read_rows(trackal_path)
    if not rows:
        print(f"[error] trackal_csv not found or empty: {trackal_path}")
        return 2

    total_runs = 0
    total_exists = 0
    total_missing_logdir = 0
    total_missing_summary = 0

    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in rows:
        group = str(r.get("group") or "").strip()
        dataset = str(r.get("dataset") or "").strip()
        run_index_json = r.get("run_index_json") or "{}"
        try:
            run_index = json.loads(run_index_json)
        except Exception:
            run_index = {}
        runs = None
        if isinstance(run_index, dict):
            runs = run_index.get("runs")
            if not isinstance(runs, list):
                ds_info = run_index.get(dataset)
                if isinstance(ds_info, dict):
                    runs = ds_info.get("runs")
        elif isinstance(run_index, list):
            runs = run_index
        if not isinstance(runs, list):
            continue

        key = (group, dataset)
        rec = stats.setdefault(
            key,
            {
                "missing": 0,
                "total": 0,
                "missing_logdir": 0,
                "missing_summary": 0,
                "missing_paths": [],
            },
        )

        for item in runs:
            try:
                log_path = Path(str(item.get("log_path")))
            except Exception:
                continue
            summary_path = log_path.with_suffix(".summary.json")
            total_runs += 1
            rec["total"] += 1
            if not log_path.exists():
                total_missing_logdir += 1
                rec["missing_logdir"] += 1
                rec["missing"] += 1
                if len(rec["missing_paths"]) < 50:
                    rec["missing_paths"].append(
                        {
                            "type": "logdir_missing",
                            "log_path": str(log_path),
                            "summary_path": str(summary_path),
                        }
                    )
                continue
            if summary_path.exists():
                total_exists += 1
            else:
                total_missing_summary += 1
                rec["missing"] += 1
                rec["missing_summary"] += 1
                if len(rec["missing_paths"]) < 50:
                    rec["missing_paths"].append(
                        {
                            "type": "summary_missing",
                            "log_path": str(log_path),
                            "summary_path": str(summary_path),
                        }
                    )

    coverage = (float(total_exists) / float(total_runs)) if total_runs else 0.0
    print(
        "[summary] "
        f"total_runs={total_runs} "
        f"logdir_missing={total_missing_logdir} "
        f"summary_missing={total_missing_summary} "
        f"exists={total_exists} "
        f"coverage={coverage:.4f}"
    )

    ranked: List[Tuple[int, float, Tuple[str, str]]] = []
    for key, rec in stats.items():
        total = int(rec["total"])
        missing = int(rec["missing"])
        rate = (float(missing) / float(total)) if total else 0.0
        ranked.append((missing, rate, key))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    topn = max(0, int(args.topn))
    if topn > 0:
        print("[top-missing]")
        for missing, rate, key in ranked[:topn]:
            total = int(stats[key]["total"])
            missing_logdir = int(stats[key]["missing_logdir"])
            missing_summary = int(stats[key]["missing_summary"])
            g, d = key
            print(
                f"- group={g} dataset={d} missing={missing} total={total} "
                f"logdir_missing={missing_logdir} summary_missing={missing_summary} "
                f"missing_rate={rate:.4f}"
            )

    if ranked:
        missing, rate, key = ranked[0]
        total = int(stats[key]["total"])
        g, d = key
        print(
            f"[sample] top1 group={g} dataset={d} missing={missing} total={total} missing_rate={rate:.4f}"
        )
        for item in stats[key]["missing_paths"][:5]:
            print(
                "  "
                f"type={item['type']} "
                f"log_path={item['log_path']} "
                f"summary_path={item['summary_path']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
