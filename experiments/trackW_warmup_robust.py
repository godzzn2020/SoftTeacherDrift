#!/usr/bin/env python
"""
NEXT_STAGE V8 - Track W（可选）：warmup 阈值稳健性

只在 sea_abrupt4 的主方法组 D（two_stage + tuned PH + cooldown=200）上，
warmup ∈ {0, 1000, 2000, 5000} 重算 acc_min@warmup，检查排序是否一致。

实现：不重跑训练；从 Track U 的 CSV 精确定位 log_path -> 读取对应 *.summary.json 的 acc_series。

输出：scripts/TRACKW_WARMUP_ROBUST.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track W: warmup robustness (reuse Track U runs)")
    p.add_argument("--tracku_csv", type=str, default="scripts/TRACKU_SYNTH_GENERALIZATION.csv")
    p.add_argument("--dataset", type=str, default="sea_abrupt4")
    p.add_argument("--group", type=str, default="D_two_stage_tunedPH_cd200")
    p.add_argument("--warmups", nargs="+", type=int, default=[0, 1000, 2000, 5000])
    p.add_argument("--out_csv", type=str, default="scripts/TRACKW_WARMUP_ROBUST.csv")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_summary(log_path: str) -> Dict[str, Any]:
    sp = Path(log_path).with_suffix(".summary.json")
    return json.loads(sp.read_text(encoding="utf-8"))


def acc_min_after(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


def main() -> int:
    args = parse_args()
    rows = read_csv(Path(args.tracku_csv))
    if not rows:
        print("[warn] tracku_csv empty/missing")
        return 0
    dataset = str(args.dataset)
    group = str(args.group)
    warmups = [int(x) for x in args.warmups]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in rows if r.get("dataset") == dataset and r.get("group") == group]
    if not filtered:
        print("[warn] no matching rows")
        return 0

    records: List[Dict[str, Any]] = []
    for r in filtered:
        log_path = str(r.get("log_path") or "")
        if not log_path:
            continue
        summ = read_summary(log_path)
        for w in warmups:
            records.append(
                {
                    "track": "W",
                    "dataset": dataset,
                    "group": group,
                    "seed": int(float(r.get("seed") or 0)),
                    "run_id": str(r.get("run_id") or ""),
                    "log_path": log_path,
                    "warmup": int(w),
                    "acc_final": summ.get("acc_final"),
                    "acc_min": acc_min_after(summ, int(w)),
                }
            )

    if not records:
        print("[warn] no records")
        return 0
    fieldnames: List[str] = []
    seen: set[str] = set()
    for rec in records:
        for k in rec.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[done] wrote {out_csv} rows={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

