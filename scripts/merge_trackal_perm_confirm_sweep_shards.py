#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并 TrackAL perm_confirm_sweep 的分片输出 CSV。

用法示例：
  python scripts/merge_trackal_perm_confirm_sweep_shards.py \
    --out_csv scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv \
    --inputs scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15_shard0.csv,scripts/TRACKAL_PERM_CONFIRM_SWEEP_V15_shard1.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded TRACKAL_PERM_CONFIRM_SWEEP CSVs")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--inputs", type=str, required=True, help="逗号分隔输入 CSV 路径列表")
    return p.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    inputs = [Path(p.strip()) for p in str(args.inputs).split(",") if p.strip()]
    if not inputs:
        raise SystemExit("未提供任何 --inputs")

    merged_header: List[str] = []
    merged_rows: List[Dict[str, str]] = []

    seen_keys: set[Tuple[str, str, str, str]] = set()
    for i, p in enumerate(inputs):
        if not p.exists():
            raise SystemExit(f"输入不存在：{p}")
        header, rows = read_csv(p)
        if i == 0:
            merged_header = header
        elif header != merged_header:
            raise SystemExit(f"CSV 表头不一致：{p}")
        for r in rows:
            k = (str(r.get("track", "")), str(r.get("phase", "")), str(r.get("dataset", "")), str(r.get("group", "")))
            if k in seen_keys:
                raise SystemExit(f"发现重复 key(track,phase,dataset,group)={k}（请检查分片是否重叠）：{p}")
            seen_keys.add(k)
            merged_rows.append(r)

    merged_rows.sort(key=lambda r: (str(r.get("group") or ""), str(r.get("dataset") or ""), str(r.get("phase") or ""), str(r.get("track") or "")))
    write_csv(out_csv, merged_header, merged_rows)
    print(f"[ok] merged_rows={len(merged_rows)} -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

