#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单 CSV 分片合并（按 header 一致性检查，直接拼接行）。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concat sharded CSV files with same header")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--inputs", type=str, required=True, help="逗号分隔输入 CSV")
    return p.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        header = list(r.fieldnames or [])
        rows = list(r)
    return header, rows


def write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    inputs = [Path(p.strip()) for p in str(args.inputs).split(",") if p.strip()]
    if not inputs:
        raise SystemExit("未提供任何 inputs")

    merged_header: List[str] = []
    merged_rows: List[Dict[str, str]] = []
    for i, p in enumerate(inputs):
        if not p.exists():
            raise SystemExit(f"输入不存在：{p}")
        header, rows = read_csv(p)
        if i == 0:
            merged_header = header
        elif header != merged_header:
            raise SystemExit(f"CSV header 不一致：{p}")
        merged_rows.extend(rows)

    write_csv(out_csv, merged_header, merged_rows)
    print(f"[ok] merged_rows={len(merged_rows)} -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

