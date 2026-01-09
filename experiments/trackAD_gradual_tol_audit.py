#!/usr/bin/env python
"""
NEXT_STAGE V11 - Track AD：Gradual drift 的区间 GT 口径重算（重点：stagger_gradual_frequent）

只做口径审计，不新跑训练：
- 通过 Track AA CSV（run_id/log_path）精确定位本轮需要的少量 run
- 从对应的 *.summary.json 读取 confirmed_sample_idxs
- 用“区间 drift GT”（start/end/mid）重算 tol500_start/mid/end 与 delay 分位数

输出：scripts/TRACKAD_GRADUAL_TOL_AUDIT.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track AD: gradual GT interval tolerance audit (stagger)")
    p.add_argument("--trackaa_csv", type=str, default="scripts/TRACKAA_GENERALIZATION_NONABRUPT.csv")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAD_GRADUAL_TOL_AUDIT.csv")
    p.add_argument("--profile", type=str, default="stagger_gradual_frequent")
    p.add_argument(
        "--groups",
        type=str,
        default="B_or_tunedPH,C_weighted_tunedPH,D_two_stage_tunedPH_cd200",
        help="逗号分隔，默认至少 B/C/D",
    )
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--concept_length", type=int, default=5000)
    p.add_argument("--transition_length", type=int, default=2000)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(v: Any) -> Optional[int]:
    x = _safe_float(v)
    return None if x is None else int(x)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))  # type: ignore[arg-type]
    if not vals:
        return None
    if q <= 0:
        return float(vals[0])
    if q >= 1:
        return float(vals[-1])
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    w = pos - lo
    return float(vals[lo] * (1 - w) + vals[hi] * w)


def merge_events(events: Sequence[int], min_sep: int) -> List[int]:
    if not events:
        return []
    ev = sorted(int(x) for x in events)
    merged = [ev[0]]
    for x in ev[1:]:
        if x - merged[-1] < int(min_sep):
            continue
        merged.append(x)
    return merged


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def build_gradual_intervals(*, n_samples: int, concept_length: int, transition_length: int) -> List[Dict[str, float]]:
    step = int(concept_length) * 2  # stagger: segment_length=concept_length*2
    drift_starts = [int(x) for x in range(step, int(n_samples), step)]
    intervals: List[Dict[str, float]] = []
    for i, s in enumerate(drift_starts):
        next_s = drift_starts[i + 1] if i + 1 < len(drift_starts) else int(n_samples)
        e = min(int(next_s), int(s + int(transition_length)))
        mid = 0.5 * (float(s) + float(e))
        intervals.append({"start": float(s), "end": float(e), "mid": float(mid)})
    return intervals


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compute_tol_and_delays(
    intervals: Sequence[Dict[str, float]],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Dict[str, Any]:
    starts = [int(d["start"]) for d in intervals]
    ends = [int(d["end"]) for d in intervals]
    mids = [float(d["mid"]) for d in intervals]

    confs = sorted(int(x) for x in confirmed)
    first_conf: List[Optional[int]] = []
    for i, s in enumerate(starts):
        end_window = starts[i + 1] if i + 1 < len(starts) else int(horizon)
        first_conf.append(first_event_in_range(confs, int(s), int(end_window)))

    def _miss(anchor: float) -> List[int]:
        out: List[int] = []
        for i, fc in enumerate(first_conf):
            thr = float(anchor) + float(tol)
            if fc is None:
                out.append(1)
            else:
                out.append(0 if float(fc) <= thr else 1)
        return out

    def _delays(anchor: Sequence[float]) -> List[float]:
        out: List[float] = []
        for i, fc in enumerate(first_conf):
            if fc is None:
                continue
            out.append(float(fc) - float(anchor[i]))
        return out

    anchors_start = [float(x) for x in starts]
    anchors_end = [float(x) for x in ends]
    anchors_mid = [float(x) for x in mids]

    miss_start = _miss(0.0)  # placeholder, per-drift below
    miss_mid = _miss(0.0)
    miss_end = _miss(0.0)
    # per drift anchor differs; recompute per drift
    miss_start = [0 if (fc is not None and fc <= s + tol) else 1 for fc, s in zip(first_conf, starts)]
    miss_mid = [0 if (fc is not None and float(fc) <= m + tol) else 1 for fc, m in zip(first_conf, mids)]
    miss_end = [0 if (fc is not None and fc <= e + tol) else 1 for fc, e in zip(first_conf, ends)]

    d_start = _delays(anchors_start)
    d_mid = _delays(anchors_mid)
    d_end = _delays(anchors_end)

    return {
        "n_drifts": int(len(intervals)),
        "miss_tol500_start": (float(sum(miss_start)) / len(miss_start)) if miss_start else None,
        "miss_tol500_mid": (float(sum(miss_mid)) / len(miss_mid)) if miss_mid else None,
        "miss_tol500_end": (float(sum(miss_end)) / len(miss_end)) if miss_end else None,
        "delay_start_P50": percentile(d_start, 0.50),
        "delay_start_P90": percentile(d_start, 0.90),
        "delay_start_P99": percentile(d_start, 0.99),
        "delay_mid_P50": percentile(d_mid, 0.50),
        "delay_mid_P90": percentile(d_mid, 0.90),
        "delay_mid_P99": percentile(d_mid, 0.99),
        "delay_end_P50": percentile(d_end, 0.50),
        "delay_end_P90": percentile(d_end, 0.90),
        "delay_end_P99": percentile(d_end, 0.99),
    }


def main() -> int:
    args = parse_args()
    trackaa_csv = Path(args.trackaa_csv)
    out_csv = Path(args.out_csv)
    profile = str(args.profile)
    groups = [x.strip() for x in str(args.groups).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]

    rows = read_csv(trackaa_csv)
    rows = [r for r in rows if r.get("dataset") == profile and r.get("group") in set(groups)]
    if seeds:
        rows = [r for r in rows if _safe_int(r.get("seed")) in set(seeds)]

    intervals = build_gradual_intervals(n_samples=int(args.n_samples), concept_length=int(args.concept_length), transition_length=int(args.transition_length))

    out: List[Dict[str, Any]] = []
    for r in rows:
        log_path_s = str(r.get("log_path") or "").strip()
        if not log_path_s:
            continue
        summ = read_run_summary(Path(log_path_s))
        horizon = int(summ.get("horizon") or r.get("horizon") or args.n_samples)
        confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
        confirmed = merge_events(confirmed_raw, int(args.min_separation))
        metrics = compute_tol_and_delays(intervals, confirmed, horizon=horizon, tol=int(args.tol))
        out.append(
            {
                "track": "AD",
                "profile": profile,
                "unit": "sample_idx",
                "group": str(r.get("group") or ""),
                "seed": _safe_int(r.get("seed")),
                "run_id": str(r.get("run_id") or ""),
                "log_path": log_path_s,
                "horizon": horizon,
                "transition_length": int(args.transition_length),
                "tol": int(args.tol),
                **metrics,
            }
        )

    write_csv(out_csv, out)
    print(f"[done] wrote {out_csv} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

