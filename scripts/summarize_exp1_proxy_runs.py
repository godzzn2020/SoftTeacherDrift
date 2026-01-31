from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


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


def default_gt_starts(n_samples: int) -> List[int]:
    step = max(1, int(n_samples) // 5)
    return [int(step), int(2 * step), int(3 * step), int(4 * step)]


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def per_drift_first_delays(
    gt_drifts: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Tuple[List[float], List[int]]:
    gt = sorted(int(d) for d in gt_drifts)
    confs = sorted(int(x) for x in confirmed)
    delays: List[float] = []
    miss_flags: List[int] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_f = first_event_in_range(confs, g, end)
        delays.append(float(end - g) if first_f is None else float(first_f - g))
        tol_end = min(end, g + int(tol) + 1)
        miss_flags.append(0 if first_event_in_range(confs, g, tol_end) is not None else 1)
    return delays, miss_flags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Exp1 proxy runs")
    p.add_argument("--run_index_glob", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    return p.parse_args()


def read_run_index_files(files: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(dict(row))
    return rows


def read_summary(log_path: Path) -> Optional[Dict[str, Any]]:
    sp = log_path.with_suffix(".summary.json")
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    files = sorted(Path(".").glob(args.run_index_glob))
    rows = read_run_index_files(files)
    if not rows:
        print("[warn] no run_index rows")
        return 1

    grouped: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    meta_by_key: Dict[Tuple[str, float, str], Dict[str, Any]] = {}

    for row in rows:
        run_tag = str(row.get("run_tag") or "")
        ratio = _safe_float(row.get("labeled_ratio"))
        dataset = str(row.get("dataset_name") or "")
        if ratio is None:
            continue
        key = (run_tag, float(ratio), dataset)
        grouped.setdefault(key, []).append(row)
        if key not in meta_by_key:
            meta_by_key[key] = {
                "monitor_preset": row.get("monitor_preset"),
                "signal_set": row.get("signal_set"),
                "perm_stat": row.get("perm_stat"),
                "candidate_signals": row.get("candidate_signals"),
            }

    out_rows: List[Dict[str, Any]] = []
    for (run_tag, ratio, dataset), items in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        cand_rates: List[Optional[float]] = []
        conf_rates: List[Optional[float]] = []
        hard_ok_flags: List[int] = []
        nodrift_rates: List[Optional[float]] = []
        for row in items:
            log_path = Path(str(row.get("log_path") or ""))
            summ = read_summary(log_path)
            if summ is None:
                continue
            horizon = _safe_int(summ.get("horizon")) or 0
            cand = _safe_int(summ.get("candidate_count_total")) or 0
            conf = _safe_int(summ.get("confirmed_count_total")) or 0
            cand_rate = (float(cand) * 10000.0 / float(horizon)) if horizon > 0 else None
            conf_rate = (float(conf) * 10000.0 / float(horizon)) if horizon > 0 else None
            cand_rates.append(cand_rate)
            conf_rates.append(conf_rate)

            if "nodrift" in dataset.lower():
                nodrift_rates.append(conf_rate)
            else:
                confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                confirmed = merge_events(confirmed_raw, int(args.min_separation))
                gt_starts = default_gt_starts(horizon)
                if dataset.lower() == "stagger_abrupt3":
                    gt_starts = [x for x in (20000, 40000) if int(x) < int(horizon)]
                delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=int(args.tol))
                miss = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None
                conf_p90 = percentile(delays, 0.90) if delays else None
                hard_ok = bool(miss is not None and miss <= 1e-12 and conf_p90 is not None and conf_p90 < 500)
                hard_ok_flags.append(1 if hard_ok else 0)

        meta = meta_by_key.get((run_tag, ratio, dataset), {})
        out_rows.append(
            {
                "run_tag": run_tag,
                "monitor_preset": meta.get("monitor_preset"),
                "signal_set": meta.get("signal_set"),
                "perm_stat": meta.get("perm_stat"),
                "candidate_signals": meta.get("candidate_signals"),
                "labeled_ratio": float(ratio),
                "dataset": dataset,
                "seed_count": int(len(items)),
                "candidate_per_10k_mean": mean(cand_rates),
                "confirmed_per_10k_mean": mean(conf_rates),
                "hard_ok_pass_rate": (float(sum(hard_ok_flags)) / float(len(hard_ok_flags))) if hard_ok_flags else None,
                "no_drift_rate_mean": mean(nodrift_rates),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        fieldnames: List[str] = []
        seen: set[str] = set()
        for r in out_rows:
            for k in r.keys():
                if k in seen:
                    continue
                fieldnames.append(k)
                seen.add(k)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"[done] wrote {out_csv} rows={len(out_rows)}")
    else:
        print("[warn] empty output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
