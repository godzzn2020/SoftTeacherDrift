from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(statistics.mean(vals))


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


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
    p = argparse.ArgumentParser(description="Exp1 summary")
    p.add_argument("--run_index_glob", type=str, default="results/exp1_signal_set/run_index_*.csv")
    p.add_argument("--out_csv", type=str, default="results/exp1_signal_set/summary_table.csv")
    p.add_argument("--out_detail_csv", type=str, default="results/exp1_signal_set/summary_detail.csv")
    p.add_argument("--tol", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(Path(".").glob(str(args.run_index_glob)))
    rows = read_run_index_files(files)
    if not rows:
        print("[warn] no run_index rows")
        return 1

    run_metrics: List[Dict[str, Any]] = []
    missing = 0
    for row in rows:
        log_path = Path(str(row.get("log_path") or ""))
        summ = read_summary(log_path)
        if summ is None:
            missing += 1
            continue
        dataset = str(row.get("dataset_name") or "")
        seed = _safe_int(row.get("seed")) or 0
        labeled_ratio = _safe_float(row.get("labeled_ratio")) or 0.0
        signal_set = str(row.get("signal_set") or "")
        horizon = _safe_int(summ.get("horizon")) or 0
        confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
        confirmed = merge_events(confirmed_raw, int(args.min_separation))
        candidate_count = _safe_int(summ.get("candidate_count_total")) or 0
        confirmed_count = _safe_int(summ.get("confirmed_count_total")) or 0
        perm_test_count = _safe_int(summ.get("perm_test_count_total")) or 0
        perm_reject_count = _safe_int(summ.get("perm_reject_count_total")) or 0
        not_tested = max(0, int(candidate_count - perm_test_count))

        rec: Dict[str, Any] = {
            "dataset": dataset,
            "seed": int(seed),
            "labeled_ratio": float(labeled_ratio),
            "signal_set": signal_set,
            "horizon": int(horizon),
            "candidate_count": int(candidate_count),
            "confirmed_count": int(confirmed_count),
            "perm_test_count": int(perm_test_count),
            "perm_reject_count": int(perm_reject_count),
            "not_tested_count": int(not_tested),
        }

        if "nodrift" in dataset.lower():
            rate = (float(confirmed_count) * 10000.0 / float(horizon)) if horizon > 0 else None
            rec["no_drift_rate_per_10k"] = rate
            rec["miss_tol500"] = None
            rec["conf_p90"] = None
            rec["hard_ok"] = None
        else:
            gt_starts = default_gt_starts(horizon)
            if dataset.lower() == "stagger_abrupt3":
                gt_starts = [x for x in (20000, 40000) if int(x) < int(horizon)]
            delays, miss_flags = per_drift_first_delays(gt_starts, confirmed, horizon=horizon, tol=int(args.tol))
            miss = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None
            conf_p90 = percentile(delays, 0.90) if delays else None
            hard_ok = bool(miss is not None and miss <= 1e-12 and conf_p90 is not None and conf_p90 < 500)
            rec["no_drift_rate_per_10k"] = None
            rec["miss_tol500"] = miss
            rec["conf_p90"] = conf_p90
            rec["hard_ok"] = int(bool(hard_ok))

        run_metrics.append(rec)

    if missing > 0:
        print(f"[warn] missing summary: {missing}")

    out_detail = Path(args.out_detail_csv)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    if run_metrics:
        fieldnames: List[str] = []
        seen: set[str] = set()
        for r in run_metrics:
            for k in r.keys():
                if k in seen:
                    continue
                fieldnames.append(k)
                seen.add(k)
        with out_detail.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(run_metrics)

    def group_key(rec: Dict[str, Any]) -> Tuple[float, str]:
        return (float(rec.get("labeled_ratio") or 0.0), str(rec.get("signal_set") or ""))

    by_key: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
    for rec in run_metrics:
        by_key.setdefault(group_key(rec), []).append(rec)

    baseline_by_ratio: Dict[float, Dict[int, float]] = {}
    overall_by_key_seed: Dict[Tuple[float, str], Dict[int, float]] = {}

    for (ratio, signal), recs in by_key.items():
        sea = {r["seed"]: r for r in recs if r["dataset"] == "sea_nodrift"}
        sine = {r["seed"]: r for r in recs if r["dataset"] == "sine_nodrift"}
        per_seed: Dict[int, float] = {}
        for seed in sorted(set(sea.keys()) & set(sine.keys())):
            a = _safe_float(sea[seed].get("no_drift_rate_per_10k"))
            b = _safe_float(sine[seed].get("no_drift_rate_per_10k"))
            if a is None or b is None:
                continue
            per_seed[int(seed)] = float((a + b) / 2.0)
        overall_by_key_seed[(ratio, signal)] = per_seed
        if signal == "error":
            baseline_by_ratio[ratio] = dict(per_seed)

    table_rows: List[Dict[str, Any]] = []
    order = {"error": "G0", "proxy": "G1", "all": "G2"}

    for (ratio, signal), recs in sorted(by_key.items(), key=lambda x: (x[0][0], order.get(x[0][1], "G9"))):
        drift_sea = {r["seed"]: r for r in recs if r["dataset"] == "sea_abrupt4"}
        drift_sine = {r["seed"]: r for r in recs if r["dataset"] == "sine_abrupt4"}
        pass_flags: List[int] = []
        for seed in sorted(set(drift_sea.keys()) & set(drift_sine.keys())):
            a = drift_sea[seed].get("hard_ok")
            b = drift_sine[seed].get("hard_ok")
            if a is None or b is None:
                continue
            pass_flags.append(int(a) * int(b))
        hard_ok_pass_rate = (float(sum(pass_flags)) / float(len(pass_flags))) if pass_flags else None

        sea_nd = [r for r in recs if r["dataset"] == "sea_nodrift"]
        sine_nd = [r for r in recs if r["dataset"] == "sine_nodrift"]
        sea_rates = [_safe_float(r.get("no_drift_rate_per_10k")) for r in sea_nd]
        sine_rates = [_safe_float(r.get("no_drift_rate_per_10k")) for r in sine_nd]

        overall_rates = list(overall_by_key_seed.get((ratio, signal), {}).values())
        overall_mean = mean(overall_rates)
        overall_std = std(overall_rates)

        cand_vals = [_safe_float(r.get("candidate_count")) for r in recs]
        conf_vals = [_safe_float(r.get("confirmed_count")) for r in recs]
        not_tested_vals = [_safe_float(r.get("not_tested_count")) for r in recs]
        reject_vals = [_safe_float(r.get("perm_reject_count")) for r in recs]

        worst_regression = None
        base_seed_rates = baseline_by_ratio.get(float(ratio), {})
        cur_seed_rates = overall_by_key_seed.get((ratio, signal), {})
        deltas: List[float] = []
        for seed, rate in cur_seed_rates.items():
            base = base_seed_rates.get(seed)
            if base is None:
                continue
            deltas.append(float(rate - base))
        if deltas:
            worst_regression = float(max(deltas))
        elif signal == "error":
            worst_regression = 0.0

        table_rows.append(
            {
                "labeled_ratio": float(ratio),
                "group": f"{order.get(signal, signal)}_{signal}",
                "hard_ok_pass_rate": hard_ok_pass_rate,
                "no_drift_rate_overall_mean": overall_mean,
                "no_drift_rate_overall_std": overall_std,
                "no_drift_rate_sea_mean": mean(sea_rates),
                "no_drift_rate_sea_std": std(sea_rates),
                "no_drift_rate_sine_mean": mean(sine_rates),
                "no_drift_rate_sine_std": std(sine_rates),
                "worst_seed_regression_per_10k": worst_regression,
                "candidate_mean": mean(cand_vals),
                "confirmed_mean": mean(conf_vals),
                "not_tested_mean": mean(not_tested_vals),
                "reject_mean": mean(reject_vals),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if table_rows:
        fieldnames: List[str] = []
        seen: set[str] = set()
        for r in table_rows:
            for k in r.keys():
                if k in seen:
                    continue
                fieldnames.append(k)
                seen.add(k)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)
        print(f"[done] wrote {out_csv} rows={len(table_rows)}")
    else:
        print("[warn] empty summary table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
