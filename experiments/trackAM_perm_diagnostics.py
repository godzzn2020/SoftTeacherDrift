#!/usr/bin/env python
"""
NEXT_STAGE V14 - Track AM（推荐）：Permutation-test confirm 机制诊断

目的：解释 perm_test 如何降低 no-drift 误报。

实现策略（避免扫描 logs/）：
- 读取 Track AL 的聚合 CSV（含 run_index_json）
- 只打开该 CSV 指定 runs 的单个 `.summary.json`（run-level 小文件）

输出：scripts/TRACKAM_PERM_DIAG.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track AM: permutation-test diagnostics (from Track AL runs)")
    p.add_argument("--trackal_csv", type=str, default="scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAM_PERM_DIAG.csv")
    p.add_argument(
        "--groups",
        type=str,
        default="",
        help="可选：逗号分隔 group；为空则自动选择 baseline + winner（优先 n20）",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="sea_nodrift,sine_nodrift,sea_abrupt4,sine_abrupt4,stagger_abrupt3",
        help="逗号分隔 dataset 列表",
    )
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(statistics.mean(vals))


def std(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def pick_winner(trackal_rows: List[Dict[str, str]], acc_tol: float) -> Tuple[Optional[str], Optional[str]]:
    # 基于 Track AL 聚合行的 C2 规则选 winner（优先 n20，否则 n5）
    if not trackal_rows:
        return None, "未找到 Track AL CSV"

    # prefer full phase if present
    def phase_rank(r: Dict[str, str]) -> int:
        ph = str(r.get("phase") or "")
        return 0 if ph == "full" else 1

    # pick best row per (group,dataset) with phase priority
    by_gd: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in sorted(trackal_rows, key=phase_rank):
        g = str(r.get("group") or "")
        d = str(r.get("dataset") or "")
        if not g or not d:
            continue
        by_gd.setdefault((g, d), r)

    by_group: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (g, d), r in by_gd.items():
        by_group.setdefault(g, {})[d] = r

    eligible: List[str] = []
    for g, dsmap in by_group.items():
        sea = dsmap.get("sea_abrupt4")
        sine = dsmap.get("sine_abrupt4")
        if not sea or not sine:
            continue
        sea_miss = _safe_float(sea.get("miss_tol500_mean"))
        sine_miss = _safe_float(sine.get("miss_tol500_mean"))
        sea_conf = _safe_float(sea.get("conf_P90_mean"))
        sine_conf = _safe_float(sine.get("conf_P90_mean"))
        if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
            continue
        if sea_miss <= 1e-12 and sine_miss <= 1e-12 and sea_conf < 500 and sine_conf < 500:
            eligible.append(g)

    if not eligible:
        return None, "无满足 drift 约束的候选"

    def drift_acc(g: str) -> float:
        sea = _safe_float(by_group[g].get("sea_abrupt4", {}).get("acc_final_mean"))
        sine = _safe_float(by_group[g].get("sine_abrupt4", {}).get("acc_final_mean"))
        vals = [v for v in [sea, sine] if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("-inf")

    best_acc = max([drift_acc(g) for g in eligible] or [float("-inf")])
    eligible2 = [g for g in eligible if drift_acc(g) >= best_acc - float(acc_tol)]
    if not eligible2:
        eligible2 = eligible

    def nd_rate(g: str) -> float:
        sea = _safe_float(by_group[g].get("sea_nodrift", {}).get("confirm_rate_per_10k_mean"))
        sine = _safe_float(by_group[g].get("sine_nodrift", {}).get("confirm_rate_per_10k_mean"))
        vals = [v for v in [sea, sine] if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("inf")

    def nd_mtfa(g: str) -> float:
        sea = _safe_float(by_group[g].get("sea_nodrift", {}).get("MTFA_win_mean"))
        sine = _safe_float(by_group[g].get("sine_nodrift", {}).get("MTFA_win_mean"))
        vals = [v for v in [sea, sine] if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("-inf")

    eligible2.sort(key=lambda g: (nd_rate(g), -nd_mtfa(g), g))
    return eligible2[0], None


def read_summary(log_path: Path) -> Dict[str, Any]:
    sp = log_path.with_suffix(".summary.json")
    return json.loads(sp.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    trackal_path = Path(args.trackal_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv(trackal_path)
    if not rows:
        print(f"[warn] empty: {trackal_path}")
        return 0

    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]

    groups: List[str] = []
    if str(args.groups).strip():
        groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    else:
        winner, err = pick_winner(rows, acc_tol=float(args.acc_tolerance))
        if err:
            print(f"[warn] winner pick failed: {err}")
        # baseline: prefer A_weighted_n20 if exists, else A_weighted_n5
        all_groups = sorted({str(r.get("group") or "") for r in rows if r.get("group")})
        baseline = "A_weighted_n20" if "A_weighted_n20" in all_groups else ("A_weighted_n5" if "A_weighted_n5" in all_groups else None)
        if baseline:
            groups.append(baseline)
        if winner:
            groups.append(str(winner))

    if not groups:
        print("[warn] no groups")
        return 0

    # select best phase per group+dataset: prefer full
    def phase_rank(r: Dict[str, str]) -> int:
        return 0 if str(r.get("phase") or "") == "full" else 1

    by_gd: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in sorted(rows, key=phase_rank):
        g = str(r.get("group") or "")
        d = str(r.get("dataset") or "")
        if g in groups and d in datasets:
            by_gd.setdefault((g, d), r)

    out_rows: List[Dict[str, Any]] = []
    for g in groups:
        for d in datasets:
            r = by_gd.get((g, d))
            if not r:
                continue
            run_index_json = r.get("run_index_json") or "{}"
            try:
                run_index = json.loads(run_index_json)
            except Exception:
                run_index = {}

            ds_info = run_index.get(d) if isinstance(run_index, dict) else None
            runs = (ds_info or {}).get("runs") if isinstance(ds_info, dict) else None
            if not isinstance(runs, list):
                continue

            cand_counts: List[Optional[float]] = []
            conf_counts: List[Optional[float]] = []
            ratios: List[Optional[float]] = []
            p50s: List[Optional[float]] = []
            p90s: List[Optional[float]] = []
            p99s: List[Optional[float]] = []
            le_alpha: List[Optional[float]] = []

            for item in runs:
                try:
                    log_path = Path(str(item.get("log_path")))
                except Exception:
                    continue
                summ = read_summary(log_path)
                ccand = _safe_float(summ.get("candidate_count_total"))
                cconf = _safe_float(summ.get("confirmed_count_total"))
                cand_counts.append(ccand)
                conf_counts.append(cconf)
                if ccand is not None and ccand > 0 and cconf is not None:
                    ratios.append(float(cconf) / float(ccand))
                else:
                    ratios.append(None)
                p50s.append(_safe_float(summ.get("perm_pvalue_p50")))
                p90s.append(_safe_float(summ.get("perm_pvalue_p90")))
                p99s.append(_safe_float(summ.get("perm_pvalue_p99")))
                le_alpha.append(_safe_float(summ.get("perm_pvalue_le_alpha_ratio")))

            out_rows.append(
                {
                    "track": "AM",
                    "dataset": d,
                    "group": g,
                    "phase": str(r.get("phase") or ""),
                    "confirm_rule": str(r.get("confirm_rule") or ""),
                    "perm_stat": str(r.get("perm_stat") or ""),
                    "delta_k": str(r.get("delta_k") or ""),
                    "perm_alpha": str(r.get("perm_alpha") or ""),
                    "perm_pre_n": str(r.get("perm_pre_n") or ""),
                    "perm_post_n": str(r.get("perm_post_n") or ""),
                    "perm_n_perm": str(r.get("perm_n_perm") or ""),
                    "n_runs": int(len(runs)),
                    "candidate_count_mean": mean(cand_counts),
                    "candidate_count_std": std(cand_counts),
                    "confirmed_count_mean": mean(conf_counts),
                    "confirmed_count_std": std(conf_counts),
                    "confirmed_over_candidate_mean": mean(ratios),
                    "perm_pvalue_p50_mean": mean(p50s),
                    "perm_pvalue_p90_mean": mean(p90s),
                    "perm_pvalue_p99_mean": mean(p99s),
                    "perm_pvalue_le_alpha_ratio_mean": mean(le_alpha),
                }
            )

    if not out_rows:
        print("[warn] no diag rows")
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
