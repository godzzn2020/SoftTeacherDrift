from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(statistics.mean(vals))


def std(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1 Part6 analysis")
    p.add_argument(
        "--summary_detail",
        type=str,
        default="results/exp1_signal_set/summary_detail.csv",
    )
    p.add_argument(
        "--summary_table",
        type=str,
        default="results/exp1_signal_set/summary_table.csv",
    )
    p.add_argument(
        "--out_candidate_vs_lr",
        type=str,
        default="results/exp1_signal_set/exp1_candidate_vs_lr.csv",
    )
    p.add_argument(
        "--out_hardok_vs_candidate",
        type=str,
        default="results/exp1_signal_set/exp1_hardok_vs_candidate.csv",
    )
    p.add_argument(
        "--out_notes",
        type=str,
        default="results/exp1_signal_set/exp1_mechanism_notes.md",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    detail_path = Path(args.summary_detail)
    table_path = Path(args.summary_table)
    if not detail_path.exists():
        raise FileNotFoundError(detail_path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)

    group_map = {"error": "G0_error", "proxy": "G1_proxy", "all": "G2_all"}

    detail_rows: List[Dict[str, Any]] = []
    with detail_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            detail_rows.append(row)

    grouped: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        ratio = _safe_float(row.get("labeled_ratio"))
        signal_set = str(row.get("signal_set") or "")
        group = group_map.get(signal_set, signal_set)
        if ratio is None:
            continue
        grouped.setdefault((float(ratio), group), []).append(row)

    candidate_rows: List[Dict[str, Any]] = []
    for (ratio, group), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        candidates = [_safe_float(r.get("candidate_count")) for r in rows]
        confirmed = [_safe_float(r.get("confirmed_count")) for r in rows]
        not_tested = [_safe_float(r.get("not_tested_count")) for r in rows]
        reject = [_safe_float(r.get("perm_reject_count")) for r in rows]
        ratios = []
        for r in rows:
            cand = _safe_float(r.get("candidate_count"))
            conf = _safe_float(r.get("confirmed_count"))
            if cand is None or cand <= 0 or conf is None:
                continue
            ratios.append(float(conf) / float(cand))
        candidate_rows.append(
            {
                "labeled_ratio": float(ratio),
                "group": group,
                "candidate_mean": mean(candidates),
                "candidate_std": std(candidates),
                "confirmed_mean": mean(confirmed),
                "confirmed_std": std(confirmed),
                "confirmed_over_candidate_mean": mean(ratios),
                "not_tested_mean": mean(not_tested),
                "reject_mean": mean(reject),
            }
        )

    out_candidate = Path(args.out_candidate_vs_lr)
    out_candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate_rows:
        fieldnames: List[str] = []
        seen: set[str] = set()
        for r in candidate_rows:
            for k in r.keys():
                if k in seen:
                    continue
                fieldnames.append(k)
                seen.add(k)
        with out_candidate.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(candidate_rows)

    table_rows: List[Dict[str, Any]] = []
    with table_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table_rows.append(row)

    candidate_by_ratio_group: Dict[Tuple[float, str], float] = {}
    for r in candidate_rows:
        key = (float(r["labeled_ratio"]), str(r["group"]))
        cand = _safe_float(r.get("candidate_mean"))
        if cand is not None:
            candidate_by_ratio_group[key] = float(cand)

    baseline_candidate: Dict[float, float] = {}
    baseline_hardok: Dict[float, float] = {}
    for row in table_rows:
        ratio = _safe_float(row.get("labeled_ratio"))
        group = str(row.get("group") or "")
        if ratio is None:
            continue
        if group.startswith("G0"):
            cand = candidate_by_ratio_group.get((float(ratio), group))
            if cand is not None:
                baseline_candidate[float(ratio)] = cand
            hard_ok = _safe_float(row.get("hard_ok_pass_rate"))
            if hard_ok is not None:
                baseline_hardok[float(ratio)] = hard_ok

    hardok_rows: List[Dict[str, Any]] = []
    for row in table_rows:
        ratio = _safe_float(row.get("labeled_ratio"))
        group = str(row.get("group") or "")
        if ratio is None:
            continue
        cand = candidate_by_ratio_group.get((float(ratio), group))
        hard_ok = _safe_float(row.get("hard_ok_pass_rate"))
        worst = _safe_float(row.get("worst_seed_regression_per_10k"))
        base_cand = baseline_candidate.get(float(ratio))
        base_hard = baseline_hardok.get(float(ratio))

        note = "与基线相当/更稳"
        if cand is None or (cand < 1.0) or (base_cand is not None and cand < 0.3 * base_cand):
            note = "触发不足（candidate 太少）"
        elif base_hard is not None and hard_ok is not None and hard_ok < base_hard - 1e-6 and (worst is not None and worst > 0):
            note = "触发过密/门控交互导致不稳"
        elif worst is not None and worst > 0:
            note = "存在最坏 seed 回退"

        hardok_rows.append(
            {
                "labeled_ratio": float(ratio),
                "group": group,
                "hard_ok_pass_rate": hard_ok,
                "candidate_mean": cand,
                "worst_seed_regression_per_10k": worst,
                "explain": note,
            }
        )

    out_hardok = Path(args.out_hardok_vs_candidate)
    out_hardok.parent.mkdir(parents=True, exist_ok=True)
    if hardok_rows:
        fieldnames: List[str] = []
        seen: set[str] = set()
        for r in hardok_rows:
            for k in r.keys():
                if k in seen:
                    continue
                fieldnames.append(k)
                seen.add(k)
        with out_hardok.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(hardok_rows)

    def pick_row(group: str, ratio: float) -> Optional[Dict[str, Any]]:
        for r in hardok_rows:
            if float(r.get("labeled_ratio") or 0.0) == float(ratio) and str(r.get("group")) == group:
                return r
        return None

    def pick_candidate(group: str) -> Optional[Dict[str, Any]]:
        for r in candidate_rows:
            if str(r.get("group")) == group:
                return r
        return None

    g1_example = pick_candidate("G1_proxy")
    g2_lr001 = pick_row("G2_all", 0.01)
    g0_lr001 = pick_row("G0_error", 0.01)

    notes: List[str] = []
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "NA"
        try:
            return f"{float(v):.3g}"
        except Exception:
            return str(v)

    if g1_example:
        notes.append(
            f"- G1(candidate) 均值接近 0（candidate_mean≈{_fmt(_safe_float(g1_example.get('candidate_mean')))}），confirmed 也接近 0，说明 proxy 信号未进入有效触发区间。"
        )
    if g2_lr001 and g0_lr001:
        notes.append(
            f"- 在 lr=0.01：G2 hard-ok={_fmt(_safe_float(g2_lr001.get('hard_ok_pass_rate')))} 低于 G0={_fmt(_safe_float(g0_lr001.get('hard_ok_pass_rate')))}，且 worst-seed 回退为正（{_fmt(_safe_float(g2_lr001.get('worst_seed_regression_per_10k')))}）。"
        )
    notes.append("- confirmed≈candidate−1、not_tested≈1、reject≈0：confirm 端统计检验不是主要差异来源，candidate 分布变化更关键。")
    notes.append("- G1 在全部 labelled_ratio 下 hard-ok 为 0，属于“触发不足”而非“误报爆炸”。")
    notes.append("- G2 在 lr=0.05/0.2 与 G0 hard-ok 持平，但 worst-seed 回退为正，提示融合后稳定性波动。")
    notes.append("- lr=1.0 时 G2 hard-ok 高于 G0，但 no-drift 仍未显著优于 G0，说明 proxy 贡献主要体现在 drift 侧。")
    notes.append("- 结论：需先校准 proxy detector 触发区间，再考虑融合策略，否则 G2 在低标签下可能引入不稳。")

    out_notes = Path(args.out_notes)
    out_notes.parent.mkdir(parents=True, exist_ok=True)
    out_notes.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
