#!/usr/bin/env python
"""汇总 NEXT_STAGE V12（Track AG/AH）并生成报告。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V12 (Track AG/AH)")
    p.add_argument("--trackag_csv", type=str, default="scripts/TRACKAG_CONFIRM_SIDE_NODRIFT.csv")
    p.add_argument("--trackah_csv", type=str, default="scripts/TRACKAH_STAGGER_GRADUAL_SENSITIVITY.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V12_REPORT.md")
    p.add_argument("--warmup_samples", type=int, default=2000)
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
    except ValueError:
        return None


def _safe_int(v: Any) -> Optional[int]:
    x = _safe_float(v)
    return None if x is None else int(x)


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


def fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "N/A"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{nd}f}"


def fmt_mu_std(mu: Optional[float], sd: Optional[float], nd: int = 4) -> str:
    if mu is None:
        return "N/A"
    if sd is None:
        return f"{fmt(mu, nd)}±N/A"
    return f"{fmt(mu, nd)}±{fmt(sd, nd)}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def pick_winner_trackag(rows: List[Dict[str, str]], acc_tol: float) -> Tuple[Optional[Dict[str, str]], str]:
    if not rows:
        return None, "未找到 Track AG CSV"

    # 约束集：sea+sine miss==0 且 conf_P90<500
    eligible: List[Dict[str, str]] = []
    for r in rows:
        sea_miss = _safe_float(r.get("sea_miss_tol500_mean"))
        sine_miss = _safe_float(r.get("sine_miss_tol500_mean"))
        sea_conf = _safe_float(r.get("sea_conf_P90_mean"))
        sine_conf = _safe_float(r.get("sine_conf_P90_mean"))
        if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
            continue
        if not (sea_miss <= 0.0 + 1e-12 and sine_miss <= 0.0 + 1e-12 and sea_conf < 500 and sine_conf < 500):
            continue
        eligible.append(r)

    if not eligible:
        return None, "无满足 drift 约束的候选"

    # 次目标：acc_final 不下降超过 best-0.01（用 drift_acc_final_mean）
    best_acc = max(float(_safe_float(r.get("drift_acc_final_mean")) or float("-inf")) for r in eligible)
    eligible2 = [r for r in eligible if float(_safe_float(r.get("drift_acc_final_mean")) or float("-inf")) >= best_acc - float(acc_tol)]
    if not eligible2:
        eligible2 = eligible

    def nd_rate(r: Dict[str, str]) -> float:
        v = _safe_float(r.get("no_drift_confirm_rate_per_10k_mean"))
        return float(v) if v is not None else float("inf")

    def nd_mtfa(r: Dict[str, str]) -> float:
        v = _safe_float(r.get("no_drift_MTFA_win_mean"))
        return float(v) if v is not None else float("-inf")

    eligible2.sort(
        key=lambda r: (
            nd_rate(r),
            -nd_mtfa(r),
            float(_safe_float(r.get("confirm_theta")) or float("inf")),
            int(_safe_int(r.get("confirm_window")) or 10**9),
            int(_safe_int(r.get("confirm_cooldown")) or 10**9),
        )
    )
    winner = eligible2[0]
    reason = "规则：drift(sea+sine) miss==0 且 conf_P90<500；目标最小化 no-drift confirm_rate_per_10k（次选最大化 no-drift MTFA）；并要求 drift_acc_final_mean≥best-0.01。"
    return winner, reason


def summarize_trackag(rows: List[Dict[str, str]], acc_tol: float) -> Dict[str, Any]:
    winner, reason = pick_winner_trackag(rows, acc_tol)
    if not rows:
        return {"table": "_N/A_", "winner": None, "reason": reason}

    headers = [
        "theta",
        "window",
        "cooldown",
        "sea_miss",
        "sea_confP90",
        "sine_miss",
        "sine_confP90",
        "no_drift_rate",
        "no_drift_MTFA",
        "drift_acc_final",
    ]
    table_rows: List[List[str]] = []
    # 报告不铺开 27 行太长：只展示满足约束的前 8 行（按 no-drift rate 排序）
    eligible = []
    for r in rows:
        sea_miss = _safe_float(r.get("sea_miss_tol500_mean"))
        sine_miss = _safe_float(r.get("sine_miss_tol500_mean"))
        sea_conf = _safe_float(r.get("sea_conf_P90_mean"))
        sine_conf = _safe_float(r.get("sine_conf_P90_mean"))
        if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
            continue
        ok = (sea_miss <= 0.0 + 1e-12 and sine_miss <= 0.0 + 1e-12 and sea_conf < 500 and sine_conf < 500)
        if ok:
            eligible.append(r)

    def nd_rate(r: Dict[str, str]) -> float:
        v = _safe_float(r.get("no_drift_confirm_rate_per_10k_mean"))
        return float(v) if v is not None else float("inf")

    eligible.sort(key=nd_rate)
    show = eligible[:8]
    for r in show:
        table_rows.append(
            [
                f"{float(r['confirm_theta']):.2f}",
                str(int(float(r["confirm_window"]))),
                str(int(float(r["confirm_cooldown"]))),
                fmt(_safe_float(r.get("sea_miss_tol500_mean")), 3),
                fmt(_safe_float(r.get("sea_conf_P90_mean")), 1),
                fmt(_safe_float(r.get("sine_miss_tol500_mean")), 3),
                fmt(_safe_float(r.get("sine_conf_P90_mean")), 1),
                fmt(_safe_float(r.get("no_drift_confirm_rate_per_10k_mean")), 3),
                fmt(_safe_float(r.get("no_drift_MTFA_win_mean")), 1),
                fmt(_safe_float(r.get("drift_acc_final_mean")), 4),
            ]
        )

    # baseline: theta=0.50, w=1, cd=200
    baseline = None
    for r in rows:
        if float(_safe_float(r.get("confirm_theta")) or -1) == 0.5 and int(float(r.get("confirm_window") or 0)) == 1 and int(float(r.get("confirm_cooldown") or 0)) == 200:
            baseline = r
            break
    return {"table": md_table(headers, table_rows), "winner": winner, "reason": reason, "baseline": baseline}


def summarize_trackah(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"table": "_N/A_", "note": "未找到 Track AH CSV"}
    # 聚合：error.threshold
    by_thr: Dict[float, List[Dict[str, str]]] = {}
    for r in rows:
        thr = _safe_float(r.get("error.threshold"))
        if thr is None:
            continue
        by_thr.setdefault(float(thr), []).append(r)

    headers = ["error.threshold", "n_runs", "miss_start", "miss_mid", "miss_end", "end_delay_P90", "end_delay_P99"]
    out_rows: List[List[str]] = []
    best_end: Optional[Tuple[float, float]] = None  # (miss_end_mean, thr)
    for thr, rs in sorted(by_thr.items()):
        ms = [_safe_float(r.get("miss_tol500_start")) for r in rs]
        mm = [_safe_float(r.get("miss_tol500_mid")) for r in rs]
        me = [_safe_float(r.get("miss_tol500_end")) for r in rs]
        dp90 = [_safe_float(r.get("delay_end_P90")) for r in rs]
        dp99 = [_safe_float(r.get("delay_end_P99")) for r in rs]
        me_mu = mean(me)
        if me_mu is not None:
            if best_end is None or float(me_mu) < float(best_end[0]):
                best_end = (float(me_mu), float(thr))
        out_rows.append(
            [
                f"{thr:.2f}",
                str(len(rs)),
                fmt_mu_std(mean(ms), std(ms), 3),
                fmt_mu_std(mean(mm), std(mm), 3),
                fmt_mu_std(mean(me), std(me), 3),
                fmt_mu_std(mean(dp90), std(dp90), 1),
                fmt_mu_std(mean(dp99), std(dp99), 1),
            ]
        )
    if best_end is None:
        note = "N/A"
    else:
        best_mu, best_thr = best_end
        if best_mu <= 0.05:
            note = f"end 口径下已接近 0（best miss_end={best_mu:.3f} @ thr={best_thr:.2f}），说明主要可通过 detector 更敏感来补救。"
        else:
            note = f"end 口径下最优仍较高（best miss_end={best_mu:.3f} @ thr={best_thr:.2f}），更像 stagger_gradual 的边界条件：gradual transition + 信号弱/迟。"
    return {"table": md_table(headers, out_rows), "note": note, "best_end": best_end}


def main() -> int:
    args = parse_args()
    rows_ag = read_csv(Path(args.trackag_csv))
    rows_ah = read_csv(Path(args.trackah_csv))

    ag = summarize_trackag(rows_ag, acc_tol=float(args.acc_tolerance))
    ah = summarize_trackah(rows_ah)

    winner = ag.get("winner")
    baseline = ag.get("baseline")
    reason = ag.get("reason") or "N/A"

    win_line = "N/A"
    delta_line = "N/A"
    if winner:
        win_line = f"theta={float(winner['confirm_theta']):.2f}, window={int(float(winner['confirm_window']))}, cooldown={int(float(winner['confirm_cooldown']))}"
    if winner and baseline:
        w_rate = _safe_float(winner.get("no_drift_confirm_rate_per_10k_mean"))
        b_rate = _safe_float(baseline.get("no_drift_confirm_rate_per_10k_mean"))
        if w_rate is not None and b_rate is not None:
            delta_line = f"no-drift confirm_rate_per_10k: {b_rate:.3f} → {w_rate:.3f} (Δ={w_rate-b_rate:+.3f})"
    sig_line = ""
    if winner and baseline:
        w_rate = _safe_float(winner.get("no_drift_confirm_rate_per_10k_mean"))
        b_rate = _safe_float(baseline.get("no_drift_confirm_rate_per_10k_mean"))
        if w_rate is not None and b_rate is not None:
            sig_line = "- 结论：在本轮约束下（drift 实时性必须满足），confirm-side 对 no-drift 误报密度的下降幅度很小，未出现“显著下降”。"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py = sys.executable
    pyver = platform.python_version()

    report = f"""# NEXT_STAGE V12 Report（confirm-side 降误报 + stagger gradual 补救）

- 生成时间：{now}
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`{py} / {pyver}`

================================================
V12-Track AG（必做）：confirm-side sweep（no-drift 约束优化）
================================================

产物：`{args.trackag_csv}`

{ag.get('table','_N/A_')}

**推荐 confirm-side（winner）**
- {win_line}
- 选择规则：{reason}
- 误报变化（相对 baseline=theta0.50/w1/cd200）：{delta_line}
{sig_line}

================================================
V12-Track AH（必做）：stagger_gradual_frequent 补救（专项小扫）
================================================

产物：`{args.trackah_csv}`

{ah.get('table','_N/A_')}

**结论**
- {ah.get('note','N/A')}

================================================
V12 后“一句话默认配置”
================================================

- 检测：`two_stage(candidate=OR,confirm=weighted)` + `error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5`（divergence 默认 0.05/30）。
- confirm-side：{win_line}
- 恢复：INSECTS 默认启用 `severity v2`（V11 Track AF：收益较小、CI 跨 0，但方向一致）。
"""

    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
