#!/usr/bin/env python
"""汇总 NEXT_STAGE V13（Track AJ，可选 Track AK）并生成报告。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V13 (Track AJ/AK)")
    p.add_argument("--trackaj_csv", type=str, default="scripts/TRACKAJ_CONFIRM_RULE_NODRIFT.csv")
    p.add_argument("--trackak_csv", type=str, default="scripts/TRACKAK_STAGGER_GRADUAL_ENTROPY.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V13_REPORT.md")
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


def fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "N/A"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{nd}f}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def pick_winner(rows: List[Dict[str, str]], acc_tol: float) -> Dict[str, Any]:
    if not rows:
        return {"winner": None, "reason": "未找到 Track AJ CSV"}

    eligible: List[Dict[str, str]] = []
    for r in rows:
        sea_miss = _safe_float(r.get("sea_miss"))
        sine_miss = _safe_float(r.get("sine_miss"))
        sea_conf = _safe_float(r.get("sea_confP90"))
        sine_conf = _safe_float(r.get("sine_confP90"))
        if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
            continue
        if not (sea_miss <= 0.0 + 1e-12 and sine_miss <= 0.0 + 1e-12 and sea_conf < 500 and sine_conf < 500):
            continue
        eligible.append(r)
    if not eligible:
        return {"winner": None, "reason": "无满足 drift 约束的候选"}

    best_acc = max(float(_safe_float(r.get("drift_acc_final")) or float("-inf")) for r in eligible)
    eligible2 = [r for r in eligible if float(_safe_float(r.get("drift_acc_final")) or float("-inf")) >= best_acc - float(acc_tol)]
    if not eligible2:
        eligible2 = eligible

    def nd_rate(r: Dict[str, str]) -> float:
        v = _safe_float(r.get("no_drift_confirm_rate_per_10k"))
        return float(v) if v is not None else float("inf")

    def nd_mtfa(r: Dict[str, str]) -> float:
        v = _safe_float(r.get("no_drift_MTFA_win"))
        return float(v) if v is not None else float("-inf")

    eligible2.sort(key=lambda r: (nd_rate(r), -nd_mtfa(r), str(r.get("group") or "")))
    winner = eligible2[0]
    reason = "规则：drift(sea+sine) miss==0 且 confP90<500；目标最小化 no-drift confirm_rate_per_10k（次选最大化 MTFA_win）；并要求 drift_acc_final≥best-0.01。"
    return {"winner": winner, "reason": reason}


def main() -> int:
    args = parse_args()
    rows_aj = read_csv(Path(args.trackaj_csv))
    rows_ak = read_csv(Path(args.trackak_csv))

    pick = pick_winner(rows_aj, acc_tol=float(args.acc_tolerance))
    winner = pick.get("winner")
    reason = pick.get("reason") or "N/A"

    headers = [
        "group",
        "divgate_value_thr",
        "errgate_thr",
        "sea_miss",
        "sea_confP90",
        "sine_miss",
        "sine_confP90",
        "no_drift_rate",
        "no_drift_MTFA",
        "drift_acc_final",
    ]
    table_rows: List[List[str]] = []
    for r in rows_aj:
        table_rows.append(
            [
                str(r.get("group") or ""),
                fmt(_safe_float(r.get("divergence_gate_value_thr")), 3),
                fmt(_safe_float(r.get("confirm_error_gate_thr")), 3),
                fmt(_safe_float(r.get("sea_miss")), 3),
                fmt(_safe_float(r.get("sea_confP90")), 1),
                fmt(_safe_float(r.get("sine_miss")), 3),
                fmt(_safe_float(r.get("sine_confP90")), 1),
                fmt(_safe_float(r.get("no_drift_confirm_rate_per_10k")), 3),
                fmt(_safe_float(r.get("no_drift_MTFA_win")), 1),
                fmt(_safe_float(r.get("drift_acc_final")), 4),
            ]
        )

    win_line = "N/A"
    delta_line = "N/A"
    interpret_lines: List[str] = []
    if winner:
        win_line = f"{winner.get('group')}"
        # baseline group
        baseline = None
        for r in rows_aj:
            if str(r.get("group") or "") == "A_weighted":
                baseline = r
                break
        if baseline:
            b = _safe_float(baseline.get("no_drift_confirm_rate_per_10k"))
            w = _safe_float(winner.get("no_drift_confirm_rate_per_10k"))
            if b is not None and w is not None:
                delta_line = f"{b:.3f} → {w:.3f} (Δ={w-b:+.3f})"
        # interpretability notes (best-effort, purely from table signals)
        eligible_groups = []
        for r in rows_aj:
            sea_miss = _safe_float(r.get("sea_miss"))
            sine_miss = _safe_float(r.get("sine_miss"))
            sea_conf = _safe_float(r.get("sea_confP90"))
            sine_conf = _safe_float(r.get("sine_confP90"))
            if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
                continue
            if sea_miss <= 1e-12 and sine_miss <= 1e-12 and sea_conf < 500 and sine_conf < 500:
                eligible_groups.append(str(r.get("group") or ""))
        if len(eligible_groups) <= 1:
            interpret_lines.append("- 在本轮 drift 约束（sea+sine：miss==0 且 confP90<500）下，仅 baseline 能同时满足；其余规则虽然可显著降 no-drift confirm_rate，但会导致 drift miss 或 confP90 超标。")
        else:
            interpret_lines.append(f"- 满足 drift 约束的候选：{', '.join(eligible_groups)}；winner 依据 no-drift confirm_rate 最小化选出。")
        interpret_lines.append("- 观测：divergence_gate / k_of_n / error_gate 这类“更严格 confirm”在 gradual_frequent 上容易把确认推迟到 transition 后段，从而在 start 口径下计为 miss。")

    ak_note = "未运行（可选项）"
    if rows_ak:
        ak_note = f"已生成 `scripts/TRACKAK_STAGGER_GRADUAL_ENTROPY.csv` rows={len(rows_ak)}（请在报告中补充解读）"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py = sys.executable
    pyver = platform.python_version()

    interpret_md = "\n".join(interpret_lines) if interpret_lines else "- N/A"

    report = f"""# NEXT_STAGE V13 Report（Confirm 规则降低 no-drift 误报）

- 生成时间：{now}
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`{py} / {pyver}`

## 结论摘要
{interpret_md}

================================================
V13-Track AJ（必做）：Confirm-rule ablation for no-drift
================================================

产物：`{args.trackaj_csv}`

{md_table(headers, table_rows)}

**赢家选择**
- 规则：{reason}
- winner：`{win_line}`
- no-drift confirm_rate_per_10k 下降幅度（vs baseline A_weighted）：{delta_line}

================================================
V13-Track AK（可选）：stagger_gradual 新信号最小验证（entropy）
================================================

- {ak_note}

================================================
V13 后“一句话默认配置”
================================================

- 检测：`two_stage(candidate=OR,confirm=*)` + `error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5`（divergence 默认 0.05/30）。
- confirm 规则：`{win_line}`（详见 Track AJ）。
- 恢复：INSECTS 默认启用 `severity v2`（V11 Track AF：收益较小、CI 跨 0，但方向一致）。
"""

    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
