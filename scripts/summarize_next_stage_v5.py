#!/usr/bin/env python
"""汇总 NEXT_STAGE V5（Track M/N）并生成报告。"""

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
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V5 (Track M/N)")
    p.add_argument("--trackm_csv", type=str, default="scripts/TRACKM_LATENCY_SWEEP.csv")
    p.add_argument("--trackn_csv", type=str, default="scripts/TRACKN_GATING_RULE.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V5_REPORT.md")
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


def percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
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


def group_rows(rows: List[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        k = str(r.get(key, "") or "")
        if not k:
            continue
        out.setdefault(k, []).append(r)
    return out


def summarize_track_m(rows: List[Dict[str, str]], acc_tol: float) -> Tuple[str, Dict[str, Any]]:
    if not rows:
        return "_N/A_", {"note": "未找到 Track M CSV"}

    by_cfg = group_rows([r for r in rows if (r.get("dataset") == "sea_abrupt4")], "config_tag")
    agg_rows: List[Dict[str, Any]] = []
    for tag, rs in sorted(by_cfg.items()):
        # run-level 去重（同 run 有多条 drift 行）
        seen_run: set[Tuple[str, str]] = set()
        acc_final_list: List[Optional[float]] = []
        acc_min_list: List[Optional[float]] = []
        mdr_win_list: List[Optional[float]] = []
        mtfa_win_list: List[Optional[float]] = []
        mtr_win_list: List[Optional[float]] = []
        mdr_tol_list: List[Optional[float]] = []
        for r in rs:
            run_id = str(r.get("run_id") or "")
            seed = str(r.get("seed") or "")
            key = (run_id, seed)
            if key in seen_run:
                continue
            seen_run.add(key)
            acc_final_list.append(_safe_float(r.get("acc_final")))
            acc_min_list.append(_safe_float(r.get("acc_min")))
            mdr_win_list.append(_safe_float(r.get("MDR_win")))
            mtfa_win_list.append(_safe_float(r.get("MTFA_win")))
            mtr_win_list.append(_safe_float(r.get("MTR_win")))
            mdr_tol_list.append(_safe_float(r.get("MDR_tol500")))

        delays_cand = [_safe_float(r.get("delay_candidate")) for r in rs]
        delays_conf = [_safe_float(r.get("delay_confirmed")) for r in rs]
        miss_tol = [_safe_float(r.get("miss_tol500")) for r in rs]

        sample = rs[0]
        agg_rows.append(
            {
                "config_tag": tag,
                "trigger_mode": str(sample.get("trigger_mode") or ""),
                "confirm_theta": _safe_float(sample.get("confirm_theta")),
                "confirm_window": _safe_int(sample.get("confirm_window")),
                "monitor_preset": str(sample.get("monitor_preset") or ""),
                "ph_error_threshold": _safe_float(sample.get("ph_error_threshold")),
                "ph_error_min_instances": _safe_int(sample.get("ph_error_min_instances")),
                "ph_div_threshold": _safe_float(sample.get("ph_divergence_threshold")),
                "ph_div_min_instances": _safe_int(sample.get("ph_divergence_min_instances")),
                "n_runs": len(seen_run),
                "acc_final_mean": mean(acc_final_list),
                "acc_final_std": std(acc_final_list),
                "acc_min_mean": mean(acc_min_list),
                "acc_min_std": std(acc_min_list),
                "MDR_win_mean": mean(mdr_win_list),
                "MTFA_win_mean": mean(mtfa_win_list),
                "MTR_win_mean": mean(mtr_win_list),
                "MDR_tol500_mean": mean(mdr_tol_list),
                "miss_tol500_mean": mean(miss_tol),
                "cand_p50": percentile(delays_cand, 0.5),
                "cand_p90": percentile(delays_cand, 0.9),
                "cand_p99": percentile(delays_cand, 0.99),
                "conf_p50": percentile(delays_conf, 0.5),
                "conf_p90": percentile(delays_conf, 0.9),
                "conf_p99": percentile(delays_conf, 0.99),
            }
        )

    # 选择推荐（two_stage）
    two_stage = [r for r in agg_rows if r.get("trigger_mode") == "two_stage"]
    best_cfg: Optional[Dict[str, Any]] = None
    reason = "N/A"
    if two_stage:
        best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in two_stage)
        eligible = [r for r in two_stage if float(r["acc_final_mean"] or float("-inf")) >= best_acc - acc_tol]
        if not eligible:
            eligible = two_stage
        eligible.sort(
            key=lambda r: (
                float(r["miss_tol500_mean"] if r["miss_tol500_mean"] is not None else 1.0),
                float(r["conf_p90"] if r["conf_p90"] is not None else float("inf")),
                -float(r["acc_min_mean"] if r["acc_min_mean"] is not None else float("-inf")),
                -float(r["MTFA_win_mean"] if r["MTFA_win_mean"] is not None else float("-inf")),
                -float(r["MTR_win_mean"] if r["MTR_win_mean"] is not None else float("-inf")),
                str(r["config_tag"]),
            )
        )
        best_cfg = eligible[0]
        reason = (
            f"规则：acc_final_mean≥best-{acc_tol}，先最小化 miss_tol500_mean，再最小化 conf_P90，"
            f"再最大化 acc_min_mean（再看 MTFA/MTR）；best_acc={best_acc:.4f}"
        )

    table_rows: List[List[str]] = []
    for r in sorted(agg_rows, key=lambda x: (str(x["trigger_mode"]), float(x["miss_tol500_mean"] or 9), str(x["config_tag"]))):
        table_rows.append(
            [
                str(r["config_tag"]),
                str(r["trigger_mode"]),
                fmt(_safe_float(r.get("confirm_theta")), 2),
                str(r.get("confirm_window") if r.get("confirm_window") is not None else "N/A"),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                fmt(r.get("MDR_win_mean"), 3),
                fmt(r.get("MTFA_win_mean"), 1),
                fmt(r.get("MTR_win_mean"), 3),
                fmt(r.get("miss_tol500_mean"), 3),
                fmt(r.get("MDR_tol500_mean"), 3),
                fmt(r.get("cand_p90"), 1),
                fmt(r.get("conf_p90"), 1),
                fmt(_safe_float(r.get("ph_error_threshold")), 3),
                str(r.get("ph_error_min_instances") if r.get("ph_error_min_instances") is not None else "N/A"),
                fmt(_safe_float(r.get("ph_div_threshold")), 3),
                str(r.get("ph_div_min_instances") if r.get("ph_div_min_instances") is not None else "N/A"),
            ]
        )
    headers = [
        "config_tag",
        "trigger",
        "theta",
        "window",
        "acc_final",
        "acc_min",
        "MDR_win",
        "MTFA_win",
        "MTR_win",
        "miss_tol500",
        "MDR_tol500",
        "cand_P90",
        "conf_P90",
        "err_thr",
        "err_min",
        "div_thr",
        "div_min",
    ]
    table = md_table(headers, table_rows)
    return table, {"best": best_cfg, "reason": reason}


def summarize_track_n(rows: List[Dict[str, str]], acc_tol: float) -> Tuple[str, Dict[str, Any]]:
    if not rows:
        return "_N/A_", {"note": "未找到 Track N CSV"}
    by_group = group_rows([r for r in rows if (r.get("dataset") == "INSECTS_abrupt_balanced")], "group")
    agg: List[Dict[str, Any]] = []
    for group, rs in sorted(by_group.items()):
        acc_final = [_safe_float(r.get("acc_final")) for r in rs]
        acc_min = [_safe_float(r.get("acc_min")) for r in rs]
        post_mean = [_safe_float(r.get("post_mean_acc")) for r in rs]
        post_min = [_safe_float(r.get("post_min_acc")) for r in rs]
        rec = [_safe_float(r.get("recovery_time_to_pre90")) for r in rs]
        agg.append(
            {
                "group": group,
                "m": _safe_int(rs[0].get("severity_gate_min_streak")),
                "n_runs": len(rs),
                "acc_final_mean": mean(acc_final),
                "acc_final_std": std(acc_final),
                "acc_min_mean": mean(acc_min),
                "acc_min_std": std(acc_min),
                "post_mean_mean": mean(post_mean),
                "post_mean_std": std(post_mean),
                "post_min_mean": mean(post_min),
                "post_min_std": std(post_min),
                "rec_mean": mean(rec),
                "rec_std": std(rec),
            }
        )

    best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in agg) if agg else float("-inf")
    eligible = [r for r in agg if float(r["acc_final_mean"] or float("-inf")) >= best_acc - acc_tol]
    if not eligible:
        eligible = agg
    eligible.sort(
        key=lambda r: (
            -float(r["post_min_mean"] if r["post_min_mean"] is not None else float("-inf")),
            -float(r["acc_min_mean"] if r["acc_min_mean"] is not None else float("-inf")),
            float(r["rec_mean"] if r["rec_mean"] is not None else float("inf")),
            str(r["group"]),
        )
    )
    best = eligible[0] if eligible else None
    reason = f"规则：acc_final_mean≥best-{acc_tol}，优先最大化 post_min@W1000，再最大化 acc_min；best_acc={best_acc:.4f}"

    table_rows: List[List[str]] = []
    for r in sorted(agg, key=lambda x: str(x["group"])):
        table_rows.append(
            [
                str(r["group"]),
                str(r["m"] if r["m"] is not None else "N/A"),
                str(r["n_runs"]),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                fmt_mu_std(r.get("post_mean_mean"), r.get("post_mean_std"), 4),
                fmt_mu_std(r.get("post_min_mean"), r.get("post_min_std"), 4),
                fmt_mu_std(r.get("rec_mean"), r.get("rec_std"), 1),
            ]
        )
    headers = [
        "group",
        "m",
        "n_runs",
        "acc_final",
        "acc_min",
        "post_mean@W1000",
        "post_min@W1000",
        "recovery_time_to_pre90",
    ]
    return md_table(headers, table_rows), {"best": best, "reason": reason}


def main() -> int:
    args = parse_args()
    trackm_path = Path(args.trackm_csv)
    trackn_path = Path(args.trackn_csv)

    rows_m = read_csv(trackm_path)
    rows_n = read_csv(trackn_path)

    table_m, sel_m = summarize_track_m(rows_m, float(args.acc_tolerance))
    table_n, sel_n = summarize_track_n(rows_n, float(args.acc_tolerance))

    python_exec = sys.executable
    python_ver = platform.python_version()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# NEXT_STAGE V5 Report (Latency Optimization + Gating Rule)")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`")
    lines.append(f"- Python：`{python_exec}` / `{python_ver}`")
    lines.append("")
    lines.append("## 关键口径（Latency → tol500）")
    lines.append("- 时间轴：统一使用 `sample_idx`；tol500 口径匹配条件为 `confirmed_step <= drift_pos + 500`。")
    lines.append("- 经验现象：`MDR_win≈0` 但 `miss_tol500/MDR_tol500` 高通常不是完全漏检，而是 **P90/P99 延迟 > 500 的晚检**。")
    lines.append("")
    lines.append("## Track M：Latency 优化（sea_abrupt4）")
    lines.append(table_m)
    lines.append("")
    lines.append("**选择规则（写入论文/报告）**")
    lines.append(f"- {sel_m.get('reason','N/A')}")
    best_m = sel_m.get("best")
    if best_m:
        lines.append("**推荐 PH 参数（sea_abrupt4）**")
        lines.append(
            "- "
            f"config_tag=`{best_m['config_tag']}`, monitor_preset=`{best_m['monitor_preset']}`, "
            f"theta={fmt(_safe_float(best_m.get('confirm_theta')),2)}, window={best_m.get('confirm_window')}"
        )
        lines.append(
            "- "
            f"效果摘要：miss_tol500={fmt(best_m.get('miss_tol500_mean'),3)}, conf_P90={fmt(best_m.get('conf_p90'),1)}, "
            f"acc_final={fmt(best_m.get('acc_final_mean'),4)}, acc_min={fmt(best_m.get('acc_min_mean'),4)}"
        )
    else:
        lines.append("- TODO：未能从 Track M 结果中选出推荐配置（请先运行 experiments/trackM_latency_sweep.py）。")
    lines.append("")
    lines.append("## Track N：Gating 规则化（INSECTS_abrupt_balanced）")
    lines.append(table_n)
    lines.append("")
    lines.append("**选择规则（可复现）**")
    lines.append(f"- {sel_n.get('reason','N/A')}")
    best_n = sel_n.get("best")
    if best_n:
        lines.append("**推荐 gating 强度**")
        lines.append(
            "- "
            f"group=`{best_n['group']}`, m={best_n.get('m')}（post_min@W1000 最大且 acc_final 不降超过阈值）"
        )
    else:
        lines.append("- TODO：未能从 Track N 结果中选出推荐 gating（请先运行 experiments/trackN_gating_rule.py）。")
    lines.append("")
    lines.append("## 最终默认配置（论文可写）")
    if best_m and best_n:
        lines.append(
            "- "
            "触发：two_stage（candidate=OR，confirm=weighted），并使用 Track M 的 tuned PageHinkley（`monitor_preset` 记录完整覆盖参数）。"
        )
        lines.append(
            "- "
            f"严重度：severity v2 + confirmed drift gating（推荐 `{best_n['group']}`），用于缓解负迁移并提升 post_min@W1000 / acc_min。"
        )
    else:
        lines.append("- TODO：等待 Track M/N 完整结果后再写入最终默认配置。")
    lines.append("")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote report -> {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

