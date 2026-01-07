#!/usr/bin/env python
"""自动汇总 Track F/G/H（并生成 NEXT_STAGE_REPORT.md）。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize next stage Track F/G/H")
    parser.add_argument("--trackf_csv", type=str, default="scripts/TRACKF_THRESHOLD_SWEEP.csv")
    parser.add_argument("--trackg_csv", type=str, default="scripts/TRACKG_TWO_STAGE.csv")
    parser.add_argument("--trackh_csv", type=str, default="scripts/TRACKH_SEVERITY_GATING.csv")
    parser.add_argument("--fig_dir", type=str, default="scripts/figures")
    parser.add_argument("--out_md", type=str, default="scripts/NEXT_STAGE_REPORT.md")
    parser.add_argument("--theta_target_mdr", type=float, default=0.05, help="推荐 theta 的目标 MDR 上限（win 口径）")
    parser.add_argument("--acc_tolerance", type=float, default=0.01, help="推荐 theta 的 acc_final 允许下降幅度")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(v: object) -> Optional[int]:
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


def fmt_ms(v: Optional[float], nd: int = 1) -> str:
    return fmt(v, nd)


def fmt_mu_std(mu: Optional[float], sd: Optional[float], nd: int = 4) -> str:
    if mu is None:
        return "N/A"
    if sd is None:
        return f"{fmt(mu, nd)}±N/A"
    return f"{fmt(mu, nd)}±{fmt(sd, nd)}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def recommend_theta(
    items: List[Dict[str, object]],
    target_mdr: float,
    acc_tol: float,
) -> Tuple[Optional[float], str]:
    # 策略：先找 acc_final_mean 的最好值，再在 acc 下降不超过 acc_tol 的候选里，选 MDR<=target_mdr 且 MTR 最大的 theta；
    # 若无 MDR<=target_mdr，则选 MTR 最大（仍受 acc_tol 约束）；若仍为空，退回 acc 最大的 theta。
    best_acc = max([float(it["acc_final_mean"]) for it in items if it.get("acc_final_mean") is not None] or [float("nan")])
    if math.isnan(best_acc):
        return None, "无法推荐：缺少 acc_final_mean"
    candidates = [it for it in items if it.get("theta") is not None and it.get("acc_final_mean") is not None and float(it["acc_final_mean"]) >= best_acc - acc_tol]
    if not candidates:
        return None, "无法推荐：无候选"
    ok = [it for it in candidates if it.get("MDR_win_mean") is not None and float(it["MDR_win_mean"]) <= target_mdr]
    pool = ok if ok else candidates
    pool.sort(key=lambda x: (float(x.get("MTR_win_mean") or -1e9), float(x.get("MTFA_win_mean") or -1e9)), reverse=True)
    chosen = pool[0]
    theta = float(chosen["theta"])
    reason = f"规则：在 acc_final_mean≥best-{acc_tol} 的 theta 中，优先 MDR_win≤{target_mdr}，再最大化 MTR_win（次选 MTFA_win）"
    return theta, reason


def main() -> int:
    args = parse_args()
    trackf_path = Path(args.trackf_csv)
    trackg_path = Path(args.trackg_csv)
    trackh_path = Path(args.trackh_csv)
    fig_dir = Path(args.fig_dir)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    f_rows = read_csv(trackf_path)
    g_rows = read_csv(trackg_path)
    h_rows = read_csv(trackh_path)

    # --- Track F：聚合（dataset+theta） ---
    f_group: Dict[Tuple[str, float], List[Dict[str, str]]] = {}
    for r in f_rows:
        ds = r.get("dataset", "")
        theta = _safe_float(r.get("theta"))
        if not ds or theta is None:
            continue
        f_group.setdefault((ds, float(theta)), []).append(r)

    f_agg_by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for (ds, theta), rows in sorted(f_group.items(), key=lambda x: (x[0][0], x[0][1])):
        def col(name: str) -> List[Optional[float]]:
            return [_safe_float(rr.get(name)) for rr in rows]

        rec = {
            "dataset": ds,
            "theta": theta,
            "n_runs": len(rows),
            "acc_final_mean": mean(col("acc_final")),
            "acc_final_std": std(col("acc_final")),
            "mean_acc_mean": mean(col("mean_acc")),
            "mean_acc_std": std(col("mean_acc")),
            "acc_min_mean": mean(col("acc_min")),
            "acc_min_std": std(col("acc_min")),
            "MDR_win_mean": mean(col("MDR_win")),
            "MDR_win_std": std(col("MDR_win")),
            "MTFA_win_mean": mean(col("MTFA_win")),
            "MTFA_win_std": std(col("MTFA_win")),
            "MTR_win_mean": mean(col("MTR_win")),
            "MTR_win_std": std(col("MTR_win")),
            "MDR_tol_mean": mean(col("MDR_tol")),
            "MDR_tol_std": std(col("MDR_tol")),
            "MTFA_tol_mean": mean(col("MTFA_tol")),
            "MTFA_tol_std": std(col("MTFA_tol")),
            "MTR_tol_mean": mean(col("MTR_tol")),
            "MTR_tol_std": std(col("MTR_tol")),
        }
        f_agg_by_dataset.setdefault(ds, []).append(rec)

    f_tables: Dict[str, str] = {}
    f_reco: Dict[str, Tuple[Optional[float], str]] = {}
    for ds, items in f_agg_by_dataset.items():
        theta_star, reason = recommend_theta(items, args.theta_target_mdr, args.acc_tolerance)
        f_reco[ds] = (theta_star, reason)
        rows_md: List[List[str]] = []
        for it in sorted(items, key=lambda x: float(x["theta"])):
            rows_md.append(
                [
                    f"{float(it['theta']):.1f}",
                    fmt_mu_std(it["acc_final_mean"], it["acc_final_std"], 4),
                    fmt_mu_std(it["mean_acc_mean"], it["mean_acc_std"], 4),
                    fmt_mu_std(it["acc_min_mean"], it["acc_min_std"], 4),
                    fmt_mu_std(it["MDR_win_mean"], it["MDR_win_std"], 3),
                    fmt_mu_std(it["MTFA_win_mean"], it["MTFA_win_std"], 1),
                    fmt_mu_std(it["MTR_win_mean"], it["MTR_win_std"], 3),
                    fmt_mu_std(it["MDR_tol_mean"], it["MDR_tol_std"], 3),
                    fmt_mu_std(it["MTFA_tol_mean"], it["MTFA_tol_std"], 1),
                    fmt_mu_std(it["MTR_tol_mean"], it["MTR_tol_std"], 3),
                ]
            )
        f_tables[ds] = md_table(
            [
                "theta",
                "acc_final",
                "mean_acc",
                "acc_min",
                "MDR_win",
                "MTFA_win",
                "MTR_win",
                "MDR_tol",
                "MTFA_tol",
                "MTR_tol",
            ],
            rows_md,
        )

    # --- Track G：聚合（dataset+mode） ---
    g_group: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in g_rows:
        ds = r.get("dataset", "")
        mode = r.get("mode", "")
        if ds and mode:
            g_group.setdefault((ds, mode), []).append(r)

    g_rows_md: List[List[str]] = []
    for (ds, mode), rows in sorted(g_group.items(), key=lambda x: (x[0][0], x[0][1])):
        def col(name: str) -> List[Optional[float]]:
            return [_safe_float(rr.get(name)) for rr in rows]

        g_rows_md.append(
            [
                ds,
                mode,
                str(len(rows)),
                fmt_mu_std(mean(col("acc_final")), std(col("acc_final")), 4),
                fmt_mu_std(mean(col("mean_acc")), std(col("mean_acc")), 4),
                fmt_mu_std(mean(col("acc_min")), std(col("acc_min")), 4),
                fmt_mu_std(mean(col("MDR_win")), std(col("MDR_win")), 3),
                fmt_mu_std(mean(col("MTFA_win")), std(col("MTFA_win")), 1),
                fmt_mu_std(mean(col("MTR_win")), std(col("MTR_win")), 3),
                fmt_mu_std(mean(col("candidate_count")), std(col("candidate_count")), 1),
                fmt_mu_std(mean(col("confirmed_count")), std(col("confirmed_count")), 1),
                fmt_mu_std(mean(col("confirm_delay_mean")), std(col("confirm_delay_mean")), 2),
                fmt_mu_std(mean(col("confirm_delay_std")), std(col("confirm_delay_std")), 2),
            ]
        )

    g_table = md_table(
        [
            "dataset",
            "mode",
            "n_runs",
            "acc_final",
            "mean_acc",
            "acc_min",
            "MDR_win",
            "MTFA_win",
            "MTR_win",
            "candidate_count",
            "confirmed_count",
            "confirm_delay_mean",
            "confirm_delay_std",
        ],
        g_rows_md,
    )

    # --- Track H（可选） ---
    h_table = ""
    if h_rows:
        h_group: Dict[str, List[Dict[str, str]]] = {}
        for r in h_rows:
            h_group.setdefault(r.get("group", "unknown"), []).append(r)
        h_rows_md: List[List[str]] = []
        for group, rows in sorted(h_group.items(), key=lambda x: x[0]):
            def col(name: str) -> List[Optional[float]]:
                return [_safe_float(rr.get(name)) for rr in rows]

            h_rows_md.append(
                [
                    group,
                    str(len(rows)),
                    fmt_mu_std(mean(col("acc_final")), std(col("acc_final")), 4),
                    fmt_mu_std(mean(col("acc_min")), std(col("acc_min")), 4),
                    fmt_mu_std(mean(col("MDR_win")), std(col("MDR_win")), 3),
                    fmt_mu_std(mean(col("MTR_win")), std(col("MTR_win")), 3),
                ]
            )
        h_table = md_table(["group", "n_runs", "acc_final", "acc_min", "MDR_win", "MTR_win"], h_rows_md)

    # --- 写报告 ---
    fig_mdr_mtfa = fig_dir / "trackF_theta_mdr_mtfa.png"
    fig_acc = fig_dir / "trackF_theta_acc_final.png"
    lines: List[str] = []
    lines.append("# NEXT_STAGE Report (Track F/G/H)")
    lines.append("")
    lines.append(f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- TrackF CSV：`{trackf_path}`")
    lines.append(f"- TrackG CSV：`{trackg_path}`")
    if h_rows:
        lines.append(f"- TrackH CSV：`{trackh_path}`")
    lines.append(f"- 图：`{fig_mdr_mtfa}`、`{fig_acc}`")
    lines.append("")

    lines.append("## Track F：weighted 阈值扫描（trade-off）")
    for ds, table in f_tables.items():
        theta_star, reason = f_reco.get(ds, (None, ""))
        lines.append("")
        lines.append(f"### {ds}")
        lines.append(table)
        lines.append("")
        lines.append(f"- 推荐 theta：{('N/A' if theta_star is None else f'{theta_star:.2f}')}（{reason}）")
    lines.append("")

    lines.append("## Track G：OR vs weighted vs two_stage")
    lines.append(g_table)
    lines.append("")
    lines.append("- 结论：重点检查 two_stage 是否在 acc 不明显下降的同时改善 MDR/MTFA 或提升 MTR，并观察 candidate→confirm 延迟。")
    lines.append("")

    if h_rows:
        lines.append("## Track H：severity v2 + gating（INSECTS 最小验证）")
        lines.append(h_table)
        lines.append("")
        lines.append("- 结论：比较 v2 vs v2_gate 的 acc_final/acc_min（以及 MDR/MTR），判断 gating 是否减少负迁移。")
        lines.append("")
    else:
        lines.append("## Track H：TODO")
        lines.append("- 未运行（可选项）；如需补全，请先生成 `scripts/TRACKH_SEVERITY_GATING.csv` 再重新运行汇总脚本。")
        lines.append("")

    lines.append("## 下一步建议")
    lines.append("- 若 Track F 的推荐 theta 在两个数据集上一致：可作为 weighted 的默认阈值；否则建议按数据集或按误报成本动态选 θ。")
    lines.append("- 若 two_stage 在保持 acc 的情况下显著改善 MTFA/MTR：建议将其作为默认触发，并把 confirmed drift 作为 severity v2 的 gate。")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

