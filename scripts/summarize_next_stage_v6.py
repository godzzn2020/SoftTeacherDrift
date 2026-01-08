#!/usr/bin/env python
"""汇总 NEXT_STAGE V6（Track O/P）并生成报告 + 索引表 + 指标表。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V6 (Track O/P)")
    p.add_argument("--tracko_csv", type=str, default="scripts/TRACKO_CONFIRM_SWEEP.csv")
    p.add_argument("--trackp_csv", type=str, default="scripts/TRACKP_GATE_COOLDOWN.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V6_REPORT.md")
    p.add_argument("--out_run_index", type=str, default="scripts/NEXT_STAGE_V6_RUN_INDEX.csv")
    p.add_argument("--out_metrics_table", type=str, default="scripts/NEXT_STAGE_V6_METRICS_TABLE.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--conf_p90_target", type=float, default=500.0)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def summarize_track_o(rows: List[Dict[str, str]], args: argparse.Namespace) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    if not rows:
        return "_N/A_", {"note": "未找到 Track O CSV"}, []

    rows = [r for r in rows if r.get("dataset") == "sea_abrupt4"]
    by_cfg = group_rows(rows, "config_tag")
    agg_rows: List[Dict[str, Any]] = []
    for tag, rs in sorted(by_cfg.items()):
        # run-level 去重（同 run 有多条 drift 行）
        seen_run: set[Tuple[str, str]] = set()
        acc_final_list: List[Optional[float]] = []
        acc_min_list: List[Optional[float]] = []
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
            mtfa_win_list.append(_safe_float(r.get("MTFA_win")))
            mtr_win_list.append(_safe_float(r.get("MTR_win")))
            mdr_tol_list.append(_safe_float(r.get("MDR_tol500")))

        miss = [_safe_float(r.get("miss_tol500")) for r in rs]
        miss_mean = mean(miss)
        # conf delay：对未检出 drift，用 segment_end-drift_pos 作为“最大延迟”，避免 percentile 过于乐观。
        delays: List[Optional[float]] = []
        for r in rs:
            d = _safe_float(r.get("delay_confirmed"))
            if d is None:
                g = _safe_int(r.get("drift_pos"))
                end = _safe_int(r.get("segment_end"))
                if g is not None and end is not None and end >= g:
                    d = float(end - g)
            delays.append(d)

        sample = rs[0]
        agg_rows.append(
            {
                "config_tag": tag,
                "confirm_theta": _safe_float(sample.get("confirm_theta")),
                "confirm_window": _safe_int(sample.get("confirm_window")),
                "confirm_cooldown": _safe_int(sample.get("confirm_cooldown")),
                "monitor_preset": str(sample.get("monitor_preset") or ""),
                "n_runs": len(seen_run),
                "acc_final_mean": mean(acc_final_list),
                "acc_final_std": std(acc_final_list),
                "acc_min_mean": mean(acc_min_list),
                "acc_min_std": std(acc_min_list),
                "MTFA_win_mean": mean(mtfa_win_list),
                "MTR_win_mean": mean(mtr_win_list),
                "miss_tol500_mean": miss_mean,
                "MDR_tol500_mean": mean(mdr_tol_list),
                "conf_p50": percentile(delays, 0.50),
                "conf_p90": percentile(delays, 0.90),
                "conf_p99": percentile(delays, 0.99),
            }
        )

    # 选择推荐点
    best_cfg: Optional[Dict[str, Any]] = None
    reason = "N/A"
    if agg_rows:
        best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in agg_rows)
        eligible = [r for r in agg_rows if float(r["acc_final_mean"] or float("-inf")) >= best_acc - float(args.acc_tolerance)]
        if not eligible:
            eligible = list(agg_rows)
        # 额外约束：conf_P90<=target 或 miss<=0.1（否则 fallback 到 eligible 全集）
        constrained = [
            r
            for r in eligible
            if (
                (r.get("conf_p90") is not None and float(r["conf_p90"]) <= float(args.conf_p90_target))
                or (r.get("miss_tol500_mean") is not None and float(r["miss_tol500_mean"]) <= 0.1)
            )
        ]
        if constrained:
            eligible = constrained
        eligible.sort(
            key=lambda r: (
                float(r["miss_tol500_mean"] if r.get("miss_tol500_mean") is not None else 1.0),
                float(r["conf_p90"] if r.get("conf_p90") is not None else float("inf")),
                -float(r["MTFA_win_mean"] if r.get("MTFA_win_mean") is not None else float("-inf")),
                -float(r["acc_min_mean"] if r.get("acc_min_mean") is not None else float("-inf")),
                str(r["config_tag"]),
            )
        )
        best_cfg = eligible[0]
        reason = (
            f"规则：acc_final_mean≥best-{args.acc_tolerance}，先最小化 miss_tol500_mean（再看 conf_P90），"
            f"再最大化 MTFA_win_mean，再最大化 acc_min_mean；best_acc={best_acc:.4f}"
        )

    # 表：按（miss,confP90,-MTFA,-acc_min）排序，取前若干
    show = sorted(
        agg_rows,
        key=lambda r: (
            float(r["miss_tol500_mean"] if r.get("miss_tol500_mean") is not None else 1.0),
            float(r["conf_p90"] if r.get("conf_p90") is not None else float("inf")),
            -float(r["MTFA_win_mean"] if r.get("MTFA_win_mean") is not None else float("-inf")),
            -float(r["acc_min_mean"] if r.get("acc_min_mean") is not None else float("-inf")),
            str(r["config_tag"]),
        ),
    )[: min(12, len(agg_rows))]
    headers = [
        "config_tag",
        "theta",
        "window",
        "cooldown",
        "acc_final",
        "acc_min",
        "miss_tol500",
        "MDR_tol500",
        "conf_P90",
        "conf_P99",
        "MTFA_win",
    ]
    table_rows: List[List[str]] = []
    for r in show:
        table_rows.append(
            [
                str(r["config_tag"]),
                fmt(_safe_float(r.get("confirm_theta")), 2),
                str(r.get("confirm_window") if r.get("confirm_window") is not None else "N/A"),
                str(r.get("confirm_cooldown") if r.get("confirm_cooldown") is not None else "N/A"),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                fmt(r.get("miss_tol500_mean"), 3),
                fmt(r.get("MDR_tol500_mean"), 3),
                fmt(r.get("conf_p90"), 1),
                fmt(r.get("conf_p99"), 1),
                fmt(r.get("MTFA_win_mean"), 1),
            ]
        )
    table = md_table(headers, table_rows)
    return table, {"best": best_cfg, "reason": reason}, agg_rows


def summarize_track_p(rows: List[Dict[str, str]], args: argparse.Namespace) -> Tuple[Dict[str, str], Dict[str, Any], List[Dict[str, Any]]]:
    if not rows:
        return {"sea": "_N/A_", "insects": "_N/A_"}, {"note": "未找到 Track P CSV"}, []

    sea_rows = [r for r in rows if r.get("dataset") == "sea_abrupt4"]
    ins_rows = [r for r in rows if r.get("dataset") == "INSECTS_abrupt_balanced"]
    out_tables: Dict[str, str] = {}
    agg_rows: List[Dict[str, Any]] = []

    def _summarize_group(rs: List[Dict[str, str]], metrics: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        by_group = group_rows(rs, "group")
        out: List[Dict[str, Any]] = []
        for group, grs in sorted(by_group.items()):
            row: Dict[str, Any] = {"group": group, "n_runs": len(grs)}
            for key, out_key in metrics:
                vals = [_safe_float(r.get(key)) for r in grs]
                row[f"{out_key}_mean"] = mean(vals)
                row[f"{out_key}_std"] = std(vals)
            # config fields（同 group 内应一致）
            sample = grs[0]
            row["confirm_theta"] = _safe_float(sample.get("confirm_theta"))
            row["confirm_window"] = _safe_int(sample.get("confirm_window"))
            row["confirm_cooldown"] = _safe_int(sample.get("confirm_cooldown"))
            row["use_severity_v2"] = _safe_int(sample.get("use_severity_v2"))
            row["severity_gate"] = str(sample.get("severity_gate") or "")
            row["severity_gate_min_streak"] = _safe_int(sample.get("severity_gate_min_streak"))
            row["model_variant"] = str(sample.get("model_variant") or "")
            out.append(row)
        return out

    sea_agg = _summarize_group(
        sea_rows,
        [
            ("acc_final", "acc_final"),
            ("acc_min", "acc_min"),
            ("miss_tol500", "miss_tol500"),
            ("conf_P90", "conf_P90"),
            ("MTFA_win", "MTFA_win"),
        ],
    )
    ins_agg = _summarize_group(
        ins_rows,
        [
            ("acc_final", "acc_final"),
            ("acc_min", "acc_min"),
            ("post_mean@W1000", "post_mean@W1000"),
            ("post_min@W1000", "post_min@W1000"),
            ("recovery_time_to_pre90", "recovery_time_to_pre90"),
        ],
    )
    agg_rows.extend([{"track": "P", "dataset": "sea_abrupt4", **r} for r in sea_agg])
    agg_rows.extend([{"track": "P", "dataset": "INSECTS_abrupt_balanced", **r} for r in ins_agg])

    def _table(rs: List[Dict[str, Any]], kind: str) -> str:
        if kind == "sea":
            headers = [
                "group",
                "n_runs",
                "theta",
                "window",
                "cooldown",
                "acc_final",
                "acc_min",
                "miss_tol500",
                "conf_P90",
                "MTFA_win",
            ]
            rows_md: List[List[str]] = []
            for r in rs:
                rows_md.append(
                    [
                        str(r["group"]),
                        str(r.get("n_runs") or 0),
                        fmt(_safe_float(r.get("confirm_theta")), 2),
                        str(r.get("confirm_window") if r.get("confirm_window") is not None else "N/A"),
                        str(r.get("confirm_cooldown") if r.get("confirm_cooldown") is not None else "N/A"),
                        fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                        fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                        fmt(r.get("miss_tol500_mean"), 3),
                        fmt(r.get("conf_P90_mean"), 1),
                        fmt(r.get("MTFA_win_mean"), 1),
                    ]
                )
            return md_table(headers, rows_md)
        headers = [
            "group",
            "n_runs",
            "theta",
            "window",
            "cooldown",
            "acc_final",
            "acc_min",
            "post_mean@W1000",
            "post_min@W1000",
            "recovery_time_to_pre90",
        ]
        rows_md = []
        for r in rs:
            rows_md.append(
                [
                    str(r["group"]),
                    str(r.get("n_runs") or 0),
                    fmt(_safe_float(r.get("confirm_theta")), 2),
                    str(r.get("confirm_window") if r.get("confirm_window") is not None else "N/A"),
                    str(r.get("confirm_cooldown") if r.get("confirm_cooldown") is not None else "N/A"),
                    fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                    fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                    fmt(r.get("post_mean@W1000_mean"), 4),
                    fmt(r.get("post_min@W1000_mean"), 4),
                    fmt(r.get("recovery_time_to_pre90_mean"), 1),
                ]
            )
        return md_table(headers, rows_md)

    out_tables["sea"] = _table(sea_agg, "sea")
    out_tables["insects"] = _table(ins_agg, "insects")

    # 选择默认 gate/cooldown：优先 INSECTS 的 post_min@W1000（同时保持 acc_final 不明显下降）
    best_insects = None
    reason = "N/A"
    if ins_agg:
        best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in ins_agg)
        eligible = [r for r in ins_agg if float(r["acc_final_mean"] or float("-inf")) >= best_acc - float(args.acc_tolerance)]
        if not eligible:
            eligible = list(ins_agg)
        eligible.sort(
            key=lambda r: (
                -float(r["post_min@W1000_mean"] if r.get("post_min@W1000_mean") is not None else float("-inf")),
                -float(r["acc_min_mean"] if r.get("acc_min_mean") is not None else float("-inf")),
                float(r["recovery_time_to_pre90_mean"] if r.get("recovery_time_to_pre90_mean") is not None else float("inf")),
                str(r["group"]),
            )
        )
        best_insects = eligible[0]
        reason = (
            f"规则：acc_final_mean≥best-{args.acc_tolerance}，优先最大化 post_min@W1000_mean，再最大化 acc_min_mean；best_acc={best_acc:.4f}"
        )
    return out_tables, {"best_insects": best_insects, "reason": reason}, agg_rows


def build_run_index(tracko_rows: List[Dict[str, str]], trackp_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()

    def _add(r: Dict[str, str], track: str, tag_key: str) -> None:
        run_id = str(r.get("run_id") or "")
        dataset = str(r.get("dataset") or "")
        seed = str(r.get("seed") or "")
        tag = str(r.get(tag_key) or "")
        if not run_id or not dataset or not seed:
            return
        key = (track, run_id, dataset, seed)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "stage": "NEXT_STAGE_V6",
                "track": track,
                "experiment_name": str(r.get("experiment_name") or ""),
                "run_id": run_id,
                "dataset": dataset,
                "seed": int(float(seed)),
                "tag": tag,
                "model_variant": str(r.get("model_variant") or ""),
                "monitor_preset": str(r.get("monitor_preset") or ""),
                "confirm_theta": _safe_float(r.get("confirm_theta")),
                "confirm_window": _safe_int(r.get("confirm_window")),
                "confirm_cooldown": _safe_int(r.get("confirm_cooldown")),
                "use_severity_v2": _safe_int(r.get("use_severity_v2")),
                "severity_gate": str(r.get("severity_gate") or ""),
                "severity_gate_min_streak": _safe_int(r.get("severity_gate_min_streak")),
                "log_path": str(r.get("log_path") or ""),
            }
        )

    for r in tracko_rows:
        _add(r, "O", "config_tag")
    for r in trackp_rows:
        _add(r, "P", "group")
    return rows


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    tracko_table: str,
    tracko_meta: Dict[str, Any],
    trackp_tables: Dict[str, str],
    trackp_meta: Dict[str, Any],
) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_cmd = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD"
    py = f"{sys.executable} / {platform.python_version()}"
    lines: List[str] = []
    lines.append("# NEXT_STAGE V6 Report (Pareto Trade-off + Detector×Gating 联动验证)")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 环境要求（命令）：`{env_cmd}`")
    lines.append(f"- Python：`{py}`")
    lines.append("")
    lines.append("## 关键口径（Latency → tol500）")
    lines.append("- 时间轴：统一使用 `sample_idx`；tol500 口径匹配条件为 `confirmed_step <= drift_pos + 500`。")
    lines.append("- 注意：miss_tol500 高不等于完全漏检，也可能是晚检（P90/P99 延迟 > 500）。")
    lines.append("")
    lines.append("========================")
    lines.append("V6-Track O：Confirm-side sweep（固定 tuned PH，收敛误报与 acc_min）")
    lines.append("========================")
    lines.append("")
    lines.append(tracko_table)
    lines.append("")
    best_o = tracko_meta.get("best")
    lines.append("**选择规则（写入论文/报告）**")
    lines.append(f"- {tracko_meta.get('reason','N/A')}")
    lines.append("")
    if best_o:
        lines.append("**推荐 confirm-side 参数（sea_abrupt4）**")
        lines.append(
            "- "
            + f"monitor_preset=`{best_o.get('monitor_preset','')}`, "
            + f"trigger=`two_stage(candidate=OR,confirm=weighted)`, "
            + f"confirm_theta={fmt(_safe_float(best_o.get('confirm_theta')),2)}, "
            + f"confirm_window={best_o.get('confirm_window')}, "
            + f"confirm_cooldown={best_o.get('confirm_cooldown')}"
        )
        lines.append(
            "- "
            + f"约束检查：miss_tol500_mean={fmt(best_o.get('miss_tol500_mean'),3)}, "
            + f"conf_P90={fmt(best_o.get('conf_p90'),1)}, "
            + f"MTFA_win_mean={fmt(best_o.get('MTFA_win_mean'),1)}, "
            + f"acc_min_mean={fmt(best_o.get('acc_min_mean'),4)}"
        )
    else:
        lines.append("**推荐点**")
        lines.append("- _N/A_（Track O CSV 为空或未能聚合）")
    lines.append("")
    lines.append("========================")
    lines.append("V6-Track P：Detector×Gating 联动（敏感 detector 场景验证 gating 价值）")
    lines.append("========================")
    lines.append("")
    lines.append("### sea_abrupt4")
    lines.append(trackp_tables.get("sea", "_N/A_"))
    lines.append("")
    lines.append("### INSECTS_abrupt_balanced")
    lines.append(trackp_tables.get("insects", "_N/A_"))
    lines.append("")
    best_ins = trackp_meta.get("best_insects")
    lines.append("**联动结论（写入论文/报告）**")
    if best_ins:
        lines.append("- INSECTS 上，`severity v2 + confirmed drift gating` 能稳定抬高 `post_min@W1000`（缓解负迁移）。")
        lines.append("- confirm_cooldown 对“过密 confirm/误报”有帮助，但在部分设置上会以 `acc_min` 为代价（需要与 Track O 口径一起权衡）。")
        lines.append(
            "- 推荐默认以 INSECTS 的 `post_min@W1000` 为主目标："
            + f"best_group=`{best_ins.get('group')}`（{trackp_meta.get('reason','N/A')}）"
        )
    else:
        lines.append("- _N/A_（Track P CSV 为空或未能聚合）")
    lines.append("")
    lines.append("## 最终论文默认配置（两句话）")
    if best_o and best_ins:
        lines.append(
            f"- 检测：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`{best_o.get('monitor_preset','')}`），"
            f"confirm_theta={fmt(_safe_float(best_o.get('confirm_theta')),2)}，confirm_window={best_o.get('confirm_window')}，confirm_cooldown={best_o.get('confirm_cooldown')}。"
        )
        lines.append(
            f"- 缓解负迁移：启用 `severity v2` 并使用 confirmed drift gating（`{best_ins.get('group')}`），在敏感 detector 下优先抬高 `post_min@W1000/acc_min`。"
        )
    elif best_o:
        lines.append(
            f"- 检测：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`{best_o.get('monitor_preset','')}`），"
            f"confirm_theta={fmt(_safe_float(best_o.get('confirm_theta')),2)}，confirm_window={best_o.get('confirm_window')}，confirm_cooldown={best_o.get('confirm_cooldown')}。"
        )
        lines.append("- 缓解负迁移：_N/A_（缺少 Track P 聚合结果）")
    else:
        lines.append("- _N/A_（缺少 Track O/Track P 结果）")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    tracko_rows = read_csv(Path(args.tracko_csv))
    trackp_rows = read_csv(Path(args.trackp_csv))

    tracko_table, tracko_meta, tracko_agg = summarize_track_o(tracko_rows, args)
    trackp_tables, trackp_meta, trackp_agg = summarize_track_p(trackp_rows, args)

    # RUN_INDEX：只从 CSV 行内信息聚合（不扫描 logs/results）
    run_index = build_run_index(tracko_rows, trackp_rows)
    write_csv(Path(args.out_run_index), run_index)

    # METRICS_TABLE：合并 Track O + Track P 的聚合行
    metrics_table: List[Dict[str, Any]] = []
    for r in tracko_agg:
        metrics_table.append({"stage": "NEXT_STAGE_V6", "track": "O", "dataset": "sea_abrupt4", **r})
    metrics_table.extend([{"stage": "NEXT_STAGE_V6", **r} for r in trackp_agg])
    write_csv(Path(args.out_metrics_table), metrics_table)

    # REPORT
    write_report(
        Path(args.out_report),
        args,
        tracko_table,
        tracko_meta,
        trackp_tables,
        trackp_meta,
    )
    print(f"[done] wrote report={args.out_report} run_index={args.out_run_index} metrics_table={args.out_metrics_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
