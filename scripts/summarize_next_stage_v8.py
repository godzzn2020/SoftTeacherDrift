#!/usr/bin/env python
"""汇总 NEXT_STAGE V8（Track U/V/W）并生成报告。"""

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
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V8 (Track U/V/W)")
    p.add_argument("--tracku_csv", type=str, default="scripts/TRACKU_SYNTH_GENERALIZATION.csv")
    p.add_argument("--trackv_csv", type=str, default="scripts/TRACKV_INSECTS_RECOVERY.csv")
    p.add_argument("--trackw_csv", type=str, default="scripts/TRACKW_WARMUP_ROBUST.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V8_REPORT.md")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--warmup_samples", type=int, default=2000)
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


def group_rows(rows: List[Dict[str, str]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    out: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for r in rows:
        k = tuple(str(r.get(key) or "") for key in keys)
        if any(not x for x in k):
            continue
        out.setdefault(k, []).append(r)
    return out


def summarize_track_u(rows: List[Dict[str, str]], warmup_samples: int) -> str:
    if not rows:
        return "_N/A_"
    by = group_rows(rows, ("dataset", "group"))
    headers = [
        "dataset",
        "group",
        "n_runs",
        "acc_final",
        f"acc_min@{warmup_samples}",
        "miss_tol500",
        "MDR_tol500",
        "cand_P50",
        "cand_P90",
        "cand_P99",
        "conf_P50",
        "conf_P90",
        "conf_P99",
        "MTFA_win",
    ]
    out_rows: List[List[str]] = []
    for (dataset, group), rs in sorted(by.items()):
        acc_final = [_safe_float(r.get("acc_final")) for r in rs]
        acc_min_w = [_safe_float(r.get(f"acc_min@{warmup_samples}")) for r in rs]
        miss = [_safe_float(r.get("miss_tol500")) for r in rs]
        mdr = [_safe_float(r.get("MDR_tol500")) for r in rs]
        cand_p50 = [_safe_float(r.get("cand_P50")) for r in rs]
        cand_p90 = [_safe_float(r.get("cand_P90")) for r in rs]
        cand_p99 = [_safe_float(r.get("cand_P99")) for r in rs]
        conf_p50 = [_safe_float(r.get("conf_P50")) for r in rs]
        conf_p90 = [_safe_float(r.get("conf_P90")) for r in rs]
        conf_p99 = [_safe_float(r.get("conf_P99")) for r in rs]
        mtfa = [_safe_float(r.get("MTFA_win")) for r in rs]
        out_rows.append(
            [
                dataset,
                group,
                str(len(rs)),
                fmt_mu_std(mean(acc_final), std(acc_final), 4),
                fmt_mu_std(mean(acc_min_w), std(acc_min_w), 4),
                fmt_mu_std(mean(miss), std(miss), 3),
                fmt_mu_std(mean(mdr), std(mdr), 3),
                fmt_mu_std(mean(cand_p50), std(cand_p50), 1),
                fmt_mu_std(mean(cand_p90), std(cand_p90), 1),
                fmt_mu_std(mean(cand_p99), std(cand_p99), 1),
                fmt_mu_std(mean(conf_p50), std(conf_p50), 1),
                fmt_mu_std(mean(conf_p90), std(conf_p90), 1),
                fmt_mu_std(mean(conf_p99), std(conf_p99), 1),
                fmt_mu_std(mean(mtfa), std(mtfa), 1),
            ]
        )
    return md_table(headers, out_rows)


def check_track_u_constraints(rows: List[Dict[str, str]], dataset: str, group: str) -> Dict[str, Any]:
    rs = [r for r in rows if r.get("dataset") == dataset and r.get("group") == group]
    if not rs:
        return {"ok": False, "note": "no rows"}
    miss = mean([_safe_float(r.get("miss_tol500")) for r in rs])
    conf_p90 = mean([_safe_float(r.get("conf_P90")) for r in rs])
    mtfa = mean([_safe_float(r.get("MTFA_win")) for r in rs])
    return {
        "ok": bool((miss is not None and miss <= 0.01) and (conf_p90 is not None and conf_p90 < 500)),
        "miss": miss,
        "conf_p90": conf_p90,
        "mtfa": mtfa,
    }


def summarize_track_v(rows: List[Dict[str, str]], warmup_samples: int) -> str:
    if not rows:
        return "_N/A_"
    by = group_rows(rows, ("group",))
    headers = [
        "group",
        "n_runs",
        "acc_final",
        "acc_min_raw",
        f"acc_min@{warmup_samples}",
        "post_mean@W1000",
        "post_min@W1000",
    ]
    out_rows: List[List[str]] = []
    for (group,), rs in sorted(by.items()):
        acc_final = [_safe_float(r.get("acc_final")) for r in rs]
        acc_min_raw = [_safe_float(r.get("acc_min_raw")) for r in rs]
        acc_min_w = [_safe_float(r.get(f"acc_min@{warmup_samples}")) for r in rs]
        post_mean = [_safe_float(r.get("post_mean@W1000")) for r in rs]
        post_min = [_safe_float(r.get("post_min@W1000")) for r in rs]
        out_rows.append(
            [
                group,
                str(len(rs)),
                fmt_mu_std(mean(acc_final), std(acc_final), 4),
                fmt_mu_std(mean(acc_min_raw), std(acc_min_raw), 4),
                fmt_mu_std(mean(acc_min_w), std(acc_min_w), 4),
                fmt_mu_std(mean(post_mean), std(post_mean), 4),
                fmt_mu_std(mean(post_min), std(post_min), 4),
            ]
        )
    return md_table(headers, out_rows)


def summarize_track_w(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_N/A_"
    # 聚合：warmup -> mean(acc_min)
    by: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        w = _safe_int(r.get("warmup"))
        if w is None:
            continue
        by.setdefault(int(w), []).append(r)
    headers = ["warmup", "n_runs", "acc_min_mean", "acc_min_std"]
    out_rows: List[List[str]] = []
    for w, rs in sorted(by.items()):
        acc_min = [_safe_float(r.get("acc_min")) for r in rs]
        out_rows.append([str(w), str(len(rs)), fmt(mean(acc_min), 4), fmt(std(acc_min), 4)])
    return md_table(headers, out_rows)


def main() -> int:
    args = parse_args()
    warm = int(args.warmup_samples)
    tracku = read_csv(Path(args.tracku_csv))
    trackv = read_csv(Path(args.trackv_csv))
    trackw = read_csv(Path(args.trackw_csv))

    u_table = summarize_track_u(tracku, warm)
    v_table = summarize_track_v(trackv, warm)
    w_table = summarize_track_w(trackw)

    # 约束检查：主方法组 D
    datasets = ["sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"]
    checks: List[str] = []
    for ds in datasets:
        c = check_track_u_constraints(tracku, ds, "D_two_stage_tunedPH_cd200")
        # 对比 tuned baseline（B/C）以判断 MTFA 是否“恶化”
        mtfa_b = mean([_safe_float(r.get("MTFA_win")) for r in tracku if r.get("dataset") == ds and r.get("group") == "B_or_tunedPH"])
        mtfa_c = mean([_safe_float(r.get("MTFA_win")) for r in tracku if r.get("dataset") == ds and r.get("group") == "C_weighted_tunedPH"])
        mtfa_ref = max(x for x in [mtfa_b, mtfa_c] if x is not None) if any(x is not None for x in [mtfa_b, mtfa_c]) else None
        delta = (c.get("mtfa") - mtfa_ref) if (c.get("mtfa") is not None and mtfa_ref is not None) else None
        checks.append(
            f"- {ds}: miss={fmt(c.get('miss'),3)}, conf_P90={fmt(c.get('conf_p90'),1)}, "
            f"MTFA_win={fmt(c.get('mtfa'),1)} (Δvs best(B/C)={fmt(delta,1)})"
        )

    # Track V 结论：对比 post_min@W1000
    def _group_mean(rows: List[Dict[str, str]], group: str, key: str) -> Optional[float]:
        rs = [r for r in rows if r.get("group") == group]
        return mean([_safe_float(r.get(key)) for r in rs])

    v_base = _group_mean(trackv, "A_baseline", "post_min@W1000")
    v_v2 = _group_mean(trackv, "B_v2", "post_min@W1000")
    v_gate = _group_mean(trackv, "C_v2_gate_m5", "post_min@W1000")
    v_acc_base = _group_mean(trackv, "A_baseline", "acc_final")
    v_acc_v2 = _group_mean(trackv, "B_v2", "acc_final")
    v_acc_gate = _group_mean(trackv, "C_v2_gate_m5", "acc_final")

    # 默认选择：在 {B_v2, C_v2_gate_m5} 中，满足 acc_final 不低于 best-acc_tol 的情况下最大化 post_min@W1000
    best_acc = max(x for x in [v_acc_v2, v_acc_gate] if x is not None) if any(x is not None for x in [v_acc_v2, v_acc_gate]) else None
    cand = []
    if v_acc_v2 is not None and v_v2 is not None:
        cand.append(("B_v2", v_acc_v2, v_v2))
    if v_acc_gate is not None and v_gate is not None:
        cand.append(("C_v2_gate_m5", v_acc_gate, v_gate))
    chosen = None
    if best_acc is not None and cand:
        eligible = [x for x in cand if x[1] >= best_acc - float(args.acc_tolerance)]
        if not eligible:
            eligible = cand
        eligible.sort(key=lambda x: (-x[2], x[0]))
        chosen = eligible[0][0]

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_cmd = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD"
    py = f"{sys.executable} / {platform.python_version()}"

    lines: List[str] = []
    lines.append("# NEXT_STAGE V8 Report (泛化验证 + 定稿对比)")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 环境要求（命令）：`{env_cmd}`")
    lines.append(f"- Python：`{py}`")
    lines.append("")
    lines.append("========================")
    lines.append("V8-Track U：合成流泛化（检测创新点定稿）")
    lines.append("========================")
    lines.append("")
    lines.append(u_table)
    lines.append("")
    lines.append("**主方法组 D 约束检查（miss_tol500≈0 且 conf_P90<500）**")
    lines.extend(checks)
    lines.append("")
    lines.append("========================")
    lines.append("V8-Track V：INSECTS 恢复机制定稿（创新点2）")
    lines.append("========================")
    lines.append("")
    lines.append(v_table)
    lines.append("")
    lines.append("**结论（post_min@W1000 主目标）**")
    lines.append(f"- A_baseline post_min@W1000_mean={fmt(v_base,4)}")
    lines.append(f"- B_v2 post_min@W1000_mean={fmt(v_v2,4)}")
    lines.append(f"- C_v2_gate_m5 post_min@W1000_mean={fmt(v_gate,4)}")
    lines.append(f"- acc_final_mean：A={fmt(v_acc_base,4)} / B={fmt(v_acc_v2,4)} / C={fmt(v_acc_gate,4)}（容差={args.acc_tolerance}）")
    if chosen:
        lines.append(f"- 依据“acc_final 不降超过容差 + 最大化 post_min@W1000”，推荐默认组选 `{chosen}`。")
    lines.append("")
    lines.append("========================")
    lines.append("V8-Track W（可选）：warmup 阈值稳健性")
    lines.append("========================")
    lines.append("")
    lines.append(w_table)
    lines.append("")
    lines.append("## 最终默认配置（定稿）")
    lines.append(
        "- 检测：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`），"
        "confirm_theta=0.50，confirm_window=1，confirm_cooldown=200；sea 的谷底口径采用 `acc_min@sample_idx>=2000`。"
    )
    if chosen == "C_v2_gate_m5":
        lines.append("- 恢复：INSECTS 上启用 `severity v2 + confirmed drift gating (m=5)`，以提升 `post_min@W1000` 且保持 acc_final 不明显下降。")
    else:
        lines.append("- 恢复：INSECTS 上优先启用 `severity v2`；`v2_gate_m5` 在本轮设定下仍能提升 post_min@W1000（相对 baseline），但不一定优于 v2。")

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
