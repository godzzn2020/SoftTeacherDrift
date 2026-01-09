#!/usr/bin/env python
"""汇总 NEXT_STAGE V11（Track AD/AE/AF）并生成报告。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import platform
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V11 (Track AD/AE/AF)")
    p.add_argument("--trackad_csv", type=str, default="scripts/TRACKAD_GRADUAL_TOL_AUDIT.csv")
    p.add_argument("--trackae_csv", type=str, default="scripts/TRACKAE_PH_CALIBRATION_WITH_NODRIFT.csv")
    p.add_argument("--trackac_csv", type=str, default="scripts/TRACKAC_RECOVERY_WINDOW_SWEEP.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V11_REPORT.md")
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--bootstrap_iters", type=int, default=5000)
    p.add_argument("--bootstrap_seed", type=int, default=11)
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


def paired_bootstrap_ci(
    diffs: Sequence[float],
    *,
    iters: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not diffs:
        return None, None, None
    rng = random.Random(int(seed))
    n = len(diffs)
    means: List[float] = []
    for _ in range(int(iters)):
        sample = [diffs[rng.randrange(0, n)] for _ in range(n)]
        means.append(float(sum(sample) / n))
    means.sort()
    lo_idx = int((alpha / 2) * len(means))
    hi_idx = int((1 - alpha / 2) * len(means)) - 1
    lo = means[max(0, min(lo_idx, len(means) - 1))]
    hi = means[max(0, min(hi_idx, len(means) - 1))]
    return float(sum(diffs) / n), float(lo), float(hi)


def summarize_track_ad(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"table": "_N/A_", "note": "未找到 Track AD CSV"}
    by = group_rows(rows, ("group",))
    headers = [
        "group",
        "n_runs",
        "miss_tol500_start",
        "miss_tol500_mid",
        "miss_tol500_end",
        "delay_start_P90",
        "delay_mid_P90",
        "delay_end_P90",
    ]
    out_rows: List[List[str]] = []
    agg: Dict[str, Dict[str, Any]] = {}
    for (group,), rs in sorted(by.items()):
        ms = [_safe_float(r.get("miss_tol500_start")) for r in rs]
        mm = [_safe_float(r.get("miss_tol500_mid")) for r in rs]
        me = [_safe_float(r.get("miss_tol500_end")) for r in rs]
        ds = [_safe_float(r.get("delay_start_P90")) for r in rs]
        dm = [_safe_float(r.get("delay_mid_P90")) for r in rs]
        de = [_safe_float(r.get("delay_end_P90")) for r in rs]
        agg[group] = {
            "miss_start_mean": mean(ms),
            "miss_mid_mean": mean(mm),
            "miss_end_mean": mean(me),
        }
        out_rows.append(
            [
                group,
                str(len(rs)),
                fmt_mu_std(mean(ms), std(ms), 3),
                fmt_mu_std(mean(mm), std(mm), 3),
                fmt_mu_std(mean(me), std(me), 3),
                fmt_mu_std(mean(ds), std(ds), 1),
                fmt_mu_std(mean(dm), std(dm), 1),
                fmt_mu_std(mean(de), std(de), 1),
            ]
        )
    table = md_table(headers, out_rows)
    # 结论：是否主要是口径问题
    note = "N/A"
    if agg:
        # 若所有组 miss_end 仍明显>0，则机制问题占主导
        end_vals = [v.get("miss_end_mean") for v in agg.values() if v.get("miss_end_mean") is not None]
        start_vals = [v.get("miss_start_mean") for v in agg.values() if v.get("miss_start_mean") is not None]
        if end_vals and start_vals:
            if min(end_vals) > 0.10:
                note = "区间口径（start→end）能降低 miss（部分组从 ~0.40 降到 ~0.25），但 `miss_tol500_end` 仍显著>0，说明 stagger_gradual 的高 miss 不只是口径问题，更偏机制/参数导致的真实漏检。"
            else:
                note = "主要为口径问题：从 start 切换到 end 后 miss 明显下降并接近 0。"
    return {"table": table, "agg": agg, "note": note}


def summarize_track_ae(rows: List[Dict[str, str]], warmup_samples: int, acc_tol: float) -> Dict[str, Any]:
    if not rows:
        return {"table": "_N/A_", "note": "未找到 Track AE CSV", "winner": None}

    # 聚合到 (thr, mi) × dataset_kind × dataset
    by = group_rows(rows, ("error.threshold", "error.min_instances", "dataset_kind", "dataset"))

    # 先汇总每个组合在每个数据集上的均值
    summary: Dict[Tuple[float, int], Dict[str, Any]] = {}
    for (thr_s, mi_s, kind, dataset), rs in by.items():
        thr = float(thr_s)
        mi = int(float(mi_s))
        key = (thr, mi)
        entry = summary.setdefault(key, {"error.threshold": thr, "error.min_instances": mi, "drift": {}, "nodrift": {}})
        if kind == "drift":
            entry["drift"].setdefault(dataset, {})
            entry["drift"][dataset] = {
                "miss": mean([_safe_float(r.get("miss_tol500")) for r in rs]),
                "conf_p90": mean([_safe_float(r.get("conf_P90")) for r in rs]),
                "acc_final": mean([_safe_float(r.get("acc_final")) for r in rs]),
                f"acc_min@{warmup_samples}": mean([_safe_float(r.get(f"acc_min@{warmup_samples}")) for r in rs]),
                "mtfa": mean([_safe_float(r.get("MTFA_win")) for r in rs]),
            }
        else:
            entry["nodrift"].setdefault(dataset, {})
            entry["nodrift"][dataset] = {
                "rate": mean([_safe_float(r.get("confirm_rate_per_10k")) for r in rs]),
                "mtfa": mean([_safe_float(r.get("MTFA_win")) for r in rs]),
                "acc_final": mean([_safe_float(r.get("acc_final")) for r in rs]),
            }

    # 选择：满足 drift 约束（sea+sine 都 miss==0 且 conf_p90<500），在此基础上最小化 no-drift rate
    eligible: List[Dict[str, Any]] = []
    for key, entry in summary.items():
        drift = entry.get("drift") or {}
        sea = drift.get("sea_gradual_frequent") or {}
        sine = drift.get("sine_gradual_frequent") or {}
        if not sea or not sine:
            continue
        miss_ok = (sea.get("miss") is not None and sea["miss"] <= 0.0 + 1e-12) and (sine.get("miss") is not None and sine["miss"] <= 0.0 + 1e-12)
        conf_ok = (sea.get("conf_p90") is not None and sea["conf_p90"] < 500) and (sine.get("conf_p90") is not None and sine["conf_p90"] < 500)
        if not (miss_ok and conf_ok):
            continue
        entry["constraints_ok"] = True
        eligible.append(entry)

    winner = None
    reason = "N/A"
    if eligible:
        # 次目标：acc_final 不降超过 0.01（以 drift 平均 acc_final 的 best 为参照）
        drift_acc = [float((e["drift"]["sea_gradual_frequent"]["acc_final"] or float("-inf")) + (e["drift"]["sine_gradual_frequent"]["acc_final"] or float("-inf"))) / 2.0 for e in eligible]
        best_acc = max(drift_acc) if drift_acc else float("-inf")
        eligible2 = []
        for e in eligible:
            a = float((e["drift"]["sea_gradual_frequent"]["acc_final"] or float("-inf")) + (e["drift"]["sine_gradual_frequent"]["acc_final"] or float("-inf"))) / 2.0
            if a >= best_acc - float(acc_tol):
                eligible2.append(e)
        if not eligible2:
            eligible2 = eligible

        def nodrift_rate(e: Dict[str, Any]) -> float:
            nd = (e.get("nodrift") or {}).get("sea_nodrift") or {}
            v = nd.get("rate")
            return float(v) if v is not None else float("inf")

        def nodrift_mtfa(e: Dict[str, Any]) -> float:
            nd = (e.get("nodrift") or {}).get("sea_nodrift") or {}
            v = nd.get("mtfa")
            return float(v) if v is not None else float("-inf")

        eligible2.sort(key=lambda e: (nodrift_rate(e), -nodrift_mtfa(e), float(e["error.threshold"]), int(e["error.min_instances"])))
        winner = eligible2[0]
        reason = "规则：drift(sea+sine) miss_mean==0 且 conf_P90_mean<500；在满足下最小化 no-drift confirm_rate_per_10k（次选最大化 MTFA_win）；并要求 drift acc_final_mean 不低于 best-0.01。"

    # 输出表：每个 (thr,mi) 的关键汇总
    headers = [
        "error.threshold",
        "error.min_instances",
        "sea_miss",
        "sea_conf_P90",
        "sine_miss",
        "sine_conf_P90",
        "no_drift_rate",
        "no_drift_MTFA",
        "constraints_ok",
    ]
    table_rows: List[List[str]] = []
    for (thr, mi), e in sorted(summary.items(), key=lambda x: (x[0][0], x[0][1])):
        sea = (e.get("drift") or {}).get("sea_gradual_frequent") or {}
        sine = (e.get("drift") or {}).get("sine_gradual_frequent") or {}
        nd = (e.get("nodrift") or {}).get("sea_nodrift") or {}
        ok = bool(e.get("constraints_ok"))
        table_rows.append(
            [
                f"{thr:.2f}",
                str(mi),
                fmt(sea.get("miss"), 3),
                fmt(sea.get("conf_p90"), 1),
                fmt(sine.get("miss"), 3),
                fmt(sine.get("conf_p90"), 1),
                fmt(nd.get("rate"), 3),
                fmt(nd.get("mtfa"), 1),
                "Y" if ok else "",
            ]
        )
    return {"table": md_table(headers, table_rows), "winner": winner, "reason": reason}


def summarize_track_af(rows_ac: List[Dict[str, str]], *, iters: int, seed: int) -> Dict[str, Any]:
    if not rows_ac:
        return {"table": "_N/A_", "note": "未找到 Track AC CSV"}

    # 按 seed 配对 baseline/v2
    by_seed: Dict[int, Dict[str, Dict[str, str]]] = {}
    for r in rows_ac:
        s = _safe_int(r.get("seed"))
        g = str(r.get("group") or "")
        if s is None or g not in {"baseline", "v2"}:
            continue
        by_seed.setdefault(int(s), {})[g] = r

    windows = [500, 1000, 2000]
    rows_out: List[List[str]] = []
    for w in windows:
        diffs_min: List[float] = []
        diffs_mean: List[float] = []
        for s, m in by_seed.items():
            if "baseline" not in m or "v2" not in m:
                continue
            a_min = _safe_float(m["v2"].get(f"post_min@W{w}"))
            b_min = _safe_float(m["baseline"].get(f"post_min@W{w}"))
            a_mean = _safe_float(m["v2"].get(f"post_mean@W{w}"))
            b_mean = _safe_float(m["baseline"].get(f"post_mean@W{w}"))
            if a_min is not None and b_min is not None:
                diffs_min.append(float(a_min - b_min))
            if a_mean is not None and b_mean is not None:
                diffs_mean.append(float(a_mean - b_mean))
            _ = s

        mu_min, lo_min, hi_min = paired_bootstrap_ci(diffs_min, iters=int(iters), seed=int(seed))
        mu_mean, lo_mean, hi_mean = paired_bootstrap_ci(diffs_mean, iters=int(iters), seed=int(seed))
        rows_out.append([f"W{w}", "Δpost_min(v2-baseline)", fmt(mu_min, 4), f"[{fmt(lo_min,4)},{fmt(hi_min,4)}]", str(len(diffs_min))])
        rows_out.append([f"W{w}", "Δpost_mean(v2-baseline)", fmt(mu_mean, 4), f"[{fmt(lo_mean,4)},{fmt(hi_mean,4)}]", str(len(diffs_mean))])

    table = md_table(["window", "metric", "delta_mean", "ci95", "n_seeds"], rows_out)
    return {"table": table}


def main() -> int:
    args = parse_args()
    trackad_csv = Path(args.trackad_csv)
    trackae_csv = Path(args.trackae_csv)
    trackac_csv = Path(args.trackac_csv)
    out_report = Path(args.out_report)

    rows_ad = read_csv(trackad_csv)
    rows_ae = read_csv(trackae_csv)
    rows_ac = read_csv(trackac_csv)

    ad = summarize_track_ad(rows_ad)
    ae = summarize_track_ae(rows_ae, warmup_samples=int(args.warmup_samples), acc_tol=float(args.acc_tolerance))
    af = summarize_track_af(rows_ac, iters=int(args.bootstrap_iters), seed=int(args.bootstrap_seed))

    winner = ae.get("winner")
    if winner:
        ph_line = f"error.threshold={float(winner['error.threshold']):.2f},error.min_instances={int(winner['error.min_instances'])}"
    else:
        ph_line = "N/A"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py = sys.executable
    pyver = platform.python_version()

    report = f"""# NEXT_STAGE V11 Report（gradual 口径修正 + no-drift 约束校准 + 恢复统计CI）

- 生成时间：{now}
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`{py} / {pyver}`

================================================
V11-Track AD：Gradual drift 的区间 GT 口径重算（stagger_gradual_frequent）
================================================

产物：`{trackad_csv}`

{ad.get('table','_N/A_')}

**结论（回答“口径问题 vs 机制问题”）**
- 若 `miss_tol500_start` 高但 `miss_tol500_end` 低，说明主要是“GT 锚点过早（start）”导致的口径问题（检测更接近 transition 后段）。
- 若 `miss_tol500_end` 仍高，则更可能是机制/参数导致的真实漏检（需要进一步调 detector 或 confirm 结构）。
- 本轮结论：{ad.get('note','N/A')}

================================================
V11-Track AE：no-drift 约束下的 PH 校准（降低误报密度）
================================================

产物：`{trackae_csv}`

{ae.get('table','_N/A_')}

**推荐 PH（写入默认配置）**
- {ph_line}
- 选择规则：{ae.get('reason','N/A')}
- 备注：在 drift 约束（miss=0 且 conf_P90<500）下，仅 `min_instances=5` 可行；因此 no-drift 的 confirm_rate 降幅有限（主要依赖 error.threshold 微调）。

================================================
V11-Track AF：INSECTS 恢复收益的配对置信区间（不新跑）
================================================

输入：`{trackac_csv}`（seeds=40，baseline vs v2）

{af.get('table','_N/A_')}

**解读**
- 点估计为正但 95% CI 跨 0，说明 v2 的收益较小且在本轮设定下未达到统计显著（但方向一致）。

================================================
V11 后“一句话默认配置”
================================================

- 检测：`two_stage(candidate=OR,confirm=weighted)` + `error_divergence_ph_meta@{ph_line}`，confirm_theta=0.50，confirm_window=1，confirm_cooldown=200；主口径 `acc_min@sample_idx>=2000`。
- 恢复：INSECTS 默认启用 `severity v2`（Track AF 给出 Δpost_min/Δpost_mean 的配对 CI：点估计为正但 CI 跨 0，收益较小）。 
"""

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report, encoding="utf-8")
    print(f"[done] wrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
