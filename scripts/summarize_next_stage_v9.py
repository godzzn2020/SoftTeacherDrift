#!/usr/bin/env python
"""汇总 NEXT_STAGE V9（Track X/Y）并生成报告（含 bootstrap CI）。"""

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V9 (Track X/Y)")
    p.add_argument("--trackx_csv", type=str, default="scripts/TRACKX_INSECTS_GATING_SWEEP.csv")
    p.add_argument("--tracky_csv", type=str, default="scripts/TRACKY_DETECTOR_GATING_INTERACTION.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V9_REPORT.md")
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--bootstrap_iters", type=int, default=5000)
    p.add_argument("--bootstrap_seed", type=int, default=7)
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


def group_rows(rows: List[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        k = str(r.get(key, "") or "")
        if not k:
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


def summarize_track_x(rows: List[Dict[str, str]], warmup_samples: int, acc_tol: float, args: argparse.Namespace) -> Dict[str, Any]:
    if not rows:
        return {"table": "_N/A_", "note": "未找到 Track X CSV"}

    rows = [r for r in rows if r.get("dataset") == "INSECTS_abrupt_balanced"]
    if not rows:
        return {"table": "_N/A_", "note": "Track X CSV 无 INSECTS_abrupt_balanced"}

    by_g = group_rows(rows, "group")
    agg: List[Dict[str, Any]] = []
    for g, rs in sorted(by_g.items()):
        acc_final = [_safe_float(x.get("acc_final")) for x in rs]
        acc_min = [_safe_float(x.get(f"acc_min@{warmup_samples}")) for x in rs]
        post_mean = [_safe_float(x.get("post_mean@W1000")) for x in rs]
        post_min = [_safe_float(x.get("post_min@W1000")) for x in rs]
        rate = [_safe_float(x.get("confirm_rate_per_10k")) for x in rs]
        agg.append(
            {
                "group": g,
                "n_runs": len(rs),
                "acc_final_mean": mean(acc_final),
                "acc_final_std": std(acc_final),
                "acc_min_mean": mean(acc_min),
                "acc_min_std": std(acc_min),
                "post_mean_mean": mean(post_mean),
                "post_mean_std": std(post_mean),
                "post_min_mean": mean(post_min),
                "post_min_std": std(post_min),
                "confirm_rate_mean": mean(rate),
                "confirm_rate_std": std(rate),
            }
        )

    # 选择赢家：在 v2 + v2_gate_m* 中按 post_min@W1000 最大化，且 acc_final 不低于 best-acc_tol
    candidates = [r for r in agg if r["group"] in {"B_v2", "v2_gate_m1", "v2_gate_m3", "v2_gate_m5"}]
    best = None
    reason = "N/A"
    if candidates:
        best_acc = max(float(r["acc_final_mean"] or float("-inf")) for r in candidates)
        eligible = [r for r in candidates if float(r["acc_final_mean"] or float("-inf")) >= best_acc - float(acc_tol)]
        if not eligible:
            eligible = candidates
        eligible.sort(
            key=lambda r: (
                -float(r["post_min_mean"] if r.get("post_min_mean") is not None else float("-inf")),
                -float(r["acc_min_mean"] if r.get("acc_min_mean") is not None else float("-inf")),
                str(r["group"]),
            )
        )
        best = eligible[0]
        reason = f"规则：acc_final_mean≥best-{acc_tol}，优先最大化 post_min@W1000_mean，再最大化 acc_min@{warmup_samples}_mean；best_acc={best_acc:.4f}"

    # Bootstrap CI：Δpost_min@W1000（winner - v2），按 seed 配对差
    ci_note = "N/A"
    ci_rows: List[List[str]] = []
    ci_tuple: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    if best and best.get("group") != "B_v2":
        winner_g = str(best["group"])
        v2_rows = [r for r in rows if r.get("group") == "B_v2"]
        win_rows = [r for r in rows if r.get("group") == winner_g]
        v2_by_seed = {int(float(r["seed"])): r for r in v2_rows if r.get("seed")}
        win_by_seed = {int(float(r["seed"])): r for r in win_rows if r.get("seed")}
        seeds = sorted(set(v2_by_seed) & set(win_by_seed))
        diffs: List[float] = []
        for s in seeds:
            a = _safe_float(win_by_seed[s].get("post_min@W1000"))
            b = _safe_float(v2_by_seed[s].get("post_min@W1000"))
            if a is None or b is None:
                continue
            diffs.append(float(a - b))
        mu, lo, hi = paired_bootstrap_ci(diffs, iters=int(args.bootstrap_iters), seed=int(args.bootstrap_seed))
        ci_note = f"Δpost_min@W1000({winner_g}-B_v2) = {fmt(mu,4)} (95% CI [{fmt(lo,4)}, {fmt(hi,4)}], n={len(diffs)})"
        ci_rows.append([winner_g, "B_v2", fmt(mu, 4), f"[{fmt(lo,4)},{fmt(hi,4)}]", str(len(diffs))])
        ci_tuple = (mu, lo, hi)

    headers = ["group", "n_runs", "acc_final", f"acc_min@{warmup_samples}", "post_mean@W1000", "post_min@W1000", "confirm_rate_per_10k"]
    table_rows: List[List[str]] = []
    for r in sorted(agg, key=lambda x: str(x["group"])):
        table_rows.append(
            [
                str(r["group"]),
                str(r["n_runs"]),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                fmt_mu_std(r.get("post_mean_mean"), r.get("post_mean_std"), 4),
                fmt_mu_std(r.get("post_min_mean"), r.get("post_min_std"), 4),
                fmt_mu_std(r.get("confirm_rate_mean"), r.get("confirm_rate_std"), 3),
            ]
        )
    table = md_table(headers, table_rows)
    ci_table = md_table(["winner", "ref", "delta_mean", "ci95", "n_seeds"], ci_rows) if ci_rows else "_N/A_"
    return {"table": table, "winner": best, "reason": reason, "ci_note": ci_note, "ci_table": ci_table, "ci_tuple": ci_tuple}


def summarize_track_y(rows: List[Dict[str, str]], warmup_samples: int, winner_m: Optional[int]) -> Dict[str, Any]:
    if not rows:
        return {"table": "_N/A_", "note": "未找到 Track Y CSV"}
    rows = [r for r in rows if r.get("dataset") == "INSECTS_abrupt_balanced"]
    if not rows:
        return {"table": "_N/A_", "note": "Track Y CSV 无 INSECTS_abrupt_balanced"}

    # detector_mode × recovery_mode 聚合
    by: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        d = str(r.get("detector_mode") or "")
        m = str(r.get("recovery_mode") or "")
        if not d or not m:
            continue
        by.setdefault((d, m), []).append(r)

    agg: List[Dict[str, Any]] = []
    for (det, rec), rs in sorted(by.items()):
        acc_final = [_safe_float(x.get("acc_final")) for x in rs]
        acc_min = [_safe_float(x.get(f"acc_min@{warmup_samples}")) for x in rs]
        post_mean = [_safe_float(x.get("post_mean@W1000")) for x in rs]
        post_min = [_safe_float(x.get("post_min@W1000")) for x in rs]
        rate = [_safe_float(x.get("confirm_rate_per_10k")) for x in rs]
        agg.append(
            {
                "detector_mode": det,
                "recovery_mode": rec,
                "n_runs": len(rs),
                "acc_final_mean": mean(acc_final),
                "acc_final_std": std(acc_final),
                "acc_min_mean": mean(acc_min),
                "acc_min_std": std(acc_min),
                "post_mean_mean": mean(post_mean),
                "post_mean_std": std(post_mean),
                "post_min_mean": mean(post_min),
                "post_min_std": std(post_min),
                "confirm_rate_mean": mean(rate),
                "confirm_rate_std": std(rate),
            }
        )

    headers = ["detector", "recovery", "n_runs", "acc_final", f"acc_min@{warmup_samples}", "post_min@W1000", "confirm_rate_per_10k"]
    rows_md: List[List[str]] = []
    for r in agg:
        rows_md.append(
            [
                str(r["detector_mode"]),
                str(r["recovery_mode"]),
                str(r["n_runs"]),
                fmt_mu_std(r.get("acc_final_mean"), r.get("acc_final_std"), 4),
                fmt_mu_std(r.get("acc_min_mean"), r.get("acc_min_std"), 4),
                fmt_mu_std(r.get("post_min_mean"), r.get("post_min_std"), 4),
                fmt_mu_std(r.get("confirm_rate_mean"), r.get("confirm_rate_std"), 3),
            ]
        )
    table = md_table(headers, rows_md)

    # 机制差异：比较 sensitive vs clean 下，gate 相对 v2 的增益是否更大
    # 记录 sensitive/clean 的 error.threshold 均值（用于报告）
    err_by_det: Dict[str, List[Optional[float]]] = {}
    for r in rows:
        det = str(r.get("detector_mode") or "")
        if not det:
            continue
        err_by_det.setdefault(det, []).append(_safe_float(r.get("error_threshold")))
    err_clean = mean(err_by_det.get("clean", []))
    err_sens = mean(err_by_det.get("sensitive", []))

    note = f"detector 设定：clean err_thr≈{fmt(err_clean,3)}，sensitive err_thr≈{fmt(err_sens,3)}"
    # 找到 gate 名
    gate_name = None
    for r in rows:
        rec = str(r.get("recovery_mode") or "")
        if rec.startswith("v2_gate_m"):
            gate_name = rec
            break
    if gate_name:
        def _get(det: str, rec: str) -> Optional[float]:
            for a in agg:
                if a["detector_mode"] == det and a["recovery_mode"] == rec:
                    return a.get("post_min_mean")
            return None

        clean_v2 = _get("clean", "v2")
        clean_gate = _get("clean", gate_name)
        sens_v2 = _get("sensitive", "v2")
        sens_gate = _get("sensitive", gate_name)
        if None not in (clean_v2, clean_gate, sens_v2, sens_gate):
            delta_clean = float(clean_gate - clean_v2)  # type: ignore[operator]
            delta_sens = float(sens_gate - sens_v2)  # type: ignore[operator]
            note = f"{note}；Δpost_min@W1000(gate-v2)：clean={fmt(delta_clean,4)}，sensitive={fmt(delta_sens,4)}（若 sensitive 更大则支持联动机制）"
    if winner_m is not None and gate_name is not None:
        note = f"{note}；TrackX winner_m={winner_m}"

    return {"table": table, "note": note}


def main() -> int:
    args = parse_args()
    warm = int(args.warmup_samples)
    trackx = read_csv(Path(args.trackx_csv))
    tracky = read_csv(Path(args.tracky_csv))

    x = summarize_track_x(trackx, warm, float(args.acc_tolerance), args)
    winner = x.get("winner")
    winner_m = None
    if winner:
        g = str(winner.get("group") or "")
        if g.startswith("v2_gate_m"):
            try:
                winner_m = int(g.split("m")[-1].split("_")[0])
            except Exception:
                winner_m = None
    y = summarize_track_y(tracky, warm, winner_m)
    ci_tuple = x.get("ci_tuple")
    recommend_recovery = "B_v2"
    recommend_reason = "默认：v2（若 gating 的提升不显著则不强推复杂机制）"
    if winner and str(winner.get("group") or "").startswith("v2_gate_m"):
        # 仅当 bootstrap CI 的下界 > 0 才认为 gating 对 v2 有稳健提升
        if isinstance(ci_tuple, tuple) and len(ci_tuple) == 3:
            _, lo, _hi = ci_tuple
            if lo is not None and float(lo) > 0:
                recommend_recovery = str(winner.get("group") or "B_v2")
                recommend_reason = "gating 相对 v2 的 Δpost_min@W1000 bootstrap 95% CI 下界 > 0"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_cmd = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD"
    py = f"{sys.executable} / {platform.python_version()}"

    lines: List[str] = []
    lines.append("# NEXT_STAGE V9 Report (gating 收敛 + detector×gating 联动验证)")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 环境要求（命令）：`{env_cmd}`")
    lines.append(f"- Python：`{py}`")
    lines.append("")
    lines.append("========================")
    lines.append("V9-Track X：INSECTS gating 强度 sweep（在 V8 定稿检测下）")
    lines.append("========================")
    lines.append("")
    lines.append(str(x.get("table") or "_N/A_"))
    lines.append("")
    lines.append("**结论与赢家**")
    lines.append(f"- {x.get('reason','N/A')}")
    if winner:
        lines.append(
            "- "
            + f"winner=`{winner.get('group')}`：post_min@W1000={fmt(winner.get('post_min_mean'),4)}，"
            + f"acc_final={fmt(winner.get('acc_final_mean'),4)}，confirm_rate/10k={fmt(winner.get('confirm_rate_mean'),3)}"
        )
    else:
        lines.append("- _N/A_（无法从 TrackX 选出赢家）")
    lines.append("")
    lines.append("**Δpost_min@W1000 的 bootstrap 95% CI（逐 seed 配对差）**")
    lines.append(str(x.get("ci_note") or "N/A"))
    lines.append(str(x.get("ci_table") or "_N/A_"))
    if isinstance(ci_tuple, tuple) and len(ci_tuple) == 3 and winner and str(winner.get("group") or "").startswith("v2_gate_m"):
        _mu, lo, _hi = ci_tuple
        if lo is not None and float(lo) <= 0:
            lines.append(f"- 结论：本轮 `Δpost_min@W1000(gate-v2)` 的 95% CI 跨 0，优势不显著；默认配置不强制启用 gating。")
    lines.append("")
    lines.append("========================")
    lines.append("V9-Track Y：detector 敏感度 × gating 联动（机制验证）")
    lines.append("========================")
    lines.append("")
    lines.append(str(y.get("table") or "_N/A_"))
    lines.append("")
    lines.append("**联动解释（用 confirm_rate 解释机制）**")
    lines.append(f"- {y.get('note','N/A')}")
    lines.append("")
    lines.append("## 回答三问（必须）")
    lines.append("1) 在 V8 定稿检测条件下，gating 是否仍优于 v2？最优 m 是多少？")
    if winner:
        lines.append(f"- TrackX 赢家：`{winner.get('group')}`；是否优于 v2 以 `post_min@W1000` 及其 bootstrap CI 为准。")
    else:
        lines.append("- _N/A_（缺少 TrackX 赢家）")
    lines.append("2) gating 的价值是否在 sensitive detector 下更明显？")
    lines.append("- TrackY 对比 clean vs sensitive 下的 `Δpost_min(gate-v2)` 以及 confirm_rate 的上升。")
    lines.append("3) 下一版最终默认配置（检测侧固定 + 恢复侧）")
    lines.append("- 检测侧（固定）：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`），confirm_theta=0.50，confirm_window=1，confirm_cooldown=200。")
    if recommend_recovery.startswith("v2_gate_m"):
        lines.append(f"- 恢复侧：`severity v2 + confirmed drift gating ({recommend_recovery})`（{recommend_reason}）。")
    else:
        lines.append(f"- 恢复侧：`severity v2`（{recommend_reason}）。")

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
