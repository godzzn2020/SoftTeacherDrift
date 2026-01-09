#!/usr/bin/env python
"""汇总 NEXT_STAGE V10（Track AA/AB/AC）并生成报告。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V10 (Track AA/AB/AC)")
    p.add_argument("--trackaa_csv", type=str, default="scripts/TRACKAA_GENERALIZATION_NONABRUPT.csv")
    p.add_argument("--trackab_csv", type=str, default="scripts/TRACKAB_NODRIFT_SANITY.csv")
    p.add_argument("--trackx_csv", type=str, default="scripts/TRACKX_INSECTS_GATING_SWEEP.csv")
    p.add_argument("--trackac_csv", type=str, default="scripts/TRACKAC_RECOVERY_WINDOW_SWEEP.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V10_REPORT.md")
    p.add_argument("--warmup_samples", type=int, default=2000)
    p.add_argument("--insects_meta", type=str, default="datasets/real/INSECTS_abrupt_balanced.json")
    p.add_argument("--recovery_windows", type=str, default="500,1000,2000")
    p.add_argument("--recovery_pre_window", type=int, default=1000)
    p.add_argument("--recovery_roll_points", type=int, default=5)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
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


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


def load_insects_positions(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    positions = obj.get("positions") or []
    return [int(x) for x in positions]


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def compute_recovery_metrics(
    acc_series: Sequence[Tuple[int, float]],
    drifts: Sequence[int],
    *,
    W: int,
    pre_window: int,
    roll_points: int,
) -> Dict[str, Optional[float]]:
    if not acc_series or not drifts:
        return {"post_mean_acc": None, "post_min_acc": None}
    acc_map = list(acc_series)
    post_means: List[Optional[float]] = []
    post_mins: List[Optional[float]] = []
    for g in drifts:
        post = [a for (x, a) in acc_map if g <= x <= g + W]
        if post:
            post_means.append(float(statistics.mean(post)))
            post_mins.append(float(min(post)))
        else:
            post_means.append(None)
            post_mins.append(None)
        _ = pre_window, roll_points
    post_mean_mu, _ = mean_std(post_means)
    post_min_mu, _ = mean_std(post_mins)
    return {"post_mean_acc": post_mean_mu, "post_min_acc": post_min_mu}


def build_trackac_from_trackx(
    trackx_rows: List[Dict[str, str]],
    *,
    insects_meta: Path,
    out_csv: Path,
    warmup_samples: int,
    windows: Sequence[int],
    pre_window: int,
    roll_points: int,
) -> List[Dict[str, Any]]:
    insects_positions = load_insects_positions(insects_meta)
    rows = [r for r in trackx_rows if r.get("dataset") == "INSECTS_abrupt_balanced" and r.get("group") in {"A_baseline", "B_v2"}]
    rows.sort(key=lambda r: (str(r.get("group") or ""), _safe_int(r.get("seed")) or 0))

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        log_path_s = str(r.get("log_path") or "").strip()
        if not log_path_s:
            continue
        log_path = Path(log_path_s)
        summ = read_run_summary(log_path)
        series_raw = summ.get("acc_series") or []
        series: List[Tuple[int, float]] = []
        for item in series_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            series.append((int(item[0]), float(item[1])))
        src_group = str(r.get("group") or "")
        group = "baseline" if src_group == "A_baseline" else ("v2" if src_group == "B_v2" else src_group)
        rec: Dict[str, Any] = {
            "track": "AC",
            "dataset": "INSECTS_abrupt_balanced",
            "unit": "sample_idx",
            "seed": _safe_int(r.get("seed")),
            "group": group,
            "src_group": src_group,
            "run_id": str(r.get("run_id") or ""),
            "log_path": str(log_path),
            "acc_final": summ.get("acc_final"),
            "acc_min_raw": summ.get("acc_min"),
            f"acc_min@{int(warmup_samples)}": acc_min_after_warmup(summ, int(warmup_samples)),
            "confirmed_count": _safe_int(summ.get("confirmed_count_total")),
            "confirm_rate_per_10k": _safe_float(r.get("confirm_rate_per_10k")) or _safe_float(summ.get("confirm_rate_per_10k")),
        }
        for w in windows:
            m = compute_recovery_metrics(series, insects_positions, W=int(w), pre_window=int(pre_window), roll_points=int(roll_points))
            rec[f"post_mean@W{int(w)}"] = m.get("post_mean_acc")
            rec[f"post_min@W{int(w)}"] = m.get("post_min_acc")
        out_rows.append(rec)

    write_csv(out_csv, out_rows)
    return out_rows


def summarize_track_aa(rows: List[Dict[str, str]], warmup_samples: int) -> Tuple[str, Dict[str, Any]]:
    if not rows:
        return "_N/A_", {}
    by = group_rows(rows, ("dataset", "group"))
    headers = [
        "dataset",
        "group",
        "n_runs",
        "acc_final",
        f"acc_min@{warmup_samples}",
        "miss_tol500",
        "MDR_tol500",
        "cand_P90",
        "conf_P90",
        "MTFA_win",
    ]
    out_rows: List[List[str]] = []
    for (dataset, group), rs in sorted(by.items()):
        acc_final = [_safe_float(r.get("acc_final")) for r in rs]
        acc_min_w = [_safe_float(r.get(f"acc_min@{warmup_samples}")) for r in rs]
        miss = [_safe_float(r.get("miss_tol500")) for r in rs]
        mdr = [_safe_float(r.get("MDR_tol500")) for r in rs]
        cand_p90 = [_safe_float(r.get("cand_P90")) for r in rs]
        conf_p90 = [_safe_float(r.get("conf_P90")) for r in rs]
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
                fmt_mu_std(mean(cand_p90), std(cand_p90), 1),
                fmt_mu_std(mean(conf_p90), std(conf_p90), 1),
                fmt_mu_std(mean(mtfa), std(mtfa), 1),
            ]
        )
    table = md_table(headers, out_rows)

    constraints: Dict[str, Any] = {}
    for dataset in sorted({r.get("dataset") for r in rows if r.get("dataset")}):
        rs = [r for r in rows if r.get("dataset") == dataset and r.get("group") == "D_two_stage_tunedPH_cd200"]
        if not rs:
            continue
        miss = mean([_safe_float(r.get("miss_tol500")) for r in rs])
        conf_p90 = mean([_safe_float(r.get("conf_P90")) for r in rs])
        ok = bool((miss is not None and miss <= 0.01) and (conf_p90 is not None and conf_p90 < 500))
        constraints[str(dataset)] = {"miss_mean": miss, "conf_p90_mean": conf_p90, "ok": ok}
    return table, constraints


def summarize_track_ab(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_N/A_"
    by = group_rows(rows, ("dataset", "group"))
    headers = ["dataset", "group", "n_runs", "confirm_rate_per_10k", "MTFA_win", "acc_final", "mean_acc"]
    out_rows: List[List[str]] = []
    for (dataset, group), rs in sorted(by.items()):
        rate = [_safe_float(r.get("confirm_rate_per_10k")) for r in rs]
        mtfa = [_safe_float(r.get("MTFA_win")) for r in rs]
        acc_final = [_safe_float(r.get("acc_final")) for r in rs]
        mean_acc_v = [_safe_float(r.get("mean_acc")) for r in rs]
        out_rows.append(
            [
                dataset,
                group,
                str(len(rs)),
                fmt_mu_std(mean(rate), std(rate), 3),
                fmt_mu_std(mean(mtfa), std(mtfa), 1),
                fmt_mu_std(mean(acc_final), std(acc_final), 4),
                fmt_mu_std(mean(mean_acc_v), std(mean_acc_v), 4),
            ]
        )
    return md_table(headers, out_rows)


def summarize_track_ac(rows: List[Dict[str, Any]], windows: Sequence[int], warmup_samples: int) -> str:
    if not rows:
        return "_N/A_"
    # 先按 group 汇总
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        g = str(r.get("group") or "")
        if not g:
            continue
        by.setdefault(g, []).append(r)
    headers = ["group", "n_runs", "acc_final", f"acc_min@{warmup_samples}"] + [f"post_min@W{w}" for w in windows] + [f"post_mean@W{w}" for w in windows]
    out_rows: List[List[str]] = []
    for g, rs in sorted(by.items(), key=lambda x: x[0]):
        acc_final = [_safe_float(r.get("acc_final")) for r in rs]
        acc_min = [_safe_float(r.get(f"acc_min@{warmup_samples}")) for r in rs]
        cols: List[str] = [
            g,
            str(len(rs)),
            fmt_mu_std(mean(acc_final), std(acc_final), 4),
            fmt_mu_std(mean(acc_min), std(acc_min), 4),
        ]
        for w in windows:
            v = [_safe_float(r.get(f"post_min@W{w}")) for r in rs]
            cols.append(fmt_mu_std(mean(v), std(v), 4))
        for w in windows:
            v = [_safe_float(r.get(f"post_mean@W{w}")) for r in rs]
            cols.append(fmt_mu_std(mean(v), std(v), 4))
        out_rows.append(cols)
    return md_table(headers, out_rows)


def summarize_track_ac_deltas(rows: List[Dict[str, Any]], windows: Sequence[int]) -> Dict[str, Optional[float]]:
    if not rows:
        return {}
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        s = _safe_int(r.get("seed"))
        g = str(r.get("group") or "")
        if s is None or g not in {"baseline", "v2"}:
            continue
        by_seed.setdefault(int(s), {})[g] = r
    deltas: Dict[str, List[float]] = {f"post_min@W{w}": [] for w in windows}
    for s, m in by_seed.items():
        if "baseline" not in m or "v2" not in m:
            continue
        for w in windows:
            a = _safe_float(m["v2"].get(f"post_min@W{w}"))
            b = _safe_float(m["baseline"].get(f"post_min@W{w}"))
            if a is None or b is None:
                continue
            deltas[f"post_min@W{w}"].append(float(a - b))
        _ = s
    return {k: (float(statistics.mean(v)) if v else None) for k, v in deltas.items()}

def main() -> int:
    args = parse_args()
    trackaa_csv = Path(args.trackaa_csv)
    trackab_csv = Path(args.trackab_csv)
    trackx_csv = Path(args.trackx_csv)
    trackac_csv = Path(args.trackac_csv)
    out_report = Path(args.out_report)
    warmup_samples = int(args.warmup_samples)
    insects_meta = Path(args.insects_meta)
    windows = [int(x) for x in str(args.recovery_windows).split(",") if str(x).strip()]

    rows_aa = read_csv(trackaa_csv)
    rows_ab = read_csv(trackab_csv)
    rows_x = read_csv(trackx_csv)

    # Track AC：复用 Track X 的 run 索引（log_path/run_id），避免扫 logs/results
    rows_ac = build_trackac_from_trackx(
        rows_x,
        insects_meta=insects_meta,
        out_csv=trackac_csv,
        warmup_samples=warmup_samples,
        windows=windows,
        pre_window=int(args.recovery_pre_window),
        roll_points=int(args.recovery_roll_points),
    )

    aa_table, aa_constraints = summarize_track_aa(rows_aa, warmup_samples)
    ab_table = summarize_track_ab(rows_ab)
    ac_table = summarize_track_ac(rows_ac, windows, warmup_samples)
    ac_deltas = summarize_track_ac_deltas(rows_ac, windows)

    # 三问（简明结论）
    q1_lines: List[str] = []
    if aa_constraints:
        ok = [d for d, v in aa_constraints.items() if bool(v.get("ok"))]
        bad = [d for d, v in aa_constraints.items() if not bool(v.get("ok"))]
        q1_lines.append(f"- Track AA 主方法组 D 约束通过：{len(ok)}/{len(aa_constraints)}（{', '.join(ok) if ok else 'N/A'}）")
        for d in sorted(aa_constraints):
            v = aa_constraints[d]
            q1_lines.append(f"  - {d}: miss_mean={fmt(v.get('miss_mean'),3)}, conf_P90_mean={fmt(v.get('conf_p90_mean'),1)}")
        if bad:
            q1_lines.append(f"- 未通过数据集（看 miss/conf_P90）：{', '.join(bad)}")
            q1_lines.append("- 边界条件：gradual drift 的 `drift_start`→`tol500` 口径更苛刻，可能出现“检测在 transition 后段才触发”导致 conf_P90≫500。")
    else:
        q1_lines.append("- Track AA 约束检查：N/A（缺少 Track AA CSV）")

    q2_lines: List[str] = [
        "- Track AB 用 no-drift 合成流（positions/drifts 为空）评估误报：看 confirm_rate_per_10k 与 MTFA_win。",
    ]
    if rows_ab:
        by_ab = group_rows(rows_ab, ("dataset", "group"))
        rates: List[float] = []
        mtfas: List[float] = []
        for _, rs in by_ab.items():
            r_mu = mean([_safe_float(x.get("confirm_rate_per_10k")) for x in rs])
            m_mu = mean([_safe_float(x.get("MTFA_win")) for x in rs])
            if r_mu is not None:
                rates.append(float(r_mu))
            if m_mu is not None:
                mtfas.append(float(m_mu))
        if rates and mtfas:
            q2_lines.append(f"- 本轮 no-drift 观测：confirm_rate_per_10k≈[{min(rates):.2f},{max(rates):.2f}]，MTFA_win≈[{min(mtfas):.1f},{max(mtfas):.1f}]。")

    q3_lines: List[str] = ["- Track AC 用同一批 seeds(1..40) 对比 baseline vs v2，并在 W=500/1000/2000 上重算 post_min/post_mean。"]
    if ac_deltas:
        q3_lines.append("- Δpost_min（v2 - baseline，逐 seed 配对均值）：")
        for w in windows:
            q3_lines.append(f"  - W{w}: {fmt(ac_deltas.get(f'post_min@W{w}'),4)}")

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py = sys.executable
    pyver = platform.python_version()

    report = f"""# NEXT_STAGE V10 Report（泛化 + 边界条件 + 恢复窗口稳健性）

- 生成时间：{now}
- 环境要求（命令）：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- Python：`{py} / {pyver}`

================================================
V10-Track AA（必做）：Gradual/Frequent drift 泛化（合成流）
================================================

产物：`{trackaa_csv}`

说明：本仓库默认仅内置 abrupt 合成流；Track AA 先用 `data.streams.generate_and_save_synth_stream` 生成 non-abrupt（gradual）版本落盘到临时目录，再在运行期间“临时替换” `data/synthetic/<base_dataset_name>/`，跑完立即恢复原始数据。

{aa_table}

**主方法组 D 约束检查（miss_tol500≈0 且 conf_P90<500）**
{chr(10).join(q1_lines) if q1_lines else '- N/A'}

================================================
V10-Track AB（必做）：No-drift sanity（误报成本）
================================================

产物：`{trackab_csv}`

{ab_table}

**解读**
{chr(10).join(q2_lines) if q2_lines else '- N/A'}

================================================
V10-Track AC（必做）：INSECTS 恢复窗口稳健性（baseline vs v2）
================================================

产物：`{trackac_csv}`（由 Track X 的 `log_path/run_id` 精确定位并重算）

{ac_table}

**解读**
{chr(10).join(q3_lines) if q3_lines else '- N/A'}

**配对差（v2 - baseline，逐 seed 均值）**
{chr(10).join([f"- Δpost_min@W{w} = {fmt(ac_deltas.get(f'post_min@W{w}'),4)}" for w in windows]) if ac_deltas else '- N/A'}

================================================
回答三问（必须）
================================================

1) non-abrupt drift 下：看 Track AA 主方法组 D 的 `miss_tol500` 与 `conf_P90`（上面已给约束检查）。
2) no-drift 情况下误报：看 Track AB 的 `confirm_rate_per_10k` 与 `MTFA_win`（并同时核对 `acc_final/mean_acc` 波动）。
3) v2 的恢复收益跨窗口一致性：看 Track AC 的 `post_min@W500/W1000/W2000`（baseline vs v2）。

## V10 后最终默认配置（一句话）
- 检测：`two_stage(candidate=OR,confirm=weighted)` + tuned PH（`error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5`），confirm_theta=0.50，confirm_window=1，confirm_cooldown=200；主口径 `acc_min@sample_idx>=2000`。
- 恢复：INSECTS 上默认启用 `severity v2`（不强制 gating）。
"""
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report, encoding="utf-8")
    print(f"[done] wrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
