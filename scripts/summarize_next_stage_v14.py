#!/usr/bin/env python
"""汇总 NEXT_STAGE V14（Track AL/AM）并生成报告与本轮索引表。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V14 (Permutation-test Confirm)")
    p.add_argument("--trackal_csv", type=str, default="scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv")
    p.add_argument("--trackam_csv", type=str, default="scripts/TRACKAM_PERM_DIAG.csv")
    p.add_argument("--out_report", type=str, default="scripts/NEXT_STAGE_V14_REPORT.md")
    p.add_argument("--out_run_index", type=str, default="scripts/NEXT_STAGE_V14_RUN_INDEX.csv")
    p.add_argument("--out_metrics_table", type=str, default="scripts/NEXT_STAGE_V14_METRICS_TABLE.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
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
            fieldnames.append(k)
            seen.add(k)
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
    except Exception:
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


def pick_best_rows_by_phase(trackal_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    # 对同一 (group,dataset) 可能存在 quick/full 两行；优先 full
    def rank(r: Dict[str, str]) -> int:
        return 0 if str(r.get("phase") or "") == "full" else 1

    best: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in sorted(trackal_rows, key=rank):
        g = str(r.get("group") or "")
        d = str(r.get("dataset") or "")
        if not g or not d:
            continue
        best.setdefault((g, d), r)
    return best


def summarize_groups(trackal_rows: List[Dict[str, str]], acc_tol: float) -> Dict[str, Any]:
    if not trackal_rows:
        return {"winner": None, "reason": "未找到 Track AL CSV", "rows": []}

    best_gd = pick_best_rows_by_phase(trackal_rows)

    by_group: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (g, d), r in best_gd.items():
        by_group.setdefault(g, {})[d] = r

    def drift_acc(g: str) -> float:
        sea = _safe_float(by_group[g].get("sea_abrupt4", {}).get("acc_final_mean"))
        sine = _safe_float(by_group[g].get("sine_abrupt4", {}).get("acc_final_mean"))
        vals = [v for v in [sea, sine] if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("-inf")

    def nd_rate(g: str) -> float:
        sea = _safe_float(by_group[g].get("sea_nodrift", {}).get("confirm_rate_per_10k_mean"))
        sine = _safe_float(by_group[g].get("sine_nodrift", {}).get("confirm_rate_per_10k_mean"))
        vals = [v for v in [sea, sine] if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("inf")

    def nd_mtfa(g: str) -> float:
        sea = _safe_float(by_group[g].get("sea_nodrift", {}).get("MTFA_win_mean"))
        sine = _safe_float(by_group[g].get("sine_nodrift", {}).get("MTFA_win_mean"))
        vals = [v for v in [sea, sine] if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("-inf")

    eligible: List[str] = []
    for g, dsmap in by_group.items():
        sea = dsmap.get("sea_abrupt4")
        sine = dsmap.get("sine_abrupt4")
        if not sea or not sine:
            continue
        sea_miss = _safe_float(sea.get("miss_tol500_mean"))
        sine_miss = _safe_float(sine.get("miss_tol500_mean"))
        sea_conf = _safe_float(sea.get("conf_P90_mean"))
        sine_conf = _safe_float(sine.get("conf_P90_mean"))
        if sea_miss is None or sine_miss is None or sea_conf is None or sine_conf is None:
            continue
        if sea_miss <= 1e-12 and sine_miss <= 1e-12 and sea_conf < 500 and sine_conf < 500:
            eligible.append(g)

    if not eligible:
        return {"winner": None, "reason": "无满足 drift 约束的候选", "rows": []}

    best_acc = max([drift_acc(g) for g in eligible] or [float("-inf")])
    eligible2 = [g for g in eligible if drift_acc(g) >= best_acc - float(acc_tol)]
    if not eligible2:
        eligible2 = eligible

    eligible2.sort(key=lambda g: (nd_rate(g), -nd_mtfa(g), g))
    winner = eligible2[0]
    reason = (
        "Step1: sea_abrupt4 & sine_abrupt4 满足 miss_tol500_mean==0 且 conf_P90_mean<500；"
        "Step2: 最小化 no-drift confirm_rate_per_10k（sea_nodrift+sine_nodrift 平均）；"
        "Step3: 并列时最大化 no-drift MTFA_win；"
        "Step4: drift_acc_final_mean 不低于 best-0.01。"
    )

    # build summary table rows per group
    rows_out: List[Dict[str, Any]] = []
    for g in sorted(by_group.keys()):
        sea_d = by_group[g].get("sea_abrupt4", {})
        sine_d = by_group[g].get("sine_abrupt4", {})
        sea_n = by_group[g].get("sea_nodrift", {})
        sine_n = by_group[g].get("sine_nodrift", {})
        base = sea_d or sine_d or sea_n or sine_n
        rows_out.append(
            {
                "group": g,
                "phase": str(base.get("phase") or ""),
                "confirm_rule": str(base.get("confirm_rule") or ""),
                "perm_stat": str(base.get("perm_stat") or ""),
                "perm_alpha": str(base.get("perm_alpha") or ""),
                "perm_pre_n": str(base.get("perm_pre_n") or ""),
                "perm_post_n": str(base.get("perm_post_n") or ""),
                "delta_k": str(base.get("delta_k") or ""),
                "sea_miss": _safe_float(sea_d.get("miss_tol500_mean")),
                "sea_confP90": _safe_float(sea_d.get("conf_P90_mean")),
                "sine_miss": _safe_float(sine_d.get("miss_tol500_mean")),
                "sine_confP90": _safe_float(sine_d.get("conf_P90_mean")),
                "no_drift_rate": nd_rate(g),
                "no_drift_MTFA": nd_mtfa(g),
                "drift_acc_final": drift_acc(g),
            }
        )

    return {"winner": winner, "reason": reason, "rows": rows_out, "by_group": by_group}


def build_run_index_rows(trackal_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in trackal_rows:
        run_index_json = r.get("run_index_json") or "{}"
        try:
            idx = json.loads(run_index_json)
        except Exception:
            continue
        if not isinstance(idx, dict):
            continue
        group = str(r.get("group") or "")
        phase = str(r.get("phase") or "")
        dataset = str(r.get("dataset") or "")
        ds = idx.get(dataset)
        runs = ds.get("runs") if isinstance(ds, dict) else None
        if not isinstance(runs, list):
            continue
        for item in runs:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "track": "AL",
                    "phase": phase,
                    "group": group,
                    "dataset": dataset,
                    "base_dataset_name": str((ds or {}).get("base_dataset_name") if isinstance(ds, dict) else ""),
                    "seed": item.get("seed"),
                    "run_id": item.get("run_id"),
                    "log_path": item.get("log_path"),
                }
            )
    return out


def join_metrics(trackal_rows: List[Dict[str, str]], trackam_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # 将 Track AM 的诊断列按 (group,dataset,phase) 左连接到 Track AL
    am_map: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in trackam_rows:
        key = (str(r.get("group") or ""), str(r.get("dataset") or ""), str(r.get("phase") or ""))
        am_map[key] = r

    out: List[Dict[str, Any]] = []
    for r in trackal_rows:
        key = (str(r.get("group") or ""), str(r.get("dataset") or ""), str(r.get("phase") or ""))
        merged: Dict[str, Any] = dict(r)
        am = am_map.get(key)
        if am:
            for k, v in am.items():
                if k in {"track", "group", "dataset", "phase"}:
                    continue
                merged[f"AM_{k}"] = v
        out.append(merged)
    return out


def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<failed: {e}>"


def main() -> int:
    args = parse_args()
    rows_al = read_csv(Path(args.trackal_csv))
    rows_am = read_csv(Path(args.trackam_csv))

    summary = summarize_groups(rows_al, acc_tol=float(args.acc_tolerance))
    winner = summary.get("winner")
    reason = summary.get("reason") or "N/A"
    table_items = summary.get("rows") or []

    # baseline group
    baseline = None
    for it in table_items:
        if str(it.get("group") or "").startswith("A_weighted"):
            # prefer n20 baseline
            if baseline is None:
                baseline = it
            elif str(it.get("group") or "").endswith("_n20") and not str(baseline.get("group") or "").endswith("_n20"):
                baseline = it

    delta_line = "N/A"
    if winner and baseline:
        b = _safe_float(baseline.get("no_drift_rate"))
        w = None
        for it in table_items:
            if str(it.get("group") or "") == str(winner):
                w = _safe_float(it.get("no_drift_rate"))
                break
        if b is not None and w is not None:
            delta_line = f"{b:.3f} → {w:.3f} (Δ={w-b:+.3f})"

    headers = [
        "group",
        "phase",
        "confirm_rule",
        "perm_stat",
        "perm_alpha",
        "perm_pre_n",
        "perm_post_n",
        "delta_k",
        "sea_miss",
        "sea_confP90",
        "sine_miss",
        "sine_confP90",
        "no_drift_rate",
        "no_drift_MTFA",
        "drift_acc_final",
    ]
    table_rows: List[List[str]] = []
    for it in table_items:
        table_rows.append(
            [
                str(it.get("group") or ""),
                str(it.get("phase") or ""),
                str(it.get("confirm_rule") or ""),
                str(it.get("perm_stat") or ""),
                str(it.get("perm_alpha") or ""),
                str(it.get("perm_pre_n") or ""),
                str(it.get("perm_post_n") or ""),
                str(it.get("delta_k") or ""),
                fmt(_safe_float(it.get("sea_miss")), 3),
                fmt(_safe_float(it.get("sea_confP90")), 1),
                fmt(_safe_float(it.get("sine_miss")), 3),
                fmt(_safe_float(it.get("sine_confP90")), 1),
                fmt(_safe_float(it.get("no_drift_rate")), 3),
                fmt(_safe_float(it.get("no_drift_MTFA")), 1),
                fmt(_safe_float(it.get("drift_acc_final")), 4),
            ]
        )

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    which_py = run_cmd(["bash", "-lc", "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && which python"])
    py_v = run_cmd(["bash", "-lc", "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python -V"])

    win_line = f"`{winner}`" if winner else "N/A"

    interpret_lines: List[str] = []
    if winner:
        interpret_lines.append(f"- winner：{win_line}（见 Track AL）")
        interpret_lines.append(f"- no-drift confirm_rate_per_10k（平均）变化：{delta_line}")
    else:
        interpret_lines.append(f"- 未找到满足 drift 约束的 winner：{reason}")

    # 额外：给出 perm_test 在本轮网格中的最佳降误报候选（即使不满足 drift 硬约束，也便于下一步迭代）
    perm_items = [it for it in table_items if str(it.get("confirm_rule") or "") == "perm_test"]
    def _k(it: dict) -> tuple[float, float, str]:
        nd = _safe_float(it.get("no_drift_rate"))
        sea_m = _safe_float(it.get("sea_miss"))
        sine_m = _safe_float(it.get("sine_miss"))
        penalty = 0.0
        if sea_m is not None:
            penalty += float(sea_m)
        if sine_m is not None:
            penalty += float(sine_m)
        return (float(nd) if nd is not None else float("inf"), penalty, str(it.get("group") or ""))
    if perm_items:
        perm_items.sort(key=_k)
        best_perm = perm_items[0]
        interpret_lines.append(
            "- perm_test 最佳降误报候选（网格内）："
            f"`{best_perm.get('group')}` "
            f"no_drift_rate={fmt(_safe_float(best_perm.get('no_drift_rate')), 3)}, "
            f"sea(miss={fmt(_safe_float(best_perm.get('sea_miss')), 3)},confP90={fmt(_safe_float(best_perm.get('sea_confP90')), 1)}), "
            f"sine(miss={fmt(_safe_float(best_perm.get('sine_miss')), 3)},confP90={fmt(_safe_float(best_perm.get('sine_confP90')), 1)})"
        )
    if rows_am:
        interpret_lines.append(f"- Track AM：已生成 `{args.trackam_csv}` rows={len(rows_am)}（p-value 分布/confirmed-candidate 比例见诊断表）")
    else:
        interpret_lines.append("- Track AM：未运行或无产物（可选）")

    report = f"""# NEXT_STAGE V14 Report（Permutation-test Confirm）

- 生成时间：{now}
- 环境确认：
  - `which python` -> `{which_py}`
  - `python -V` -> `{py_v}`

## 关键口径与硬约束
- drift 约束：`sea_abrupt4` + `sine_abrupt4` 上 `miss_tol500_mean==0` 且 `conf_P90_mean<500`
- 目标：在满足约束下最小化 no-drift `confirm_rate_per_10k`（`sea_nodrift` + `sine_nodrift` 平均；次选最大化 no-drift `MTFA_win`）

## Track AL：Perm-confirm sweep
- 产物：`{args.trackal_csv}`

{md_table(headers, table_rows)}

**winner 选择规则（写死）**
- {reason}

**winner**
- {win_line}
- no-drift confirm_rate_per_10k 下降幅度（vs baseline）：{delta_line}

## Track AM：机制诊断（可选）
""" + "\n".join([f"- {x}" for x in interpret_lines]) + f"""

## 可复制运行命令
- `source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD`
- `python experiments/trackAL_perm_confirm_sweep.py`
- `python experiments/trackAM_perm_diagnostics.py`  （可选）
- `python scripts/summarize_next_stage_v14.py`
"""

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report, encoding="utf-8")

    # RUN_INDEX：不扫描 logs，只从 run_index_json 展开
    run_index_rows = build_run_index_rows(rows_al)
    write_csv(Path(args.out_run_index), run_index_rows)

    # METRICS_TABLE：Track AL（左连接 Track AM 的诊断列）
    metrics_rows = join_metrics(rows_al, rows_am)
    write_csv(Path(args.out_metrics_table), metrics_rows)

    print(f"[done] wrote {out_report}")
    print(f"[done] wrote {args.out_run_index} rows={len(run_index_rows)}")
    print(f"[done] wrote {args.out_metrics_table} rows={len(metrics_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
