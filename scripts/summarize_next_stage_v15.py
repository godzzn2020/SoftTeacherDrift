#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 汇总入口：
- 先复用 V14 的 summarize 逻辑生成 report/run_index/metrics_table
- 再生成“主报告（瘦身）”与“全量报告（大表）”
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize NEXT_STAGE V15 (Permutation-test Confirm + vote_score)")
    p.add_argument("--trackal_csv", type=str, default="artifacts/v15/tables/TRACKAL_PERM_CONFIRM_SWEEP_V15.csv")
    p.add_argument("--trackam_csv", type=str, default="artifacts/v15/tables/TRACKAM_PERM_DIAG_V15.csv")
    p.add_argument("--out_report", "--out_md", type=str, default="artifacts/v15/reports/NEXT_STAGE_V15_REPORT.md")
    p.add_argument("--out_report_full", "--out_full_md", type=str, default="artifacts/v15/reports/NEXT_STAGE_V15_REPORT_FULL.md")
    p.add_argument("--out_run_index", type=str, default="artifacts/v15/tables/NEXT_STAGE_V15_RUN_INDEX.csv")
    p.add_argument("--out_metrics_table", type=str, default="artifacts/v15/tables/NEXT_STAGE_V15_METRICS_TABLE.csv")
    p.add_argument("--acc_tolerance", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=20, help="主报告顶部 Top-K hard-ok 表行数")
    p.add_argument(
        "--raw_csv",
        "--trackal_raw_csv",
        type=str,
        default="",
        help="可选：稳定性复核的逐 seed 明细 CSV（用于在报告顶部输出均值/方差/最差 case）",
    )
    return p.parse_args()


def _safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a"}:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _is_zero(v: Optional[float], tol: float = 1e-12) -> bool:
    return v is not None and abs(v) <= tol


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _collect_candidate_csv_paths(missing_path: Path) -> List[Path]:
    # NOTE: 只做提示（不自动替换用户输入路径）。
    # 目标：用户常在 scripts/ 与 artifacts/*/tables 之间切换，给出可行动的候选列表。
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        repo_root = Path.cwd()

    scripts_dir = repo_root / "scripts"
    artifacts_dir = repo_root / "artifacts"
    name = str(missing_path.name or "").strip()

    # glob 候选（仅收集“存在且非空”的 CSV）
    found: List[Path] = []
    try:
        found.extend([p for p in artifacts_dir.glob("**/tables/*.csv") if p.is_file() and p.stat().st_size > 0])
    except Exception:
        pass
    try:
        found.extend([p for p in repo_root.glob("**/TRACKAL_PERM_CONFIRM_STABILITY*.csv") if p.is_file() and p.stat().st_size > 0])
    except Exception:
        pass
    try:
        found.extend([p for p in scripts_dir.glob("**/*.csv") if p.is_file() and p.stat().st_size > 0])
    except Exception:
        pass
    if name:
        try:
            found.extend([p for p in repo_root.glob(f"**/{name}") if p.is_file() and p.suffix.lower() == ".csv" and p.stat().st_size > 0])
        except Exception:
            pass

    # 优先把同名候选排前；否则按 token 相关性排序
    tokens = {t.upper() for t in re.findall(r"[A-Za-z0-9]+", name) if len(t) >= 3} if name else set()

    def score(p: Path) -> Tuple[int, str]:
        pn = p.name
        if name and pn == name:
            return 0, str(p)
        if name and name in pn:
            return 1, str(p)
        up = pn.upper()
        if tokens and any(t in up for t in tokens):
            return 2, str(p)
        return 3, str(p)

    uniq: Dict[str, Path] = {}
    for p in found:
        uniq[str(p)] = p
    out = list(uniq.values())
    out.sort(key=score)
    return out


def _print_missing_hint(path: Path) -> None:
    print(f"[error] missing or empty: {path}", file=sys.stderr)
    candidates = _collect_candidate_csv_paths(path)
    print("[hint] candidate existing CSV paths (NOT auto-selected):", file=sys.stderr)
    shown = candidates[:10]
    for p in shown:
        print(f"  - {p}", file=sys.stderr)
    if len(candidates) > len(shown):
        print(f"  ...(+{len(candidates) - len(shown)} more)", file=sys.stderr)


def _has_csv_rows(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row is not None:
                    return True
    except Exception:
        return False
    return False


def _ensure_csv(path: Path) -> None:
    if not path.exists() or not path.is_file():
        _print_missing_hint(path)
        raise FileNotFoundError(f"空或缺失：{path}")
    try:
        if path.stat().st_size == 0:
            _print_missing_hint(path)
            raise FileNotFoundError(f"空或缺失：{path}")
    except Exception:
        _print_missing_hint(path)
        raise FileNotFoundError(f"空或缺失：{path}")
    if not _has_csv_rows(path):
        _print_missing_hint(path)
        raise FileNotFoundError(f"空或缺失：{path}")


def _pick_best_phase(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    # (group,dataset) 可能出现 quick/full；优先 full
    def rank(r: Dict[str, str]) -> int:
        return 0 if str(r.get("phase") or "") == "full" else 1

    best: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in sorted(rows, key=rank):
        g = str(r.get("group") or "")
        d = str(r.get("dataset") or "")
        if not g or not d:
            continue
        best.setdefault((g, d), r)
    return best


def _nd_rate(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> Optional[float]:
    sea = _safe_float(by_group.get(g, {}).get("sea_nodrift", {}).get("confirm_rate_per_10k_mean"))
    sine = _safe_float(by_group.get(g, {}).get("sine_nodrift", {}).get("confirm_rate_per_10k_mean"))
    if sea is None or sine is None:
        return None
    return (sea + sine) / 2.0


def _nd_mtfa(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> Optional[float]:
    sea = _safe_float(by_group.get(g, {}).get("sea_nodrift", {}).get("MTFA_win_mean"))
    sine = _safe_float(by_group.get(g, {}).get("sine_nodrift", {}).get("MTFA_win_mean"))
    if sea is None or sine is None:
        return None
    return (sea + sine) / 2.0


def _drift_acc_final(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> Optional[float]:
    sea = _safe_float(by_group.get(g, {}).get("sea_abrupt4", {}).get("acc_final_mean"))
    sine = _safe_float(by_group.get(g, {}).get("sine_abrupt4", {}).get("acc_final_mean"))
    if sea is None or sine is None:
        return None
    return (sea + sine) / 2.0


def _hard_ok(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> bool:
    sea = by_group.get(g, {}).get("sea_abrupt4", {})
    sine = by_group.get(g, {}).get("sine_abrupt4", {})
    sea_miss = _safe_float(sea.get("miss_tol500_mean"))
    sine_miss = _safe_float(sine.get("miss_tol500_mean"))
    sea_p90 = _safe_float(sea.get("conf_P90_mean"))
    sine_p90 = _safe_float(sine.get("conf_P90_mean"))
    return _is_zero(sea_miss) and _is_zero(sine_miss) and (sea_p90 is not None and sea_p90 < 500.0) and (sine_p90 is not None and sine_p90 < 500.0)


def _violations(by_group: Dict[str, Dict[str, Dict[str, str]]], g: str) -> List[str]:
    sea = by_group.get(g, {}).get("sea_abrupt4", {})
    sine = by_group.get(g, {}).get("sine_abrupt4", {})
    sea_miss = _safe_float(sea.get("miss_tol500_mean"))
    sine_miss = _safe_float(sine.get("miss_tol500_mean"))
    sea_p90 = _safe_float(sea.get("conf_P90_mean"))
    sine_p90 = _safe_float(sine.get("conf_P90_mean"))
    out: List[str] = []
    if not _is_zero(sea_miss) or not _is_zero(sine_miss):
        out.append(f"miss!=0（sea={sea_miss} sine={sine_miss}）")
    if (sea_p90 is None or sea_p90 >= 500.0) or (sine_p90 is None or sine_p90 >= 500.0):
        out.append(f"confP90>=500（sea={sea_p90} sine={sine_p90}）")
    return out or ["N/A"]


def _parse_v14_baseline_nd(v14_report: Path) -> Optional[float]:
    if not v14_report.exists():
        return None
    lines = v14_report.read_text(encoding="utf-8", errors="replace").splitlines()
    header: Optional[List[str]] = None
    for i, line in enumerate(lines):
        if line.startswith("|") and "group" in line and "no_drift_rate" in line:
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            # 下一行是 --- 分隔行；从后续开始找数据行
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                cols = [c.strip() for c in row.strip().strip("|").split("|")]
                if not cols or cols[0] != "A_weighted_n5":
                    continue
                try:
                    idx = header.index("no_drift_rate")
                except ValueError:
                    return None
                if idx >= len(cols):
                    return None
                return _safe_float(cols[idx])
    return None


def _percentile(xs: List[float], q: float) -> Optional[float]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    q = max(0.0, min(1.0, float(q)))
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


def _mean(xs: List[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _var(xs: List[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) < 2:
        return None
    mu = float(sum(vals) / len(vals))
    return float(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))


def _postprocess_report_header_text(text: str, *, title: str, summarize_cmd: str) -> str:
    out = str(text)
    out = out.replace("# NEXT_STAGE V14 Report（Permutation-test Confirm）", title, 1)
    out = out.replace("python scripts/summarize_next_stage_v14.py", summarize_cmd)
    return out


def _detect_version_tag(name: str) -> str:
    s = str(name or "")
    if "V15P3" in s or "V15.3" in s:
        return "V15.3"
    if "V15P2" in s or "V15.2" in s:
        return "V15.2"
    if "V15P1" in s or "V15.1" in s:
        return "V15.1"
    return "V15"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_N/A_"
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{nd}f}"


def _build_slim_report(
    *,
    trackal_csv: Path,
    trackam_csv: Path,
    out_report: Path,
    out_report_full: Path,
    topk: int,
    raw_csv: Optional[Path],
) -> None:
    rows = _read_csv(trackal_csv)
    if not rows:
        _print_missing_hint(trackal_csv)
        raise FileNotFoundError(f"空或缺失：{trackal_csv}")

    best = _pick_best_phase(rows)
    by_group: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (g, d), r in best.items():
        by_group.setdefault(g, {})[d] = r

    baseline_group = "A_weighted_n5" if "A_weighted_n5" in by_group else (sorted(by_group.keys())[0] if by_group else "")
    baseline_nd = _nd_rate(by_group, baseline_group) if baseline_group else None

    hard_ok_groups = [g for g in by_group.keys() if _hard_ok(by_group, g)]
    hard_ok_groups.sort(
        key=lambda g: (
            _nd_rate(by_group, g) if _nd_rate(by_group, g) is not None else float("inf"),
            -(float(_nd_mtfa(by_group, g) or -1e18)),
            g,
        )
    )

    best_acceptance = hard_ok_groups[0] if hard_ok_groups else None

    best_drift_acc_among_hard_ok: Optional[float] = None
    for g in hard_ok_groups:
        acc = _drift_acc_final(by_group, g)
        if acc is None:
            continue
        if best_drift_acc_among_hard_ok is None or float(acc) > float(best_drift_acc_among_hard_ok):
            best_drift_acc_among_hard_ok = float(acc)

    pass_guardrail: Dict[str, bool] = {}
    if best_drift_acc_among_hard_ok is not None:
        thr = float(best_drift_acc_among_hard_ok - 0.01)
        for g in hard_ok_groups:
            acc = _drift_acc_final(by_group, g)
            pass_guardrail[g] = bool(acc is not None and float(acc) >= thr)
    else:
        for g in hard_ok_groups:
            pass_guardrail[g] = False

    hard_ok_pass = [g for g in hard_ok_groups if pass_guardrail.get(g, False)]
    best_with_guardrail = hard_ok_pass[0] if hard_ok_pass else None

    def meta(g: str, key: str) -> str:
        sea = by_group.get(g, {}).get("sea_abrupt4", {})
        v = sea.get(key)
        return str(v) if v is not None else ""

    top_rows: List[List[str]] = []
    for g in hard_ok_groups[: max(0, int(topk))]:
        sea = by_group.get(g, {}).get("sea_abrupt4", {})
        sine = by_group.get(g, {}).get("sine_abrupt4", {})
        top_rows.append(
            [
                g,
                meta(g, "perm_stat"),
                meta(g, "perm_alpha"),
                meta(g, "perm_pre_n"),
                meta(g, "perm_post_n"),
                (meta(g, "perm_side") or "N/A"),
                "True" if pass_guardrail.get(g, False) else "False",
                _fmt(_safe_float(sea.get("miss_tol500_mean")), 3),
                _fmt(_safe_float(sine.get("miss_tol500_mean")), 3),
                _fmt(_safe_float(sea.get("conf_P90_mean")), 1),
                _fmt(_safe_float(sine.get("conf_P90_mean")), 1),
                _fmt(_nd_rate(by_group, g), 3),
                _fmt(_nd_mtfa(by_group, g), 1),
                _fmt(_drift_acc_final(by_group, g), 4),
            ]
        )

    version_tag = "V15"
    name = out_report.name
    version_tag = _detect_version_tag(name)
    title = f"# NEXT_STAGE {version_tag} Report（TopK + 双最优点 + 验收前置）"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append(title)
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- TrackAL：`{trackal_csv}`")
    lines.append(f"- TrackAM：`{trackam_csv}`")
    lines.append(f"- 全量报告：`{out_report_full}`（含全表）")
    lines.append("")
    lines.append("## 说明（口径）")
    lines.append("- V15 在 V14 pipeline 上新增 `vote_score` 的 `perm_test` 统计量分支，并支持更小的 `post_n` 与分片并行。")
    lines.append("- 本节验收：只在 hard-ok 集合（sea/sine miss==0 且 confP90<500）内比较。")
    if baseline_nd is not None:
        lines.append(f"- baseline：`{baseline_group}` no_drift_rate={baseline_nd:.3f}（一律取本次聚合/metrics_table 口径）。")
    lines.append("")
    lines.append("## 双最优点（前置）")
    lines.append(f"- best_drift_acc_among_hard_ok={_fmt(best_drift_acc_among_hard_ok, 4)}（只在 hard-ok 集合内取最大）")

    def _line_for(tag: str, g: Optional[str]) -> str:
        if not g:
            return f"- {tag}=N/A"
        nd = _nd_rate(by_group, g)
        mtfa = _nd_mtfa(by_group, g)
        acc = _drift_acc_final(by_group, g)
        delta_s = ""
        if nd is not None and baseline_nd is not None:
            delta_s = f", Δ_vs_baseline={nd - baseline_nd:+.3f}"
        return f"- {tag}：`{g}` (no_drift_rate={_fmt(nd,3)}{delta_s}; MTFA={_fmt(mtfa,1)}; drift_acc={_fmt(acc,4)})"

    lines.append(_line_for("best_acceptance", best_acceptance))
    lines.append(_line_for("best_with_guardrail", best_with_guardrail))
    if best_acceptance and best_with_guardrail and best_acceptance != best_with_guardrail:
        lines.append("- 注：best_acceptance 与 best_with_guardrail 不同，原因是 Step4 guardrail 过滤导致。")

    if raw_csv is not None and raw_csv.exists():
        lines.append("")
        lines.append("## 稳定性复核摘要（前置）")
        raw_rows = _read_csv(raw_csv)

        def per_seed_nd(group: str) -> Dict[str, float]:
            by_seed: Dict[str, Dict[str, float]] = {}
            for r in raw_rows:
                if str(r.get("group") or "") != group:
                    continue
                seed = str(r.get("seed") or "").strip()
                ds = str(r.get("dataset") or "").strip()
                if not seed:
                    continue
                if ds not in {"sea_nodrift", "sine_nodrift"}:
                    continue
                v = _safe_float(r.get("confirm_rate_per_10k"))
                if v is None:
                    continue
                by_seed.setdefault(seed, {})[ds] = float(v)
            out: Dict[str, float] = {}
            for seed, m in by_seed.items():
                if "sea_nodrift" in m and "sine_nodrift" in m:
                    out[seed] = float((m["sea_nodrift"] + m["sine_nodrift"]) / 2.0)
            return out

        def delta_stats(group: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
            if group == baseline_group:
                return None, None, None, None, None
            b = per_seed_nd(baseline_group)
            g = per_seed_nd(group)
            deltas = [float(g[s] - b[s]) for s in g.keys() if s in b]
            return _mean(deltas), _var(deltas), _percentile(deltas, 0.10), _percentile(deltas, 0.50), _percentile(deltas, 0.90)

        def write_delta_line(tag: str, group: Optional[str]) -> None:
            if not group:
                return
            mu, var, p10, p50, p90 = delta_stats(group)
            if mu is None:
                return
            lines.append(
                f"- {tag} vs baseline：Δ_no_drift_rate mean={_fmt(mu,3)} var={_fmt(var,4)} p10={_fmt(p10,3)} p50={_fmt(p50,3)} p90={_fmt(p90,3)}"
            )

        write_delta_line("best_acceptance", best_acceptance)
        write_delta_line("best_with_guardrail", best_with_guardrail)

        worst: Optional[Dict[str, Any]] = None
        for r in raw_rows:
            miss = _safe_float(r.get("miss_tol500"))
            p90 = _safe_float(r.get("conf_P90"))
            bad = (miss is not None and miss > 0.0) or (p90 is not None and p90 >= 500.0)
            if not bad:
                continue
            score = (p90 if p90 is not None else 0.0) + 1000.0 * (miss if miss is not None else 0.0)
            if worst is None or float(score) > float(worst["score"]):
                worst = {
                    "score": float(score),
                    "group": r.get("group"),
                    "dataset": r.get("dataset"),
                    "seed": r.get("seed"),
                    "miss": miss,
                    "conf_P90": p90,
                }
        if worst is None:
            lines.append("- hard constraint 破坏：未发现任何 seed/dataset 出现 miss>0 或 confP90>=500")
        else:
            lines.append(
                f"- hard constraint 破坏：存在最差 case group={worst['group']} dataset={worst['dataset']} seed={worst['seed']} miss={worst['miss']} confP90={worst['conf_P90']}"
            )
    lines.append("")
    lines.append(f"## Top-K Hard-OK candidates（K={int(topk)}）")
    headers = [
        "group",
        "perm_stat",
        "perm_alpha",
        "perm_pre_n",
        "perm_post_n",
        "perm_side",
        "pass_guardrail",
        "sea_miss",
        "sine_miss",
        "sea_confP90",
        "sine_confP90",
        "no_drift_rate",
        "no_drift_MTFA",
        "drift_acc_final",
    ]
    lines.append(_md_table(headers, top_rows))
    lines.append("")
    lines.append("## 产物")
    lines.append(f"- `{out_report}`")
    lines.append(f"- `{out_report_full}`")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    for p in (args.out_report, args.out_report_full, args.out_run_index, args.out_metrics_table):
        try:
            Path(str(p)).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    trackal_csv = Path(str(args.trackal_csv))
    trackam_csv = Path(str(args.trackam_csv))
    _ensure_csv(trackal_csv)
    _ensure_csv(trackam_csv)
    raw_csv = Path(str(args.raw_csv)) if str(args.raw_csv).strip() else None
    if raw_csv is not None:
        _ensure_csv(raw_csv)
    v14_summarizer = Path(__file__).resolve().parent / "summarize_next_stage_v14.py"
    if not v14_summarizer.exists():
        print(f"[error] missing: {v14_summarizer}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(v14_summarizer),
        "--trackal_csv",
        str(args.trackal_csv),
        "--trackam_csv",
        str(args.trackam_csv),
        "--out_report",
        str(args.out_report_full),
        "--out_run_index",
        str(args.out_run_index),
        "--out_metrics_table",
        str(args.out_metrics_table),
        "--acc_tolerance",
        str(args.acc_tolerance),
    ]
    subprocess.check_call(cmd)
    # 修正 full 报告头部，避免标题/命令误导
    out_full = Path(args.out_report_full)
    if out_full.exists():
        full_text = out_full.read_text(encoding="utf-8", errors="replace")
        ver = _detect_version_tag(str(out_full.name))
        full_text = _postprocess_report_header_text(
            full_text,
            title=f"# NEXT_STAGE {ver} Report FULL（Permutation-test Confirm + vote_score）",
            summarize_cmd="python scripts/summarize_next_stage_v15.py",
        )
        out_full.write_text(full_text, encoding="utf-8")

    _build_slim_report(
        trackal_csv=trackal_csv,
        trackam_csv=trackam_csv,
        out_report=Path(args.out_report),
        out_report_full=Path(args.out_report_full),
        topk=int(args.topk),
        raw_csv=raw_csv.resolve() if raw_csv is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
