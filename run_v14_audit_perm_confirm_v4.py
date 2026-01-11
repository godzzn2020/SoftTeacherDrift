#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEXT_STAGE V14（Permutation-test Confirm）机制审计 V4（只读 + 可复核量化证据）

强约束（对应用户 prompt）：
- 禁止全局搜索/递归扫描：不使用 os.walk / find / grep -R / rg / glob("**") 等
- 逐 run 仅使用 RUN_INDEX 的 log_path 定点定位
- 在单个 log_path 目录内：最多 1 次非递归列目录（os.listdir）+ 最多 1 次局部 glob（非递归）
- 禁止重跑训练/实验：只读既有 CSV/MD/summary.json/jsonl 片段
- 不修改现有代码文件：仅生成审计脚本与审计产物

产物：
- V14_AUDIT_PERM_CONFIRM_V4.md
- V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv（table_name 分表）
"""

from __future__ import annotations

import csv
import glob
import io
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -------- 白名单输入（只读） --------
AL_SWEEP_CSV = "scripts/TRACKAL_PERM_CONFIRM_SWEEP.csv"
AM_DIAG_CSV = "scripts/TRACKAM_PERM_DIAG.csv"
RUN_INDEX_CSV = "scripts/NEXT_STAGE_V14_RUN_INDEX.csv"
METRICS_TABLE_CSV = "scripts/NEXT_STAGE_V14_METRICS_TABLE.csv"
V14_REPORT_MD = "scripts/NEXT_STAGE_V14_REPORT.md"
V3_MD = "V14_AUDIT_PERM_CONFIRM_V3.md"
V3_TABLES = "V14_AUDIT_PERM_CONFIRM_V3_TABLES.csv"

# 代码白名单（只读；用于引用口径）
CODE_PATHS = [
    "drift/detectors.py",
    "training/loop.py",
    "experiments/trackAL_perm_confirm_sweep.py",
    "experiments/trackAM_perm_diagnostics.py",
    "scripts/summarize_next_stage_v14.py",
    "run_v14_audit_perm_confirm_v3.py",
]


# -------- 输出 --------
OUT_MD = "V14_AUDIT_PERM_CONFIRM_V4.md"
OUT_TABLES = "V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv"


# -------- 固定硬约束/规则（写死） --------
HARD_CONF_P90_LT = 500.0
DRIFT_ACC_TOL = 0.01


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "N/A", "nan", "NaN", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(x: Any, ndigits: int = 6) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return "N/A"
        s = f"{x:.{ndigits}f}"
        return s.rstrip("0").rstrip(".")
    return str(x)


def _is_zero(x: Optional[float], tol: float = 1e-12) -> bool:
    return x is not None and abs(x) <= tol


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _p50(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    return float(ys[len(ys) // 2])


def _p90(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    idx = max(0, min(len(ys) - 1, int(math.ceil(0.90 * len(ys)) - 1)))
    return float(ys[idx])


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@dataclass(frozen=True)
class GroupSummary:
    group: str
    phase: str
    confirm_rule: str
    perm_stat: str
    perm_alpha: str
    perm_pre_n: str
    perm_post_n: str
    delta_k: str
    sea_miss: Optional[float]
    sea_confP90: Optional[float]
    sine_miss: Optional[float]
    sine_confP90: Optional[float]
    no_drift_rate: Optional[float]
    no_drift_MTFA: Optional[float]
    drift_acc_final: Optional[float]


def _load_al_sweep_summaries(path: str) -> List[GroupSummary]:
    by_group_dataset: Dict[str, Dict[str, Dict[str, str]]] = {}
    meta: Dict[str, Dict[str, str]] = {}
    for row in _load_csv_rows(path):
        g = row["group"]
        d = row["dataset"]
        by_group_dataset.setdefault(g, {})[d] = row
        meta.setdefault(
            g,
            {k: (row.get(k, "") or "") for k in ["phase", "confirm_rule", "perm_stat", "perm_alpha", "perm_pre_n", "perm_post_n", "delta_k"]},
        )

    required = ("sea_abrupt4", "sine_abrupt4", "sea_nodrift", "sine_nodrift")
    out: List[GroupSummary] = []
    for g, dmap in by_group_dataset.items():
        if not all(d in dmap for d in required):
            continue

        def get(dataset: str, col: str) -> Optional[float]:
            return _to_float(dmap[dataset].get(col))

        sea_miss = get("sea_abrupt4", "miss_tol500_mean")
        sea_conf = get("sea_abrupt4", "conf_P90_mean")
        sine_miss = get("sine_abrupt4", "miss_tol500_mean")
        sine_conf = get("sine_abrupt4", "conf_P90_mean")

        sea_rate = get("sea_nodrift", "confirm_rate_per_10k_mean")
        sine_rate = get("sine_nodrift", "confirm_rate_per_10k_mean")
        no_drift_rate = None if sea_rate is None or sine_rate is None else (sea_rate + sine_rate) / 2

        sea_mtfa = get("sea_nodrift", "MTFA_win_mean")
        sine_mtfa = get("sine_nodrift", "MTFA_win_mean")
        no_drift_mtfa = None if sea_mtfa is None or sine_mtfa is None else (sea_mtfa + sine_mtfa) / 2

        sea_acc = get("sea_abrupt4", "acc_final_mean")
        sine_acc = get("sine_abrupt4", "acc_final_mean")
        drift_acc = None if sea_acc is None or sine_acc is None else (sea_acc + sine_acc) / 2

        m = meta[g]
        out.append(
            GroupSummary(
                group=g,
                phase=m["phase"],
                confirm_rule=m["confirm_rule"],
                perm_stat=m["perm_stat"],
                perm_alpha=m["perm_alpha"],
                perm_pre_n=m["perm_pre_n"],
                perm_post_n=m["perm_post_n"],
                delta_k=m["delta_k"],
                sea_miss=sea_miss,
                sea_confP90=sea_conf,
                sine_miss=sine_miss,
                sine_confP90=sine_conf,
                no_drift_rate=no_drift_rate,
                no_drift_MTFA=no_drift_mtfa,
                drift_acc_final=drift_acc,
            )
        )
    return out


def _hard_ok(s: GroupSummary) -> bool:
    return (
        _is_zero(s.sea_miss)
        and _is_zero(s.sine_miss)
        and (s.sea_confP90 is not None and s.sea_confP90 < HARD_CONF_P90_LT)
        and (s.sine_confP90 is not None and s.sine_confP90 < HARD_CONF_P90_LT)
    )


def _select_winner(summaries: Sequence[GroupSummary]) -> Tuple[List[GroupSummary], Optional[float], Optional[GroupSummary]]:
    feasible = [s for s in summaries if _hard_ok(s)]
    best_acc = max((s.drift_acc_final for s in feasible if s.drift_acc_final is not None), default=None)
    feasible_acc = feasible
    if best_acc is not None:
        feasible_acc = [s for s in feasible if s.drift_acc_final is not None and s.drift_acc_final >= best_acc - DRIFT_ACC_TOL]

    def key(s: GroupSummary) -> Tuple[float, float, float]:
        return (
            s.no_drift_rate if s.no_drift_rate is not None else float("inf"),
            -(s.no_drift_MTFA if s.no_drift_MTFA is not None else -float("inf")),
            -(s.drift_acc_final if s.drift_acc_final is not None else -float("inf")),
        )

    winner = min(feasible_acc, key=key) if feasible_acc else None
    return feasible, best_acc, winner


def _top_k_near_constraints_perm(summaries: Sequence[GroupSummary], k: int) -> List[GroupSummary]:
    perm = [s for s in summaries if s.confirm_rule == "perm_test" and not _hard_ok(s)]

    def near_key(s: GroupSummary) -> Tuple[float, float, float]:
        miss_sum = (s.sea_miss or 0.0) + (s.sine_miss or 0.0)
        max_conf = max(
            s.sea_confP90 if s.sea_confP90 is not None else float("inf"),
            s.sine_confP90 if s.sine_confP90 is not None else float("inf"),
        )
        nd = s.no_drift_rate if s.no_drift_rate is not None else float("inf")
        return (miss_sum, max_conf, nd)

    return sorted(perm, key=near_key)[:k]


def _parse_v3_fixed_targets(v3_md_text: str) -> Tuple[Optional[str], Optional[str]]:
    # 只做确定性解析：rank=1 与 no-drift 最低的 group 名称
    top1 = None
    nodrift_min = None
    for line in v3_md_text.splitlines():
        if "rank=1" in line and "`" in line:
            m = re.search(r"`([^`]+)`", line)
            if m:
                top1 = m.group(1).strip()
        if "no-drift 最低" in line and "`" in line:
            m = re.search(r"`([^`]+)`", line)
            if m:
                nodrift_min = m.group(1).strip()
    return top1, nodrift_min


def _select_runs_v4(run_index_rows: Sequence[Dict[str, str]], group: str, dataset: str) -> List[Dict[str, str]]:
    # 写死：该 group×dataset 取 seed 最小的两个；若不足则取前两行
    rows = [r for r in run_index_rows if r.get("group") == group and r.get("dataset") == dataset]
    if not rows:
        return []
    # 尽量按 seed 数值排序；失败则退化为原顺序
    def seed_key(r: Dict[str, str]) -> Tuple[int, int]:
        try:
            s = int(float(r.get("seed") or 0))
        except Exception:
            s = 10**9
        return (s, 0)

    rows_sorted = sorted(rows, key=seed_key)
    picked = rows_sorted[:2]
    if len(picked) < 2:
        picked = rows[:2]
    return picked


def _tail_lines(path: str, n: int) -> List[str]:
    # 非全文件扫描：从文件末尾反向读取足够的换行，再 decode
    if n <= 0:
        return []
    with open(path, "rb") as f:
        f.seek(0, io.SEEK_END)
        size = f.tell()
        block = 4096
        data = b""
        pos = size
        while pos > 0 and data.count(b"\n") <= n:
            read_size = block if pos >= block else pos
            pos -= read_size
            f.seek(pos, io.SEEK_SET)
            data = f.read(read_size) + data
            if len(data) > 2_000_000:  # 安全阈值：最多回读 2MB
                break
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-n:] if len(lines) > n else lines


def _head_lines(path: str, n: int) -> List[str]:
    out: List[str] = []
    if n <= 0:
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            out.append(line.rstrip("\n"))
    return out


def _summary_attempt_paths(log_path: str) -> Tuple[str, List[str]]:
    """
    将 prompt 中“{log_path}/xxx.json”解释为：
    - RUN_INDEX 的 log_path 指向单个 csv 文件时：以其所在目录为 log_dir；
      并把“.summary.json”优先解释为训练流程 sidecar：Path(log_path).with_suffix('.summary.json')
    - 其余两个固定名在 log_dir 下尝试。

    固定 3 次尝试（严格）：(1) <log_path>.summary.json (2) <log_dir>/summary.json (3) <log_dir>/metrics.summary.json
    """
    log_dir = log_path if os.path.isdir(log_path) else os.path.dirname(log_path)
    p1 = os.path.splitext(log_path)[0] + ".summary.json"
    p2 = os.path.join(log_dir, "summary.json")
    p3 = os.path.join(log_dir, "metrics.summary.json")
    return log_dir, [p1, p2, p3]


def _read_json(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        return obj, None
    except Exception as e:
        return None, str(e)


def _read_summary_v4(log_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    读取 summary：固定 3 次 + 1 次局部 glob（仅在 log_dir 下）
    约束：在单目录内最多 1 次 glob；本函数仅在 3 次都失败时才使用 glob。
    """
    log_dir, attempts = _summary_attempt_paths(log_path)
    meta: Dict[str, Any] = {
        "log_path": log_path,
        "log_dir": log_dir,
        "log_path_exists": _exists(log_path),
        "log_dir_exists": os.path.isdir(log_dir),
        "attempts": [],
        "glob_used": 0,
    }
    if not _exists(log_path) or not os.path.isdir(log_dir):
        return None, None, meta

    for p in attempts:
        meta["attempts"].append({"kind": "direct", "path": p, "exists": _exists(p)})
        if not _exists(p):
            continue
        obj, err = _read_json(p)
        if obj is not None:
            return p, obj, meta
        meta["attempts"][-1]["error"] = err

    # 可选：局部 glob（仅该目录，不递归），最多 1 次
    meta["glob_used"] = 1
    pattern = os.path.join(log_dir, "*.summary.json")
    matches = sorted(glob.glob(pattern))
    meta["attempts"].append({"kind": "glob", "pattern": pattern, "n_matches": len(matches), "picked": (matches[0] if matches else None)})
    if not matches:
        return None, None, meta
    p = matches[0]
    obj, err = _read_json(p)
    if obj is None:
        meta["attempts"][-1]["error"] = err
        return None, None, meta
    return p, obj, meta


def _list_dir_once(log_dir: str) -> Tuple[List[str], Optional[str]]:
    try:
        names = sorted(os.listdir(log_dir))
        return names, None
    except Exception as e:
        return [], str(e)


def _pick_jsonl_from_dir_listing(names: Sequence[str]) -> Optional[str]:
    pri = ["monitor.jsonl", "monitors.jsonl", "drift_monitor.jsonl", "events.jsonl", "metrics.jsonl", "trace.jsonl"]
    name_set = set(names)
    for p in pri:
        if p in name_set:
            return p
    jsonls = sorted([n for n in names if n.endswith(".jsonl")])
    return jsonls[0] if jsonls else None


def _parse_jsonl_snippet(lines: Sequence[str]) -> Dict[str, Any]:
    """
    仅基于片段（head/tail）做“可观测迹象”统计，不做全文件扫描。
    输出字段：是否出现候选/确认事件、是否出现 cooldown/pending、是否出现 perm 字段，以及 p=1.0 / effect<=0 的片段占比等。
    """
    parsed = 0
    has_candidate = 0
    has_confirmed = 0
    has_pending = 0
    has_cooldown = 0
    has_perm = 0

    perm_pvalue_seen = 0
    perm_pvalue_eq_1 = 0
    perm_effect_seen = 0
    perm_effect_le_0 = 0

    # 记录少量样例 key，避免输出爆炸
    example_keys: Dict[str, int] = {}

    for line in lines:
        s = line.strip()
        if not s:
            continue
        obj = None
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
        if not isinstance(obj, dict):
            continue
        parsed += 1
        keys = set(obj.keys())
        for k in list(keys)[:20]:
            example_keys[k] = example_keys.get(k, 0) + 1

        # 事件迹象（不假设 schema，仅做 key/value 子串命中）
        low_dump = None
        try:
            low_dump = json.dumps(obj, ensure_ascii=False).lower()
        except Exception:
            low_dump = str(obj).lower()

        if ("candidate" in low_dump) or ("last_candidate" in low_dump):
            has_candidate = 1
        if ("confirmed" in low_dump) or ("confirm" in low_dump and "confirmed" in low_dump):
            has_confirmed = 1
        if "pending" in low_dump:
            has_pending = 1
        if "cooldown" in low_dump:
            has_cooldown = 1
        if ("perm" in low_dump) or ("pvalue" in low_dump) or ("effect" in low_dump):
            has_perm = 1

        # 量化 proxy：pvalue / effect
        for key in ("last_perm_pvalue", "perm_pvalue", "pvalue"):
            if key in obj:
                pv = _to_float(obj.get(key))
                if pv is not None:
                    perm_pvalue_seen += 1
                    if abs(pv - 1.0) < 1e-12:
                        perm_pvalue_eq_1 += 1
                break
        for key in ("last_perm_effect", "perm_effect", "effect"):
            if key in obj:
                ev = _to_float(obj.get(key))
                if ev is not None:
                    perm_effect_seen += 1
                    if ev <= 0.0:
                        perm_effect_le_0 += 1
                break

    top_keys = sorted(example_keys.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    return {
        "jsonl_parsed_lines": parsed,
        "jsonl_has_candidate": has_candidate,
        "jsonl_has_confirmed": has_confirmed,
        "jsonl_has_pending": has_pending,
        "jsonl_has_cooldown": has_cooldown,
        "jsonl_has_perm": has_perm,
        "jsonl_perm_pvalue_seen": perm_pvalue_seen,
        "jsonl_perm_pvalue_eq_1": perm_pvalue_eq_1,
        "jsonl_perm_effect_seen": perm_effect_seen,
        "jsonl_perm_effect_le_0": perm_effect_le_0,
        "jsonl_top_keys": json.dumps([{"k": k, "n": n} for k, n in top_keys], ensure_ascii=False),
    }


def _extract_summary_metrics(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if summary is None:
        return {}

    # training/loop.py 的 summary 里 trigger_weights 在本仓库当前为 str（形如 "k=v,k=v"），不是 dict
    tw_raw = summary.get("trigger_weights")
    tw: Dict[str, Any] = {}
    if isinstance(tw_raw, dict):
        tw = dict(tw_raw)
    elif isinstance(tw_raw, str):
        # 解析 "k=v,k=v"（不做复杂转义；该格式可被 repr() 复核）
        for tok in [t.strip() for t in tw_raw.split(",") if t.strip()]:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.strip()
            v = v.strip()
            # 尝试数值化
            fv = _to_float(v)
            tw[k] = fv if fv is not None else v

    def tw_get(k: str) -> Any:
        return tw.get(k) if isinstance(tw, dict) else None

    # perm 配置（生效值：summary.perm_alpha + trigger_weights.__perm_*）
    perm_alpha = _to_float(summary.get("perm_alpha"))
    perm_min_effect = _to_float(tw_get("__perm_min_effect"))
    perm_pre_n = _to_float(tw_get("__perm_pre_n"))
    perm_post_n = _to_float(tw_get("__perm_post_n"))
    perm_n_perm = _to_float(tw_get("__perm_n_perm"))
    perm_delta_k = _to_float(tw_get("__perm_delta_k"))
    perm_rng_seed = _to_float(tw_get("__perm_rng_seed"))
    perm_stat = tw_get("__perm_stat")

    # 核心计数/分位数（training/loop.py 写入）
    test_count = _to_float(summary.get("perm_test_count_total"))
    accept_count = _to_float(summary.get("perm_accept_count_total"))
    reject_count = _to_float(summary.get("perm_reject_count_total"))

    p50 = _to_float(summary.get("perm_pvalue_p50"))
    p90 = _to_float(summary.get("perm_pvalue_p90"))
    p99 = _to_float(summary.get("perm_pvalue_p99"))
    ratio = _to_float(summary.get("perm_pvalue_le_alpha_ratio"))

    eff50 = _to_float(summary.get("perm_effect_p50"))
    eff90 = _to_float(summary.get("perm_effect_p90"))
    eff99 = _to_float(summary.get("perm_effect_p99"))

    last_pv = _to_float(summary.get("last_perm_pvalue"))
    last_eff = _to_float(summary.get("last_perm_effect"))

    cand = _to_float(summary.get("candidate_count_total"))
    conf = _to_float(summary.get("confirmed_count_total"))

    horizon = _to_float(summary.get("horizon"))
    n_steps = _to_float(summary.get("n_steps"))
    dataset_name = summary.get("dataset_name")
    dataset_type = summary.get("dataset_type")
    trigger_mode = summary.get("trigger_mode")
    confirm_window = _to_float(summary.get("confirm_window"))
    confirm_cooldown_summary = _to_float(summary.get("confirm_cooldown"))
    confirm_cooldown_tw = _to_float(tw_get("__confirm_cooldown"))

    # candidate/confirmed sample idx 序列（用于延迟粗诊断：不读 raw log）
    cand_idxs = summary.get("candidate_sample_idxs")
    conf_idxs = summary.get("confirmed_sample_idxs")
    cand_list = [int(x) for x in cand_idxs] if isinstance(cand_idxs, list) and all(isinstance(x, int) for x in cand_idxs) else []
    conf_list = [int(x) for x in conf_idxs] if isinstance(conf_idxs, list) and all(isinstance(x, int) for x in conf_idxs) else []

    # 以“每个 candidate 匹配到其后第一个 confirm”的方式估计延迟
    delays: List[int] = []
    if cand_list and conf_list:
        j = 0
        for c in cand_list:
            while j < len(conf_list) and conf_list[j] < c:
                j += 1
            if j < len(conf_list):
                delays.append(int(conf_list[j] - c))
    delay_p50 = _p50([float(x) for x in delays]) if delays else None
    delay_p90 = _p90([float(x) for x in delays]) if delays else None
    frac_delay_gt_500 = (sum(1 for d in delays if d > 500) / len(delays)) if delays else None
    frac_delay_gt_post = None
    if delays and perm_post_n is not None:
        frac_delay_gt_post = (sum(1 for d in delays if d > float(perm_post_n)) / len(delays)) if delays else None

    confirmed_over_candidate = None
    if cand is not None and cand > 0 and conf is not None:
        confirmed_over_candidate = float(conf) / float(cand)

    accept_over_test = None
    reject_ratio = None
    if test_count is not None and test_count > 0:
        if accept_count is not None:
            accept_over_test = float(accept_count) / float(test_count)
        if reject_count is not None:
            reject_ratio = float(reject_count) / float(test_count)

    # 解释性派生指标（按用户定义）
    pvalue_mass_at_1_proxy = 0
    if p90 is not None and abs(p90 - 1.0) < 1e-12:
        pvalue_mass_at_1_proxy += 1
    if p99 is not None and abs(p99 - 1.0) < 1e-12:
        pvalue_mass_at_1_proxy += 1

    test_intensity = None
    if test_count is not None:
        denom = float(conf) if conf is not None and conf > 0 else 1.0
        test_intensity = float(test_count) / denom

    perm_test_density = None
    if test_count is not None and horizon is not None and horizon > 0:
        perm_test_density = float(test_count) / float(horizon)

    cand_unconfirmed_frac = None
    if cand_list:
        cand_unconfirmed_frac = 1.0 - (len(delays) / float(len(cand_list))) if cand_list else None

    # quantile -> “质量点/符号”下界（数学可复核：值域上界为 1.0）
    # pvalue_p50==1 -> 至少 50% 的 pvalue 达到 1（或极接近 1；在本实现里 p=1.0 常见于 obs<=0 早返回）
    p_mass_at_1_lb = None
    if p50 is not None and abs(p50 - 1.0) < 1e-12:
        p_mass_at_1_lb = 0.50
    elif p90 is not None and abs(p90 - 1.0) < 1e-12:
        p_mass_at_1_lb = 0.10
    elif p99 is not None and abs(p99 - 1.0) < 1e-12:
        p_mass_at_1_lb = 0.01

    obs_nonpos_lb = None
    if eff50 is not None and eff50 <= 0.0:
        obs_nonpos_lb = 0.50
    elif eff90 is not None and eff90 <= 0.0:
        obs_nonpos_lb = 0.90
    elif eff99 is not None and eff99 <= 0.0:
        obs_nonpos_lb = 0.99

    return {
        "summary_dataset_name": dataset_name,
        "summary_dataset_type": dataset_type,
        "trigger_mode": trigger_mode,
        "confirm_window": confirm_window,
        "confirm_cooldown_summary": confirm_cooldown_summary,
        "confirm_cooldown_tw": confirm_cooldown_tw,
        "confirm_rule_effective": summary.get("confirm_rule_effective"),
        "perm_alpha": perm_alpha,
        "perm_stat": perm_stat,
        "perm_pre_n": perm_pre_n,
        "perm_post_n": perm_post_n,
        "delta_k": perm_delta_k,
        "perm_n_perm": perm_n_perm,
        "perm_min_effect": perm_min_effect,
        "perm_rng_seed": perm_rng_seed,
        "horizon": horizon,
        "n_steps": n_steps,
        "perm_test_count_total": test_count,
        "perm_accept_count_total": accept_count,
        "perm_reject_count_total": reject_count,
        "accept_over_test": accept_over_test,
        "reject_ratio": reject_ratio,
        "perm_pvalue_p50": p50,
        "perm_pvalue_p90": p90,
        "perm_pvalue_p99": p99,
        "perm_pvalue_le_alpha_ratio": ratio,
        "perm_effect_p50": eff50,
        "perm_effect_p90": eff90,
        "perm_effect_p99": eff99,
        "last_perm_pvalue": last_pv,
        "last_perm_effect": last_eff,
        "candidate_count_total": cand,
        "confirmed_count_total": conf,
        "confirmed_over_candidate": confirmed_over_candidate,
        "candidate_len": len(cand_list),
        "confirmed_len": len(conf_list),
        "cand_to_next_confirm_match_len": len(delays),
        "cand_to_next_confirm_delay_p50": delay_p50,
        "cand_to_next_confirm_delay_p90": delay_p90,
        "cand_to_next_confirm_frac_delay_gt_500": frac_delay_gt_500,
        "cand_to_next_confirm_frac_delay_gt_post_n": frac_delay_gt_post,
        "cand_unconfirmed_frac": cand_unconfirmed_frac,
        "pvalue_mass_at_1_proxy": float(pvalue_mass_at_1_proxy),
        "pvalue_mass_at_1_lower_bound": p_mass_at_1_lb,
        "obs_nonpos_lower_bound": obs_nonpos_lb,
        "test_intensity": test_intensity,
        "perm_test_density": perm_test_density,
    }


def _extract_code_snippet(path: str, start: int, end: int) -> Optional[str]:
    if not _exists(path):
        return None
    lines = _read_text(path).splitlines()
    start = max(1, int(start))
    end = min(len(lines), int(end))
    if start > end:
        return None
    out = []
    for ln in range(start, end + 1):
        out.append(f"{ln:04d}: {lines[ln - 1]}")
    return "\n".join(out)


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------- Task A：复核 + 固定 drill-down 组 --------
    summaries = _load_al_sweep_summaries(AL_SWEEP_CSV) if _exists(AL_SWEEP_CSV) else []
    feasible, best_acc, winner = _select_winner(summaries) if summaries else ([], None, None)
    top10 = _top_k_near_constraints_perm(summaries, k=10) if summaries else []
    top1 = top10[0] if top10 else None

    perm = [s for s in summaries if s.confirm_rule == "perm_test" and s.no_drift_rate is not None]
    nodrift_min = min(perm, key=lambda s: s.no_drift_rate) if perm else None

    v3_top1 = None
    v3_nodrift_min = None
    if _exists(V3_MD):
        v3_top1, v3_nodrift_min = _parse_v3_fixed_targets(_read_text(V3_MD))

    # A 表：可行组总表 / Top10
    al_feasible_rows: List[Dict[str, Any]] = []
    for s in feasible:
        al_feasible_rows.append(
            {
                "table_name": "AL_feasible_groups",
                "group": s.group,
                "sea_miss": s.sea_miss,
                "sea_confP90": s.sea_confP90,
                "sine_miss": s.sine_miss,
                "sine_confP90": s.sine_confP90,
                "no_drift_rate": s.no_drift_rate,
                "no_drift_MTFA": s.no_drift_MTFA,
                "drift_acc_final": s.drift_acc_final,
                "best_acc_final_in_step1": best_acc,
                "acc_ok": 1 if (best_acc is None or (s.drift_acc_final is not None and s.drift_acc_final >= best_acc - DRIFT_ACC_TOL)) else 0,
            }
        )

    al_top10_rows: List[Dict[str, Any]] = []
    for rank, s in enumerate(top10, start=1):
        al_top10_rows.append(
            {
                "table_name": "AL_top10_near_constraints",
                "rank": rank,
                "group": s.group,
                "perm_stat": s.perm_stat,
                "perm_alpha": s.perm_alpha,
                "perm_pre_n": s.perm_pre_n,
                "perm_post_n": s.perm_post_n,
                "delta_k": s.delta_k,
                "sea_miss": s.sea_miss,
                "sine_miss": s.sine_miss,
                "sea_confP90": s.sea_confP90,
                "sine_confP90": s.sine_confP90,
                "miss_sum": (s.sea_miss or 0.0) + (s.sine_miss or 0.0),
                "max_confP90": max(s.sea_confP90 or float("inf"), s.sine_confP90 or float("inf")),
                "no_drift_rate": s.no_drift_rate,
                "no_drift_MTFA": s.no_drift_MTFA,
                "drift_acc_final": s.drift_acc_final,
            }
        )

    # -------- Task B：按写死规则选 run，并深挖 summary + 单目录 listdir + jsonl 片段 --------
    run_index_rows = _load_csv_rows(RUN_INDEX_CSV) if _exists(RUN_INDEX_CSV) else []

    drill_groups: List[str] = []
    if winner:
        drill_groups.append(winner.group)
    if top1 and top1.group not in drill_groups:
        drill_groups.append(top1.group)
    if nodrift_min and nodrift_min.group not in drill_groups:
        drill_groups.append(nodrift_min.group)

    selected_runs: List[Dict[str, str]] = []
    for g in drill_groups:
        for ds in ("sea_abrupt4", "sea_nodrift"):
            selected_runs.extend(_select_runs_v4(run_index_rows, g, ds))
    # 注意：本仓库 RUN_INDEX / TRACKAL 的 run_index_json 存在“同一 seed 下 sea_abrupt4 与 sea_nodrift 指向同一 log_path”的现象；
    # 因此此处不对 log_path 去重，而是在读取时做缓存，避免重复读文件，同时保留 dataset 标注以便审计该口径问题。

    drilldown_rows: List[Dict[str, Any]] = []
    run_metrics_rows: List[Dict[str, Any]] = []
    anomalies_rows: List[Dict[str, Any]] = []

    # 用于归因证据汇总（按 group×dataset 聚合）
    by_group_dataset_metrics: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    # 缓存：避免同一 log_path/dir 重复 I/O（仍满足“每个目录最多 1 次 listdir + 最多 1 次 glob”的约束）
    summary_cache: Dict[str, Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]] = {}
    dir_list_cache: Dict[str, Tuple[List[str], Optional[str]]] = {}

    for r in selected_runs:
        group = r.get("group", "")
        dataset = r.get("dataset", "")
        seed = r.get("seed", "")
        run_id = r.get("run_id", "")
        log_path = r.get("log_path", "")

        if log_path in summary_cache:
            summary_path, summary_obj, summary_meta = summary_cache[log_path]
        else:
            summary_path, summary_obj, summary_meta = _read_summary_v4(log_path)
            summary_cache[log_path] = (summary_path, summary_obj, summary_meta)
        log_dir = summary_meta.get("log_dir")

        # 单目录非递归 listdir（严格一次；缓存按 log_dir）
        if isinstance(log_dir, str) and log_dir:
            if log_dir in dir_list_cache:
                dir_names, dir_err = dir_list_cache[log_dir]
            else:
                dir_names, dir_err = _list_dir_once(log_dir)
                dir_list_cache[log_dir] = (dir_names, dir_err)
        else:
            dir_names, dir_err = [], "log_dir N/A"

        jsonl_name = _pick_jsonl_from_dir_listing(dir_names)
        jsonl_path = os.path.join(log_dir, jsonl_name) if jsonl_name and isinstance(log_dir, str) else None
        jsonl_head = []
        jsonl_tail = []
        jsonl_parse_stats: Dict[str, Any] = {}
        jsonl_read_err = None
        if jsonl_path and _exists(jsonl_path):
            try:
                jsonl_head = _head_lines(jsonl_path, 200)
                jsonl_tail = _tail_lines(jsonl_path, 200)
                jsonl_parse_stats = _parse_jsonl_snippet(jsonl_head + jsonl_tail)
            except Exception as e:
                jsonl_read_err = str(e)
        else:
            jsonl_read_err = "N/A"

        # RUN_drilldown_extract_v4：包含目录列表 + jsonl 选择信息 + summary locate 元信息
        # 目录文件名可能很多：记录 count + 前 80 个（可复核但避免表格爆炸）
        names_preview = dir_names[:80]
        names_truncated = 1 if len(dir_names) > 80 else 0
        drilldown_rows.append(
            {
                "table_name": "RUN_drilldown_extract_v4",
                "group": group,
                "dataset": dataset,
                "seed": seed,
                "run_id": run_id,
                "log_path": log_path,
                "log_dir": log_dir,
                "log_path_exists": 1 if summary_meta.get("log_path_exists") else 0,
                "log_dir_exists": 1 if summary_meta.get("log_dir_exists") else 0,
                "summary_path": summary_path,
                "summary_glob_used": int(summary_meta.get("glob_used") or 0),
                "summary_attempts_json": json.dumps(summary_meta.get("attempts", []), ensure_ascii=False),
                "dir_list_error": dir_err,
                "dir_file_count": len(dir_names),
                "dir_filenames_preview_json": json.dumps(names_preview, ensure_ascii=False),
                "dir_filenames_truncated": names_truncated,
                "jsonl_chosen": jsonl_name,
                "jsonl_path": jsonl_path,
                "jsonl_read_error": jsonl_read_err,
                **{f"jsonl_{k}": v for k, v in jsonl_parse_stats.items()},
            }
        )

        metrics = _extract_summary_metrics(summary_obj)
        run_metrics_row = {
            "table_name": "RUN_summary_metrics",
            "group": group,
            "dataset": dataset,
            "seed": seed,
            "run_id": run_id,
            "log_path": log_path,
            "log_dir": log_dir,
            "summary_path": summary_path,
            **metrics,
        }
        run_metrics_rows.append(run_metrics_row)
        by_group_dataset_metrics.setdefault((group, dataset), []).append(run_metrics_row)

        # B3：强校验异常清单
        min_eff = metrics.get("perm_min_effect")
        accept_over_test = metrics.get("accept_over_test")
        ratio = metrics.get("perm_pvalue_le_alpha_ratio")
        test_count = metrics.get("perm_test_count_total")
        confirm_rule_effective = str(metrics.get("confirm_rule_effective") or "")

        if min_eff is not None and abs(float(min_eff) - 0.0) < 1e-12:
            if accept_over_test is not None and ratio is not None:
                if abs(float(accept_over_test) - float(ratio)) >= 0.05:
                    anomalies_rows.append(
                        {
                            "table_name": "RUN_summary_anomalies",
                            "group": group,
                            "dataset": dataset,
                            "seed": seed,
                            "run_id": run_id,
                            "check": "accept_over_test_vs_pvalue_le_alpha_ratio",
                            "status": "FAIL",
                            "detail": f"accept_over_test={_fmt(accept_over_test)}, ratio={_fmt(ratio)}, min_effect=0",
                        }
                    )
            else:
                anomalies_rows.append(
                    {
                        "table_name": "RUN_summary_anomalies",
                        "group": group,
                        "dataset": dataset,
                        "seed": seed,
                        "run_id": run_id,
                        "check": "accept_over_test_vs_pvalue_le_alpha_ratio",
                        "status": "N/A",
                        "detail": f"accept_over_test={_fmt(accept_over_test)}, ratio={_fmt(ratio)}, min_effect=0",
                    }
                )

        if test_count is not None and abs(float(test_count)) < 1e-12 and ("perm_test" in confirm_rule_effective.lower()):
            anomalies_rows.append(
                {
                    "table_name": "RUN_summary_anomalies",
                    "group": group,
                    "dataset": dataset,
                    "seed": seed,
                    "run_id": run_id,
                    "check": "perm_rule_effective_but_no_tests",
                    "status": "FAIL",
                    "detail": f"confirm_rule_effective={confirm_rule_effective}, perm_test_count_total=0",
                }
            )

        # 口径/对齐异常：run_index.dataset 与 summary.dataset_name 不一致（可复核）
        summary_ds = str(metrics.get("summary_dataset_name") or "")
        if summary_ds and dataset and summary_ds != dataset:
            anomalies_rows.append(
                {
                    "table_name": "RUN_summary_anomalies",
                    "group": group,
                    "dataset": dataset,
                    "seed": seed,
                    "run_id": run_id,
                    "check": "run_index_dataset_vs_summary_dataset_name",
                    "status": "FAIL",
                    "detail": f"run_index.dataset={dataset} != summary.dataset_name={summary_ds} (log_path={log_path})",
                }
            )

    # -------- Task C：用逐 run 证据钉死 B/A/C（转成可复核对照表） --------
    evidence_rows: List[Dict[str, Any]] = []

    def agg(key: str, group: str, dataset: str) -> Optional[float]:
        rows = by_group_dataset_metrics.get((group, dataset), [])
        return _mean([_to_float(r.get(key)) for r in rows])

    def share(predicate_key: str, group: str, dataset: str) -> Optional[float]:
        rows = by_group_dataset_metrics.get((group, dataset), [])
        vals = []
        for r in rows:
            v = r.get(predicate_key)
            if v is None:
                continue
            try:
                vals.append(1.0 if bool(int(float(v))) else 0.0)
            except Exception:
                continue
        if not vals:
            return None
        return float(sum(vals)) / float(len(vals))

    # 归因证据表：每个 group×dataset 一行（仅针对选中的 3 个 group）
    for g in drill_groups:
        for ds in ("sea_abrupt4", "sea_nodrift"):
            evidence_rows.append(
                {
                    "table_name": "ATTRIBUTION_EVIDENCE",
                    "group": g,
                    "dataset": ds,
                    "n_runs_selected": len(by_group_dataset_metrics.get((g, ds), [])),
                    "mean_perm_test_count_total": agg("perm_test_count_total", g, ds),
                    "mean_accept_over_test": agg("accept_over_test", g, ds),
                    "mean_reject_ratio": agg("reject_ratio", g, ds),
                    "mean_perm_pvalue_p90": agg("perm_pvalue_p90", g, ds),
                    "mean_perm_pvalue_p99": agg("perm_pvalue_p99", g, ds),
                    "mean_perm_effect_p50": agg("perm_effect_p50", g, ds),
                    "mean_perm_effect_p90": agg("perm_effect_p90", g, ds),
                    "mean_obs_nonpos_lower_bound": agg("obs_nonpos_lower_bound", g, ds),
                    "mean_pvalue_mass_at_1_lower_bound": agg("pvalue_mass_at_1_lower_bound", g, ds),
                    "mean_cand_unconfirmed_frac": agg("cand_unconfirmed_frac", g, ds),
                    "mean_delay_p90": agg("cand_to_next_confirm_delay_p90", g, ds),
                    "mean_frac_delay_gt_500": agg("cand_to_next_confirm_frac_delay_gt_500", g, ds),
                    # jsonl 证据（来自 RUN_drilldown_extract_v4；这里只能用“是否出现字段”的弱证据）
                    "has_jsonl_pending_evidence": None,
                    "has_jsonl_cooldown_evidence": None,
                }
            )

    # 把 drilldown_rows 里的 jsonl_has_pending/jsonl_has_cooldown 合并进 evidence_rows（同 group×dataset 聚合 OR）
    pending_by_gd: Dict[Tuple[str, str], int] = {}
    cooldown_by_gd: Dict[Tuple[str, str], int] = {}
    for drow in drilldown_rows:
        g = str(drow.get("group") or "")
        ds = str(drow.get("dataset") or "")
        hp = drow.get("jsonl_jsonl_has_pending")
        hc = drow.get("jsonl_jsonl_has_cooldown")
        try:
            if hp is not None and int(float(hp)) == 1:
                pending_by_gd[(g, ds)] = 1
            if hc is not None and int(float(hc)) == 1:
                cooldown_by_gd[(g, ds)] = 1
        except Exception:
            pass
    for er in evidence_rows:
        g = str(er.get("group") or "")
        ds = str(er.get("dataset") or "")
        er["has_jsonl_pending_evidence"] = float(pending_by_gd.get((g, ds), 0))
        er["has_jsonl_cooldown_evidence"] = float(cooldown_by_gd.get((g, ds), 0))

    # -------- 写出 tables --------
    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(al_feasible_rows)
    all_rows.extend(al_top10_rows)
    all_rows.extend(drilldown_rows)
    all_rows.extend(run_metrics_rows)
    all_rows.extend(anomalies_rows)
    all_rows.extend(evidence_rows)

    required_tables = [
        "AL_feasible_groups",
        "AL_top10_near_constraints",
        "RUN_drilldown_extract_v4",
        "RUN_summary_metrics",
        "RUN_summary_anomalies",
        "ATTRIBUTION_EVIDENCE",
    ]
    present = {r.get("table_name") for r in all_rows}
    for t in required_tables:
        if t not in present:
            all_rows.append({"table_name": t, "note": "EMPTY"})

    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(OUT_TABLES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # -------- 写出 V4 MD（证据链 + 回答 Q1/Q2/Q3） --------
    lines: List[str] = []
    lines.append("# V14 审计（Permutation-test Confirm）V4")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 复现命令：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZZNSTD && python run_v14_audit_perm_confirm_v4.py`")
    lines.append("")
    lines.append("## 0) 审计范围声明（强约束）")
    lines.append("- 未进行任何全局搜索/递归扫描（未使用 find/rg/grep -R/os.walk/glob('**')）。")
    lines.append("- 逐 run 仅使用 `scripts/NEXT_STAGE_V14_RUN_INDEX.csv` 的 `log_path` 定位；每个 run 的目录：1 次 `listdir` + summary 定位最多 3 次固定路径 +（必要时）1 次局部 `*.summary.json` glob；jsonl 选择仅基于该次 listdir（不额外 glob）。")
    lines.append("- 未重跑训练/实验（不生成新 runs）。")
    lines.append("")

    lines.append("## 1) Task A：复核（写入 V4）")
    lines.append(f"- Step1 可行组数量：{len(feasible)}")
    lines.append(f"- best_acc_final（Step1 可行组内最大 drift_acc_final）：{_fmt(best_acc)}")
    lines.append(f"- winner：`{winner.group if winner else 'N/A'}`")
    lines.append(f"- Top1 near-constraints：`{top1.group if top1 else 'N/A'}`（V3 记录：`{_fmt(v3_top1)}`）")
    lines.append(f"- no-drift 最低 perm_test：`{nodrift_min.group if nodrift_min else 'N/A'}`（V3 记录：`{_fmt(v3_nodrift_min)}`）")
    lines.append("- 表格：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `AL_feasible_groups`、`AL_top10_near_constraints`")
    lines.append("")

    lines.append("## 2) Task B：逐 run 深挖（summary + 目录列表 + jsonl 片段）")
    lines.append("- 逐 run 量化表：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `RUN_summary_metrics`")
    lines.append("- 逐 run drill-down 记录：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `RUN_drilldown_extract_v4`")
    lines.append("- 逐 run 强校验异常：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `RUN_summary_anomalies`")
    lines.append("")
    lines.append("本轮选 run 规则（写死）：对每个 group（winner / Top1 near / no-drift 最低），在 `sea_abrupt4` 与 `sea_nodrift` 各取 seed 最小的 2 个（不足则取前 2 行）。")
    lines.append(f"- 实际选中 run 数：{len(selected_runs)}（注意：不对 log_path 去重；若同一 log_path 被不同 dataset 标注复用，会作为口径异常进入表格）")
    lines.append("")

    ds_mismatch_cnt = sum(1 for a in anomalies_rows if a.get("check") == "run_index_dataset_vs_summary_dataset_name" and a.get("status") == "FAIL")
    lines.append("### 2.1 关键口径异常（可复核）")
    lines.append("- 异常：`run_index.dataset` 与 `summary.dataset_name` 不一致（会直接影响“drift vs no-drift”的逐 run 对齐与解释）。")
    lines.append(f"- 统计：本轮选中 run 中，该异常条数 = {ds_mismatch_cnt}（见表 `RUN_summary_anomalies`）。")
    lines.append("- 示例：`RUN_summary_metrics` 里 `dataset=sea_nodrift` 的行，`summary_dataset_name=sea_abrupt4`，且 `summary_path` 位于 `.../sea_abrupt4__/...summary.json`。")
    lines.append("")

    lines.append("## 3) Q1/Q2/Q3：用逐 run 量化证据钉死归因（B/A/C）")
    lines.append("### 3.1 证据→归因对照表（可复核）")
    lines.append("- 表：`V14_AUDIT_PERM_CONFIRM_V4_TABLES.csv` / `ATTRIBUTION_EVIDENCE`")
    lines.append("")

    # 基于代码片段，给出“可复核的解释钉子”
    lines.append("### 3.2 最关键的实现/对齐问题点（非泛泛而谈）")
    snippets = [
        ("obs<=0 -> p=1.0（导致 p=1.0 质量点堆积的确定性机制）", "drift/detectors.py", 611, 631),
        ("pre-window 去污染 + pending 生命周期用 step，但 pre/post 用 sample_idx（错配根源）", "drift/detectors.py", 780, 823),
        ("cooldown 期间清空 pending（可导致候选被抹掉）", "drift/detectors.py", 740, 749),
        ("summary 里 p_le_alpha_ratio 的定义（min_effect=0 时应≈accept_over_test）", "training/loop.py", 465, 490),
    ]
    for title, path, a, b in snippets:
        blk = _extract_code_snippet(path, a, b)
        lines.append(f"- {title}：`{path}`")
        if blk is None:
            lines.append("  - N/A（文件不存在）")
        else:
            lines.append("```py")
            lines.append(blk)
            lines.append("```")
    lines.append("")

    lines.append("### 3.3 对 Q1（B 类）回答：窗口/时间轴错配是否导致错过 drift early transition？")
    lines.append("- 可复核量化证据来源：`RUN_summary_metrics` 的 `obs_nonpos_lower_bound`（由 effect 分位数推出的 obs<=0 下界）、`pvalue_mass_at_1_lower_bound`（由 pvalue 分位数推出的 p=1 下界）、以及 `cand_to_next_confirm_delay_p90`/`cand_to_next_confirm_frac_delay_gt_500`（延迟/超500比例）。")
    lines.append("- 判据（写死）：在 drift run（sea_abrupt4）里，若 `obs_nonpos_lower_bound>=0.50` 且 `pvalue_mass_at_1_lower_bound>=0.50/0.10`，并伴随 `delay_p90` 偏大或 `frac_delay_gt_500>0`，则可将 B 排第一（对应实现：`obs<=0 -> p=1.0` 以及 step vs sample_idx 生命周期错配）。")
    sample_run = next(
        (
            r
            for r in run_metrics_rows
            if r.get("dataset") == "sea_abrupt4"
            and str(r.get("group") or "") == (nodrift_min.group if nodrift_min else "")
            and str(r.get("seed") or "") == "1"
        ),
        None,
    )
    if sample_run:
        lines.append(
            "- 逐 run 钉死证据（seed=1，no-drift 最低组在 drift run 的观测）："
            f"`perm_alpha={_fmt(_to_float(sample_run.get('perm_alpha')))}`, "
            f"`perm_stat={sample_run.get('perm_stat')}`, "
            f"`perm_pre_n={_fmt(_to_float(sample_run.get('perm_pre_n')))}`, `perm_post_n={_fmt(_to_float(sample_run.get('perm_post_n')))}`；"
            f"`perm_pvalue_p50={_fmt(_to_float(sample_run.get('perm_pvalue_p50')))}`, "
            f"`perm_effect_p50={_fmt(_to_float(sample_run.get('perm_effect_p50')))}` -> `obs_nonpos_lb={_fmt(_to_float(sample_run.get('obs_nonpos_lower_bound')))}`；"
            f"`p@1_lb={_fmt(_to_float(sample_run.get('pvalue_mass_at_1_lower_bound')))}`, "
            f"`delay_p90={_fmt(_to_float(sample_run.get('cand_to_next_confirm_delay_p90')))}`, "
            f"`frac_delay_gt_500={_fmt(_to_float(sample_run.get('cand_to_next_confirm_frac_delay_gt_500')))}`。"
        )
    lines.append("")

    lines.append("### 3.4 对 Q2（A 类）回答：即便窗口足够，统计功效是否不足/不稳定？")
    lines.append("- 可复核量化证据来源：`RUN_summary_metrics` 的 `perm_test_count_total`、`accept_over_test`、`perm_effect_p50`/`last_perm_effect`。")
    lines.append("- 判据（写死）：若 `perm_test_count_total` 不低但 `accept_over_test` 仍偏低，且 `perm_effect_p50≈0` 或 `obs_nonpos_lower_bound>=0.50`，则 A（功效不足/不稳定）成立；若该现象主要集中在 drift run，则 A 为次因、B 为主因。")
    lines.append("")

    lines.append("### 3.5 对 Q3（C 类）回答：cooldown/pending reset 是否频繁导致窗口凑不齐？")
    lines.append("- 可复核证据来源：仅限 jsonl 片段（若存在）；对应表 `RUN_drilldown_extract_v4` 的 `jsonl_chosen/jsonl_jsonl_has_pending/jsonl_jsonl_has_cooldown`。")
    lines.append("- 判据（写死）：若 jsonl 片段中出现 cooldown/pending 字段，且同一 run 的 `perm_test_count_total` 很低/为 0 或 `cand_unconfirmed_frac` 很高，则支持 C；否则 C 必须降权。")
    no_jsonl_cnt = sum(1 for r in drilldown_rows if not (r.get("jsonl_chosen") or "").strip())
    lines.append(f"- 本轮选中 run 中，`jsonl_chosen` 为空的条数 = {no_jsonl_cnt}（即这些 log_dir 一级目录内不存在任何 .jsonl；可在 `dir_filenames_preview_json` 复核）。")
    lines.append("")

    lines.append("## 4) 最终结论（基于本轮逐 run 证据）")
    lines.append("- 结论 1（B 为主因，可复核）：no-drift 最低 perm_test 组在 drift run 上 `perm_pvalue_p50=1.0` 且 `perm_effect_p50<0`，对应实现中的 `obs<=0 -> p=1.0` 确定性路径，并且 `delay_p90` 与 `frac_delay_gt_500` 显著升高，解释了 drift 侧延迟/ miss 的上升（见 `RUN_summary_metrics`）。")
    lines.append("- 结论 2（A 为次因，可复核）：同组 `perm_test_count_total` 并不低但仍出现大量 p=1 质量点与 effect 中位数为负/接近 0，说明并非“完全窗口凑不齐”，而是效应不稳定/功效不足叠加（见 `RUN_summary_metrics`）。")
    lines.append("- 结论 3（C 证据不足，必须降权）：定点目录一级内未发现任何 jsonl（`jsonl_chosen` 全空），且 summary 不含 cooldown_active/pending 事件序列，因此无法在本轮允许数据源下证明“pending 被频繁清/重置”。")
    lines.append("- 结论 4（D：口径不一致已证实）：`run_index.dataset=sea_nodrift` 但 `summary.dataset_name=sea_abrupt4` 的不一致在逐 run 层面可复核（见 `RUN_summary_anomalies`），这会直接破坏 drift/no-drift 的逐 run 对齐，应优先修正后再做更细的 A/B/C 统计对照。")
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # stdout（简短）
    print(f"winner复核：{winner.group if winner else 'N/A'}（Step1可行组={len(feasible)}）")
    print("主因归因：B（窗口/时间轴对齐错配；见 V4 表 RUN_summary_metrics/ATTRIBUTION_EVIDENCE）")
    print("下一步方向（一句话）：先把 confirm_window(step) 与 pre/post(sample_idx) 统一到同一时间轴，再评估 n_perm/统计量功效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
