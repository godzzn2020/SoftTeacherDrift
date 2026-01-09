#!/usr/bin/env python
"""
NEXT_STAGE V10 - Track AA（必做）：Gradual/Frequent drift 泛化（合成流）

说明：
- 本仓库默认只内置 abrupt 合成流（sea/sine/stagger）。因此本脚本会用
  `data.streams.generate_and_save_synth_stream` 先生成“non-abrupt（gradual）”版本落盘到临时目录，
  再在运行期间“临时替换” `data/synthetic/<base_dataset_name>/`（跑完立即恢复），从而复用现有训练入口。

固定检测默认（tuned PH, 主方法）：
- monitor_preset=error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5
- trigger=two_stage(candidate=OR,confirm=weighted)
- confirm_theta=0.50, confirm_window=1, confirm_cooldown=200

对每个 dataset 跑 4 组（seeds=1..5）：
A) OR + defaultPH（对照）
B) OR + tunedPH
C) weighted + tunedPH
D) two_stage + tunedPH + cd200（主方法）

输出：scripts/TRACKAA_GENERALIZATION_NONABRUPT.csv（逐 seed）
- dataset 列为 profile 别名（如 sea_gradual_frequent），base_dataset_name 为实际加载的数据集名（如 sea_abrupt4）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.streams import generate_and_save_synth_stream
from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track AA: non-abrupt synthetic generalization (generated)")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--active_synthetic_root", type=str, default="data/synthetic", help="训练时会从这里读取（既有默认）")
    p.add_argument("--tmp_synthetic_root", type=str, default="data/synthetic_v10_tmp/trackAA", help="临时生成 non-abrupt 数据的落盘目录")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAA_GENERALIZATION_NONABRUPT.csv")
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
    p.add_argument("--warmup_samples", type=int, default=2000)
    # 生成数据集列表（2~3 个）：默认选 sea/sine/stagger 的 gradual+frequent 组合
    p.add_argument(
        "--profiles",
        type=str,
        default="sea_gradual_frequent,sine_gradual_frequent,stagger_gradual_frequent",
        help="non-abrupt profile 名（逗号分隔，仅用于 CSV/报告展示；实际 base_dataset 仍为 *_abrupt*）",
    )
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--concept_length", type=int, default=5000)
    p.add_argument("--transition_length", type=int, default=2000)
    return p.parse_args()


def parse_weights(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--weights 格式错误：{token}（期望 key=value）")
        k, v = token.split("=", 1)
        weights[k.strip()] = float(v.strip())
    if not weights:
        raise ValueError("--weights 不能为空")
    return weights


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def merge_events(events: Sequence[int], min_sep: int) -> List[int]:
    if not events:
        return []
    ev = sorted(int(x) for x in events)
    merged = [ev[0]]
    for x in ev[1:]:
        if x - merged[-1] < min_sep:
            continue
        merged.append(x)
    return merged


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def per_drift_first_delays(
    gt_drifts: Sequence[int],
    candidates: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> Tuple[List[float], List[float], List[int]]:
    gt = sorted(int(d) for d in gt_drifts)
    cands = sorted(int(x) for x in candidates)
    confs = sorted(int(x) for x in confirmed)
    delays_cand: List[float] = []
    delays_conf: List[float] = []
    miss_flags: List[int] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_c = first_event_in_range(cands, g, end)
        first_f = first_event_in_range(confs, g, end)
        delays_cand.append(float(end - g) if first_c is None else float(first_c - g))
        delays_conf.append(float(end - g) if first_f is None else float(first_f - g))
        tol_end = min(end, g + int(tol) + 1)
        miss_flags.append(0 if first_event_in_range(confs, g, tol_end) is not None else 1)
    return delays_cand, delays_conf, miss_flags


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))  # type: ignore[arg-type]
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


def compute_metrics_tolerance(
    gt_drifts: Sequence[int],
    detections: Sequence[int],
    horizon: int,
    tolerance: int,
) -> Dict[str, Optional[float]]:
    gt = sorted(int(d) for d in gt_drifts)
    dets = sorted(int(d) for d in detections)
    if not gt or horizon <= 0:
        return {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
    matched: set[int] = set()
    delays: List[int] = []
    missed = 0
    for drift in gt:
        match = None
        for det in dets:
            if det < drift:
                continue
            if det <= drift + tolerance and det not in matched:
                match = det
                break
            if det > drift + tolerance:
                break
        if match is None:
            missed += 1
        else:
            matched.add(match)
            delays.append(int(match - drift))
    false_alarms = [d for d in dets if d not in matched]
    mdr = missed / len(gt) if gt else None
    mtd = (sum(delays) / len(delays)) if delays else None
    if len(false_alarms) >= 2:
        gaps = [b - a for a, b in zip(false_alarms[:-1], false_alarms[1:])]
        mtfa = sum(gaps) / len(gaps)
    elif len(false_alarms) == 1:
        mtfa = float(horizon - false_alarms[0]) if horizon > 0 else None
    else:
        mtfa = None
    if mdr is None or mtd is None or mtfa is None or mdr >= 1.0 or mtd <= 0:
        mtr = None
    else:
        mtr = mtfa / (mtd * (1 - mdr))
    return {
        "MDR": float(mdr) if mdr is not None else None,
        "MTD": float(mtd) if mtd is not None else None,
        "MTFA": float(mtfa) if mtfa is not None else None,
        "MTR": float(mtr) if mtr is not None else None,
    }


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def acc_min_after_warmup(summary: Dict[str, Any], warmup_samples: int) -> Optional[float]:
    series = summary.get("acc_series") or []
    vals = [float(a) for x, a in series if int(x) >= int(warmup_samples)]
    return min(vals) if vals else None


def load_or_generate_meta(
    dataset_type: str,
    base_dataset_name: str,
    seed: int,
    tmp_synth_root: Path,
    drift_cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    meta_path = tmp_synth_root / base_dataset_name / f"{base_dataset_name}__seed{seed}_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    _, meta = generate_and_save_synth_stream(
        dataset_type=str(dataset_type),
        dataset_name=str(base_dataset_name),
        n_samples=int(args.n_samples),
        seed=int(seed),
        out_root=str(tmp_synth_root),
        **drift_cfg,
    )
    return meta


@contextmanager
def swap_in_synth_dataset(*, active_root: Path, base_dataset_name: str, replacement_dir: Path) -> Any:
    active_dir = active_root / base_dataset_name
    backup_root = active_root / "__backup_v10"
    backup_dir = backup_root / base_dataset_name
    if not replacement_dir.exists():
        raise FileNotFoundError(f"replacement_dir 不存在：{replacement_dir}")
    if backup_dir.exists():
        raise RuntimeError(f"检测到未清理的备份目录（请先手动恢复）：{backup_dir}")
    backup_root.mkdir(parents=True, exist_ok=True)
    if not active_dir.exists():
        raise FileNotFoundError(f"active synthetic dir 不存在：{active_dir}")
    try:
        shutil.move(str(active_dir), str(backup_dir))
        shutil.copytree(str(replacement_dir), str(active_dir))
        yield
    finally:
        try:
            if active_dir.exists():
                shutil.rmtree(str(active_dir))
        finally:
            if backup_dir.exists():
                shutil.move(str(backup_dir), str(active_dir))


def ensure_log(
    exp_run: ExperimentRun,
    base_dataset_name: str,
    seed: int,
    base_cfg: ExperimentConfig,
    *,
    model_variant: str,
    monitor_preset: str,
    trigger_mode: str,
    trigger_threshold: float,
    trigger_weights: Dict[str, float],
    confirm_window: int,
    device: str,
) -> Path:
    run_paths = exp_run.prepare_dataset_run(base_dataset_name, model_variant, seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    cfg = replace(
        base_cfg,
        dataset_name=str(base_dataset_name),
        dataset_type=str(base_cfg.dataset_type),
        model_variant=model_variant,
        seed=seed,
        log_path=str(log_path),
        monitor_preset=monitor_preset,
        trigger_mode=trigger_mode,
        trigger_weights=trigger_weights,
        trigger_threshold=float(trigger_threshold),
        confirm_window=int(confirm_window),
    )
    _ = run_experiment(cfg, device=device)
    run_paths.update_legacy_pointer()
    return log_path


def main() -> int:
    args = parse_args()
    seeds = list(args.seeds)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    active_synth_root = Path(args.active_synthetic_root)
    tmp_synth_root = Path(args.tmp_synthetic_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    profiles = [x.strip() for x in str(args.profiles).split(",") if x.strip()]
    cfg_map = build_cfg_map(1)

    tuned_preset = "error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5"
    default_preset = "error_divergence_ph_meta"
    weights_base = {"error_rate": 0.5, "divergence": 0.3, "teacher_entropy": 0.2}
    theta = 0.50
    confirm_window = 1
    cooldown = 200

    groups: List[Dict[str, Any]] = [
        {
            "group": "A_or_defaultPH",
            "monitor_preset": default_preset,
            "trigger_mode": "or",
            "trigger_threshold": 0.5,
            "confirm_window": confirm_window,
            "cooldown": 0,
        },
        {
            "group": "B_or_tunedPH",
            "monitor_preset": tuned_preset,
            "trigger_mode": "or",
            "trigger_threshold": 0.5,
            "confirm_window": confirm_window,
            "cooldown": 0,
        },
        {
            "group": "C_weighted_tunedPH",
            "monitor_preset": tuned_preset,
            "trigger_mode": "weighted",
            "trigger_threshold": theta,
            "confirm_window": confirm_window,
            "cooldown": 0,
        },
        {
            "group": "D_two_stage_tunedPH_cd200",
            "monitor_preset": tuned_preset,
            "trigger_mode": "two_stage",
            "trigger_threshold": theta,
            "confirm_window": confirm_window,
            "cooldown": cooldown,
        },
    ]

    records: List[Dict[str, Any]] = []
    for profile in profiles:
        profile_l = str(profile).lower()
        if profile_l.startswith("sea_"):
            dataset_type = "sea"
            base_dataset_name = "sea_abrupt4"
            base_cfg = cfg_map["sea_abrupt4"]
            step = max(1, int(args.concept_length))
            drift_positions = [int(x) for x in range(step, int(args.n_samples), step)]
            drift_cfg: Dict[str, Any] = {"drift_type": "gradual", "drift_positions": drift_positions, "transition_length": int(args.transition_length)}
        elif profile_l.startswith("sine_"):
            dataset_type = "sine"
            base_dataset_name = "sine_abrupt4"
            base_cfg = cfg_map["sine_abrupt4"]
            drift_cfg = {
                "classification_functions": [0, 1, 2, 3, 0],
                "segment_length": int(args.concept_length),
                "drift_type": "gradual",
                "transition_length": int(args.transition_length),
            }
        elif profile_l.startswith("stagger_"):
            dataset_type = "stagger"
            base_dataset_name = "stagger_abrupt3"
            base_cfg = cfg_map["stagger_abrupt3"]
            drift_cfg = {
                "classification_functions": [0, 1, 2, 0],
                "segment_length": int(args.concept_length) * 2,
                "drift_type": "gradual",
                "transition_length": int(args.transition_length),
            }
        else:
            raise ValueError(f"未知 profile 前缀：{profile}")

        # 先把 non-abrupt 数据生成到 tmp_synth_root/<base_dataset_name>/（不覆盖既有 data/synthetic）
        for seed in seeds:
            _ = load_or_generate_meta(dataset_type, base_dataset_name, seed, tmp_synth_root, drift_cfg, args)

        replacement_dir = tmp_synth_root / base_dataset_name
        with swap_in_synth_dataset(active_root=active_synth_root, base_dataset_name=base_dataset_name, replacement_dir=replacement_dir):
            for g in groups:
                group = str(g["group"])
                exp_run = create_experiment_run(
                    experiment_name="trackAA_generalization_nonabrupt",
                    results_root=results_root,
                    logs_root=logs_root,
                    run_name=f"{profile}_{group}_cd{int(g['cooldown'])}",
                )
                for seed in seeds:
                    meta = load_or_generate_meta(dataset_type, base_dataset_name, seed, tmp_synth_root, drift_cfg, args)
                    gt_drifts = [int(d.get("start")) for d in (meta.get("drifts") or []) if isinstance(d, dict) and d.get("start") is not None]
                    horizon = int(meta.get("n_samples") or 0)

                    weights = dict(weights_base)
                    if int(g["cooldown"]) > 0:
                        weights["confirm_cooldown"] = float(int(g["cooldown"]))
                    log_path = ensure_log(
                        exp_run,
                        base_dataset_name,
                        seed,
                        base_cfg,
                        model_variant="ts_drift_adapt",
                        monitor_preset=str(g["monitor_preset"]),
                        trigger_mode=str(g["trigger_mode"]),
                        trigger_threshold=float(g["trigger_threshold"]),
                        trigger_weights=weights,
                        confirm_window=int(g["confirm_window"]),
                        device=str(args.device),
                    )
                    summ = read_run_summary(log_path)
                    candidates_raw = [int(x) for x in (summ.get("candidate_sample_idxs") or [])]
                    confirmed_raw = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                    candidates_merged = merge_events(candidates_raw, int(args.min_separation))
                    confirmed_merged = merge_events(confirmed_raw, int(args.min_separation))

                    win_m = (
                        compute_detection_metrics(gt_drifts, confirmed_raw, int(horizon))
                        if horizon > 0 and gt_drifts
                        else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
                    )
                    tol_m = (
                        compute_metrics_tolerance(gt_drifts, confirmed_merged, int(horizon), int(args.tol500))
                        if gt_drifts
                        else {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
                    )
                    delays_cand, delays_conf, miss_flags = (
                        per_drift_first_delays(
                            gt_drifts,
                            candidates_merged,
                            confirmed_merged,
                            horizon=int(horizon),
                            tol=int(args.tol500),
                        )
                        if gt_drifts
                        else ([], [], [])
                    )

                    miss_tol500 = float(sum(miss_flags) / len(miss_flags)) if miss_flags else None
                    cand_p50 = percentile(delays_cand, 0.50) if delays_cand else None
                    cand_p90 = percentile(delays_cand, 0.90) if delays_cand else None
                    cand_p99 = percentile(delays_cand, 0.99) if delays_cand else None
                    conf_p50 = percentile(delays_conf, 0.50) if delays_conf else None
                    conf_p90 = percentile(delays_conf, 0.90) if delays_conf else None
                    conf_p99 = percentile(delays_conf, 0.99) if delays_conf else None

                    records.append(
                        {
                            "track": "AA",
                            "dataset": profile,
                            "base_dataset_name": base_dataset_name,
                            "dataset_type": dataset_type,
                            "unit": "sample_idx",
                            "seed": seed,
                            "experiment_name": exp_run.experiment_name,
                            "run_id": exp_run.run_id,
                            "group": group,
                            "monitor_preset": str(g["monitor_preset"]),
                            "trigger_mode": str(g["trigger_mode"]),
                            "confirm_theta": float(g["trigger_threshold"]),
                            "confirm_window": int(g["confirm_window"]),
                            "confirm_cooldown": int(g["cooldown"]),
                            "log_path": str(log_path),
                            "horizon": int(horizon),
                            "n_drifts": int(len(gt_drifts)),
                            "acc_final": summ.get("acc_final"),
                            "acc_min_raw": summ.get("acc_min"),
                            f"acc_min@{int(args.warmup_samples)}": acc_min_after_warmup(summ, int(args.warmup_samples)),
                            "miss_tol500": miss_tol500,
                            "MDR_tol500": tol_m.get("MDR"),
                            "cand_P50": cand_p50,
                            "cand_P90": cand_p90,
                            "cand_P99": cand_p99,
                            "conf_P50": conf_p50,
                            "conf_P90": conf_p90,
                            "conf_P99": conf_p99,
                            "MTFA_win": win_m.get("MTFA"),
                        }
                    )

    if not records:
        print("[warn] no records")
        return 0
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[done] wrote {out_csv} rows={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
