#!/usr/bin/env python
"""
Track I：延迟诊断（解释 window vs tol500 口径冲突）。

输出：scripts/TRACKI_DELAY_DIAG.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.streams import generate_default_abrupt_synth_datasets
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track I：延迟诊断（or/weighted/two_stage）")
    p.add_argument("--datasets", type=str, default="sea_abrupt4,stagger_abrupt3")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--weighted_thetas", nargs="+", type=float, default=[0.4, 0.5])
    p.add_argument("--two_stage_theta_map", type=str, default="sea_abrupt4=0.5,stagger_abrupt3=0.4")
    p.add_argument("--two_stage_confirm_window", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--synthetic_root", type=str, default="data/synthetic")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKI_DELAY_DIAG.csv")
    p.add_argument("--tol500", type=int, default=500)
    p.add_argument("--min_separation", type=int, default=200)
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


def parse_theta_map(spec: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--two_stage_theta_map 格式错误：{token}（期望 dataset=theta）")
        k, v = token.split("=", 1)
        mapping[k.strip()] = float(v.strip())
    return mapping


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def load_synth_meta(synth_root: Path, dataset: str, seed: int) -> Tuple[List[int], int]:
    meta_path = synth_root / dataset / f"{dataset}__seed{seed}_meta.json"
    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    drifts = [int(d["start"]) for d in obj.get("drifts", [])]
    horizon = int(obj.get("n_samples", 0) or 0)
    return drifts, horizon


def _safe_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _safe_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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


def read_log_events(csv_path: Path) -> Dict[str, object]:
    """
    读取日志并提取：
    - acc_final/mean_acc/acc_min
    - candidates/confirmed：使用 sample_idx 作为时间轴（与 meta.json drift positions 对齐）
    - vote_count / vote_score 仅用于 candidate 判定（vote_count>=1）
    """
    summary_path = csv_path.with_suffix(".summary.json")
    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            return {
                "acc_final": obj.get("acc_final"),
                "mean_acc": obj.get("mean_acc"),
                "acc_min": obj.get("acc_min"),
                "candidates": [int(x) for x in (obj.get("candidate_sample_idxs") or [])],
                "confirmed": [int(x) for x in (obj.get("confirmed_sample_idxs") or [])],
                "horizon": int(obj.get("horizon") or 0),
            }
        except Exception:
            pass
    accs: List[float] = []
    candidates: List[int] = []
    confirmed: List[int] = []
    last_sample_idx: Optional[int] = None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x = _safe_int(r.get("sample_idx")) or _safe_int(r.get("seen_samples")) or _safe_int(r.get("step"))
            if x is None:
                continue
            last_sample_idx = x
            acc = _safe_float(r.get("metric_accuracy"))
            if acc is not None and not math.isnan(acc):
                accs.append(float(acc))
            vote_count = _safe_int(r.get("monitor_vote_count"))
            # 对非 two_stage 的脚本，candidate_flag 可能退化为 drift_flag；因此优先使用 vote_count
            is_candidate = (vote_count is not None and vote_count >= 1) or (r.get("candidate_flag") == "1")
            if is_candidate:
                candidates.append(int(x))
            if r.get("drift_flag") == "1":
                confirmed.append(int(x))
    return {
        "acc_final": accs[-1] if accs else None,
        "mean_acc": (sum(accs) / len(accs)) if accs else None,
        "acc_min": min(accs) if accs else None,
        "candidates": candidates,
        "confirmed": confirmed,
        "horizon": int(last_sample_idx + 1) if last_sample_idx is not None else 0,
    }


def first_event_in_range(events: Sequence[int], start: int, end: int) -> Optional[int]:
    for t in events:
        if t < start:
            continue
        if t >= end:
            return None
        return int(t)
    return None


def per_drift_delay_rows(
    gt_drifts: Sequence[int],
    candidates: Sequence[int],
    confirmed: Sequence[int],
    *,
    horizon: int,
    tol: int,
) -> List[Dict[str, object]]:
    gt = sorted(int(d) for d in gt_drifts)
    cands = sorted(int(x) for x in candidates)
    confs = sorted(int(x) for x in confirmed)
    rows: List[Dict[str, object]] = []
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < len(gt) else int(horizon)
        first_cand = first_event_in_range(cands, g, end)
        first_conf = first_event_in_range(confs, g, end)
        miss_tol500 = 1
        tol_end = min(end, g + int(tol) + 1)
        conf_in_tol = first_event_in_range(confs, g, tol_end)
        if conf_in_tol is not None:
            miss_tol500 = 0
        rows.append(
            {
                "drift_idx": i,
                "drift_pos": g,
                "segment_end": end,
                "first_candidate_step": first_cand,
                "first_confirmed_step": first_conf,
                "delay_candidate": None if first_cand is None else int(first_cand - g),
                "delay_confirmed": None if first_conf is None else int(first_conf - g),
                "miss_tol500": int(miss_tol500),
            }
        )
    return rows


def ensure_log(
    exp_run: ExperimentRun,
    dataset: str,
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
    run_paths = exp_run.prepare_dataset_run(dataset, model_variant, seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    cfg = replace(
        base_cfg,
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
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = list(args.seeds)
    weights = parse_weights(args.weights)
    theta_map = parse_theta_map(args.two_stage_theta_map)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    generate_default_abrupt_synth_datasets(seeds=seeds, out_root=str(synth_root))

    # 定义模式集合：or / weighted(0.4,0.5) / two_stage(default map)
    modes: List[Tuple[str, Dict[str, object]]] = []
    modes.append(("or", {"trigger_mode": "or", "trigger_threshold": 0.5, "confirm_window": int(args.two_stage_confirm_window)}))
    for th in [float(x) for x in args.weighted_thetas]:
        modes.append((f"weighted_t{th:.2f}", {"trigger_mode": "weighted", "trigger_threshold": th, "confirm_window": int(args.two_stage_confirm_window)}))
    modes.append(("two_stage", {"trigger_mode": "two_stage", "trigger_threshold": None, "confirm_window": int(args.two_stage_confirm_window)}))

    records: List[Dict[str, object]] = []
    for mode_name, mode_cfg in modes:
        exp_run = create_experiment_run(
            experiment_name="trackI_delay_diag",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{mode_name}",
        )
        for dataset in datasets:
            for seed in seeds:
                cfg_map = build_cfg_map(seed)
                if dataset.lower() not in cfg_map:
                    raise ValueError(f"未知 dataset: {dataset}（请在 first_stage_experiments._default_experiment_configs 中配置）")
                base_cfg = cfg_map[dataset.lower()]
                trigger_threshold = float(mode_cfg["trigger_threshold"] or theta_map.get(dataset, 0.5))
                log_path = ensure_log(
                    exp_run,
                    dataset,
                    seed,
                    base_cfg,
                    model_variant="ts_drift_adapt",
                    monitor_preset=args.monitor_preset,
                    trigger_mode=str(mode_cfg["trigger_mode"]),
                    trigger_threshold=trigger_threshold,
                    trigger_weights=weights,
                    confirm_window=int(mode_cfg["confirm_window"]),
                    device=args.device,
                )
                gt_drifts, meta_horizon = load_synth_meta(synth_root, dataset, seed)
                stats = read_log_events(log_path)
                horizon = int(meta_horizon or stats["horizon"] or 0)
                candidates = merge_events(stats["candidates"], int(args.min_separation))  # type: ignore[arg-type]
                confirmed = merge_events(stats["confirmed"], int(args.min_separation))  # type: ignore[arg-type]
                per_drift = per_drift_delay_rows(
                    gt_drifts,
                    candidates,
                    confirmed,
                    horizon=horizon,
                    tol=int(args.tol500),
                )
                for row in per_drift:
                    records.append(
                        {
                            "track": "I",
                            "unit": "sample_idx",
                            "experiment_name": exp_run.experiment_name,
                            "run_id": exp_run.run_id,
                            "mode": mode_name,
                            "dataset": dataset,
                            "seed": seed,
                            "model_variant": "ts_drift_adapt",
                            "monitor_preset": args.monitor_preset,
                            "trigger_mode": str(mode_cfg["trigger_mode"]),
                            "trigger_threshold": float(trigger_threshold),
                            "confirm_window": int(mode_cfg["confirm_window"]),
                            "weights": args.weights,
                            "log_path": str(log_path),
                            "acc_final": stats["acc_final"],
                            "mean_acc": stats["mean_acc"],
                            "acc_min": stats["acc_min"],
                            "candidate_count": len(candidates),
                            "confirmed_count": len(confirmed),
                            **row,
                        }
                    )
    if not records:
        print("[warn] no records")
        return 0
    fieldnames = list(records[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[done] wrote {out_csv} rows={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
