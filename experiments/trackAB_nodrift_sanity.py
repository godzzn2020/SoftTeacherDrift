#!/usr/bin/env python
"""
NEXT_STAGE V10 - Track AB（必做）：No-drift sanity（误报成本）

做法：
- 生成一个“无漂移”的合成流（默认 profile=sea_nodrift），positions/drifts 为空。
- 为避免覆盖既有 `data/synthetic/<base_dataset_name>/`，本脚本会先把 no-drift 数据落盘到临时目录，
  再在运行期间“临时替换” `data/synthetic/<base_dataset_name>/`（跑完立即恢复）。
- 跑 3 组（seeds=1..5）：
  B) OR + tunedPH
  C) weighted + tunedPH
  D) two_stage + tunedPH + cd200

输出：scripts/TRACKAB_NODRIFT_SANITY.csv（逐 seed）
包含：
- confirm_rate_per_10k（confirmed_count_total / horizon）
- MTFA_win（对无 GT drift：把所有 confirm 视为 false alarms，用 gap 均值估计）
- acc_final / mean_acc
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
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track AB: no-drift sanity (false alarms)")
    p.add_argument("--dataset_type", type=str, default="sea", choices=["sea", "sine", "stagger"])
    p.add_argument("--dataset_alias", type=str, default="sea_nodrift", help="仅用于 CSV/报告展示，不影响实际加载的数据集名")
    p.add_argument(
        "--base_dataset_name",
        type=str,
        default="",
        help="实际加载/生成时使用的 dataset_name（留空则按 dataset_type 选 sea_abrupt4/sine_abrupt4/stagger_abrupt3）",
    )
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--active_synthetic_root", type=str, default="data/synthetic", help="训练时会从这里读取（既有默认）")
    p.add_argument("--tmp_synthetic_root", type=str, default="data/synthetic_v10_tmp/trackAB", help="临时生成 no-drift 数据的落盘目录")
    p.add_argument("--out_csv", type=str, default="scripts/TRACKAB_NODRIFT_SANITY.csv")
    p.add_argument("--confirm_theta", type=float, default=0.5)
    p.add_argument("--confirm_window", type=int, default=1)
    p.add_argument("--confirm_cooldown", type=int, default=200)
    return p.parse_args()


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def read_run_summary(log_csv_path: Path) -> Dict[str, Any]:
    summary_path = log_csv_path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json：{summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def mtfa_from_false_alarms(detections: Sequence[int], horizon: int) -> Optional[float]:
    dets = sorted(int(x) for x in detections)
    if horizon <= 0:
        return None
    if len(dets) >= 2:
        gaps = [b - a for a, b in zip(dets[:-1], dets[1:])]
        return float(sum(gaps) / len(gaps)) if gaps else None
    if len(dets) == 1:
        return float(horizon - dets[0])
    return None


def ensure_dataset(dataset_type: str, base_dataset_name: str, seed: int, n_samples: int, tmp_synth_root: Path) -> Dict[str, Any]:
    meta_path = tmp_synth_root / base_dataset_name / f"{base_dataset_name}__seed{seed}_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    # no-drift：只用一个 concept/function，且覆盖全程
    if dataset_type == "sea":
        drift_cfg = {"concept_ids": [0], "concept_length": int(n_samples), "drift_type": "abrupt"}
    elif dataset_type == "sine":
        drift_cfg = {"classification_functions": [0], "segment_length": int(n_samples), "drift_type": "abrupt"}
    else:  # stagger
        drift_cfg = {"classification_functions": [0], "segment_length": int(n_samples), "drift_type": "abrupt"}
    _, meta = generate_and_save_synth_stream(
        dataset_type=str(dataset_type),
        dataset_name=str(base_dataset_name),
        n_samples=int(n_samples),
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
    monitor_preset: str,
    trigger_mode: str,
    trigger_threshold: float,
    trigger_weights: Dict[str, float],
    confirm_window: int,
    device: str,
) -> Path:
    run_paths = exp_run.prepare_dataset_run(base_dataset_name, "ts_drift_adapt", seed)
    log_path = run_paths.log_csv_path()
    if log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    cfg = replace(
        base_cfg,
        dataset_name=str(base_dataset_name),
        dataset_type=str(base_cfg.dataset_type),
        model_variant="ts_drift_adapt",
        seed=seed,
        log_path=str(log_path),
        monitor_preset=str(monitor_preset),
        trigger_mode=str(trigger_mode),
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

    # 按 dataset_type 选择对应 base_cfg，避免 dataset_type 不匹配
    cfg_map = build_cfg_map(1)
    if str(args.dataset_type) == "sea":
        base_cfg = cfg_map["sea_abrupt4"]
        base_dataset_default = "sea_abrupt4"
    elif str(args.dataset_type) == "sine":
        base_cfg = cfg_map["sine_abrupt4"]
        base_dataset_default = "sine_abrupt4"
    else:
        base_cfg = cfg_map["stagger_abrupt3"]
        base_dataset_default = "stagger_abrupt3"

    base_dataset_name = str(args.base_dataset_name or base_dataset_default)

    tuned_preset = "error_divergence_ph_meta@error.threshold=0.05,error.min_instances=5"
    weights_base = {"error_rate": 0.5, "divergence": 0.3, "teacher_entropy": 0.2}
    theta = float(args.confirm_theta)
    window = int(args.confirm_window)
    cooldown = int(args.confirm_cooldown)

    groups: List[Dict[str, Any]] = [
        {"group": "B_or_tunedPH", "monitor_preset": tuned_preset, "trigger_mode": "or", "trigger_threshold": 0.5, "cooldown": 0},
        {"group": "C_weighted_tunedPH", "monitor_preset": tuned_preset, "trigger_mode": "weighted", "trigger_threshold": theta, "cooldown": 0},
        {"group": "D_two_stage_tunedPH_cd200", "monitor_preset": tuned_preset, "trigger_mode": "two_stage", "trigger_threshold": theta, "cooldown": cooldown},
    ]

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        _ = ensure_dataset(str(args.dataset_type), base_dataset_name, seed, int(args.n_samples), tmp_synth_root)

    # 用临时生成的数据替换 active data/synthetic/<base_dataset_name>，跑完后恢复
    replacement_dir = tmp_synth_root / base_dataset_name
    with swap_in_synth_dataset(active_root=active_synth_root, base_dataset_name=base_dataset_name, replacement_dir=replacement_dir):
        for seed in seeds:
            meta = ensure_dataset(str(args.dataset_type), base_dataset_name, seed, int(args.n_samples), tmp_synth_root)
            horizon = int(meta.get("n_samples") or args.n_samples)
            for g in groups:
                weights = dict(weights_base)
                if int(g["cooldown"]) > 0:
                    weights["confirm_cooldown"] = float(int(g["cooldown"]))
                exp_run = create_experiment_run(
                    experiment_name="trackAB_nodrift_sanity",
                    results_root=results_root,
                    logs_root=logs_root,
                    run_name=f"{args.dataset_alias}_{g['group']}_cd{int(g['cooldown'])}",
                )
                log_path = ensure_log(
                    exp_run,
                    base_dataset_name,
                    seed,
                    base_cfg,
                    monitor_preset=str(g["monitor_preset"]),
                    trigger_mode=str(g["trigger_mode"]),
                    trigger_threshold=float(g["trigger_threshold"]),
                    trigger_weights=weights,
                    confirm_window=window,
                    device=str(args.device),
                )
                summ = read_run_summary(log_path)
                confirmed = [int(x) for x in (summ.get("confirmed_sample_idxs") or [])]
                cc = int(summ.get("confirmed_count_total") or len(confirmed))
                rate = (cc * 10000.0 / float(horizon)) if horizon > 0 else None
                mtfa = mtfa_from_false_alarms(confirmed, int(horizon))
                records.append(
                    {
                        "track": "AB",
                        "dataset": str(args.dataset_alias),
                        "base_dataset_name": base_dataset_name,
                        "dataset_type": str(args.dataset_type),
                        "unit": "sample_idx",
                        "seed": seed,
                        "experiment_name": exp_run.experiment_name,
                        "run_id": exp_run.run_id,
                        "group": str(g["group"]),
                        "monitor_preset": str(g["monitor_preset"]),
                        "trigger_mode": str(g["trigger_mode"]),
                        "confirm_theta": float(g["trigger_threshold"]),
                        "confirm_window": int(window),
                        "confirm_cooldown": int(g["cooldown"]),
                        "log_path": str(log_path),
                        "horizon": int(horizon),
                        "acc_final": summ.get("acc_final"),
                        "mean_acc": summ.get("mean_acc"),
                        "acc_min_raw": summ.get("acc_min"),
                        "confirmed_count_total": cc,
                        "confirm_rate_per_10k": rate,
                        "MTFA_win": mtfa,
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
