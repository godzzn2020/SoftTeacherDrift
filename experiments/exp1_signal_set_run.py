from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import streams as _streams
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


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


def format_ratio_tag(value: float) -> str:
    txt = f"{value:.3g}"
    if txt.endswith(".0"):
        txt = txt[:-2]
    return txt.replace(".", "p")


def format_alpha_tag(value: float) -> str:
    txt = f"{value:.5g}"
    return txt.replace(".", "p")


def register_nodrift_configs(n_samples: int) -> None:
    sea_base = dict(_streams.SEA_CONFIGS.get("sea_abrupt4") or {"concept_ids": [0, 1, 2, 3, 0], "concept_length": 10000})
    _streams.SEA_CONFIGS.setdefault("sea_nodrift", {"concept_ids": [0], "concept_length": int(n_samples)})
    _streams.SEA_CONFIGS.setdefault("sea_gradual_frequent", sea_base)

    sine_base = dict(_streams.SINE_DEFAULT.get("sine_abrupt4") or {"classification_functions": [0, 1, 2, 3, 0], "segment_length": 10000, "balance_classes": False, "has_noise": False})
    _streams.SINE_DEFAULT.setdefault("sine_nodrift", {**sine_base, "classification_functions": [0], "segment_length": int(n_samples)})
    _streams.SINE_DEFAULT.setdefault("sine_gradual_frequent", sine_base)


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[str(cfg.dataset_name).lower()] = cfg
    return mapping


def resolve_base_cfg(dataset_name: str, cfg_map: Dict[str, ExperimentConfig]) -> ExperimentConfig:
    key = str(dataset_name).lower()
    if key == "sea_nodrift":
        key = "sea_abrupt4"
    if key == "sine_nodrift":
        key = "sine_abrupt4"
    cfg = cfg_map.get(key)
    if cfg is None:
        raise KeyError(f"未找到 base_cfg: {dataset_name}")
    return cfg


def build_log_path(
    logs_root: Path,
    run_tag: str,
    ratio_tag: str,
    signal_set: str,
    dataset_name: str,
    seed: int,
) -> Path:
    log_dir = logs_root / "exp1_signal_set" / run_tag / f"lr{ratio_tag}" / f"sig_{signal_set}" / dataset_name / f"seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{dataset_name}__ts_drift_adapt__seed{seed}.csv"


def ensure_run(
    base_cfg: ExperimentConfig,
    *,
    dataset_name: str,
    seed: int,
    labeled_ratio: float,
    monitor_preset: str,
    signal_set: str,
    trigger_weights: Dict[str, Any],
    confirm_theta: float,
    confirm_window: int,
    log_path: Path,
    device: str,
) -> None:
    sp = log_path.with_suffix(".summary.json")
    if log_path.exists() and log_path.stat().st_size > 0 and sp.exists() and sp.stat().st_size > 0:
        return
    cfg = replace(
        base_cfg,
        dataset_name=str(dataset_name),
        dataset_type=str(base_cfg.dataset_type),
        seed=int(seed),
        log_path=str(log_path),
        labeled_ratio=float(labeled_ratio),
        monitor_preset=str(monitor_preset),
        signal_set=str(signal_set),
        trigger_mode="two_stage",
        trigger_weights=dict(trigger_weights),
        trigger_threshold=float(confirm_theta),
        confirm_window=int(confirm_window),
    )
    _ = run_experiment(cfg, device=str(device))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1: signal_set sweep runner")
    p.add_argument("--dataset_type", type=str, required=True, choices=["sea", "sine", "stagger"])
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--labeled_ratio", type=float, required=True)
    p.add_argument("--signal_set", type=str, required=True, choices=["error", "proxy", "all"])
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--log_root_suffix", type=str, default="exp1")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta@error.threshold=0.10,error.min_instances=5",
    )
    p.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    p.add_argument("--confirm_theta", type=float, default=0.50)
    p.add_argument("--confirm_window", type=int, default=3)
    p.add_argument("--confirm_cooldown", type=int, default=200)
    p.add_argument("--perm_alpha", type=float, default=0.025)
    p.add_argument("--perm_pre_n", type=int, default=500)
    p.add_argument("--perm_post_n", type=int, default=10)
    p.add_argument("--perm_n_perm", type=int, default=200)
    p.add_argument("--perm_stat", type=str, default="vote_score")
    p.add_argument("--perm_side", type=str, default="")
    p.add_argument(
        "--candidate_signals",
        type=str,
        default="",
        help="逗号分隔的 candidate 信号列表（如 error_rate,divergence）；为空表示使用全部 detectors",
    )
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--run_tag", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logs_root = Path(str(args.logs_root))
    suffix = str(args.log_root_suffix or "").strip()
    if suffix:
        logs_root = Path(f"{logs_root}_{suffix}")
    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    register_nodrift_configs(int(args.n_samples))

    seeds = parse_int_list(str(args.seeds))
    if not seeds:
        raise ValueError("--seeds 不能为空")
    labeled_ratio = float(args.labeled_ratio)
    ratio_tag = format_ratio_tag(labeled_ratio)
    signal_set = str(args.signal_set).strip().lower()
    if signal_set not in {"error", "proxy", "all"}:
        raise ValueError(f"未知 signal_set: {signal_set}")

    weights = parse_weights(str(args.weights))
    tw = dict(weights)
    tw["__confirm_rule"] = "perm_test"
    tw["__perm_stat"] = str(args.perm_stat or "vote_score").strip()
    tw["__perm_alpha"] = float(args.perm_alpha)
    tw["__perm_pre_n"] = float(args.perm_pre_n)
    tw["__perm_post_n"] = float(args.perm_post_n)
    tw["__perm_n_perm"] = float(args.perm_n_perm)
    tw["__perm_min_effect"] = 0.0
    tw["__perm_rng_seed"] = 0.0
    if str(args.perm_side or "").strip():
        tw["__perm_side"] = str(args.perm_side).strip()
    if str(args.candidate_signals or "").strip():
        tw["__candidate_signals"] = str(args.candidate_signals).strip()
    tw["__confirm_cooldown"] = float(int(args.confirm_cooldown))

    run_tag = str(args.run_tag or "").strip()
    if not run_tag:
        run_tag = f"perm_vote_score_a{format_alpha_tag(float(args.perm_alpha))}_pre{int(args.perm_pre_n)}_post{int(args.perm_post_n)}_n{int(args.perm_n_perm)}"

    cfg_map = build_cfg_map(int(seeds[0]))
    base_cfg = resolve_base_cfg(str(args.dataset_name), cfg_map)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        log_path = build_log_path(
            logs_root=logs_root,
            run_tag=run_tag,
            ratio_tag=ratio_tag,
            signal_set=signal_set,
            dataset_name=str(args.dataset_name),
            seed=int(seed),
        )
        ensure_run(
            base_cfg,
            dataset_name=str(args.dataset_name),
            seed=int(seed),
            labeled_ratio=labeled_ratio,
            monitor_preset=str(args.monitor_preset),
            signal_set=signal_set,
            trigger_weights=tw,
            confirm_theta=float(args.confirm_theta),
            confirm_window=int(args.confirm_window),
            log_path=log_path,
            device=str(args.device),
        )
        rows.append(
            {
                "dataset_name": str(args.dataset_name),
                "dataset_type": str(args.dataset_type),
                "dataset_kind": "nodrift" if "nodrift" in str(args.dataset_name).lower() else "drift",
                "labeled_ratio": float(labeled_ratio),
                "signal_set": signal_set,
                "seed": int(seed),
                "log_path": str(log_path),
                "monitor_preset": str(args.monitor_preset),
                "confirm_theta": float(args.confirm_theta),
                "confirm_window": int(args.confirm_window),
                "confirm_cooldown": int(args.confirm_cooldown),
                "perm_alpha": float(args.perm_alpha),
                "perm_pre_n": int(args.perm_pre_n),
                "perm_post_n": int(args.perm_post_n),
                "perm_n_perm": int(args.perm_n_perm),
                "perm_side": str(args.perm_side or ""),
                "perm_stat": str(args.perm_stat or ""),
                "candidate_signals": str(args.candidate_signals or ""),
                "weights": str(args.weights),
                "run_tag": run_tag,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            fieldnames.append(k)
            seen.add(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
