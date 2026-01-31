from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

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


def parse_weights(spec: str) -> Optional[Tuple[float, float, float]]:
    if not spec:
        return None
    if spec.strip().lower() in {"none", "null"}:
        return None
    if "=" not in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--ema_severity_weights 需要 3 个数或 key=value")
        return tuple(float(x) for x in parts)  # type: ignore[return-value]
    weights = {"error_rate": 0.0, "divergence": 0.0, "teacher_entropy": 0.0}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--ema_severity_weights 格式错误：{token}（期望 key=value）")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"error", "err"}:
            key = "error_rate"
        if key in {"div", "divergence"}:
            key = "divergence"
        if key in {"entropy", "teacher_entropy"}:
            key = "teacher_entropy"
        if key not in weights:
            raise ValueError(f"--ema_severity_weights 未知 key：{key}")
        weights[key] = float(value)
    return (weights["error_rate"], weights["divergence"], weights["teacher_entropy"])


def format_ratio_tag(value: float) -> str:
    txt = f"{value:.3g}"
    if txt.endswith(".0"):
        txt = txt[:-2]
    return txt.replace(".", "p")


def register_nodrift_configs(n_samples: int) -> None:
    sea_base = dict(
        _streams.SEA_CONFIGS.get("sea_abrupt4")
        or {"concept_ids": [0, 1, 2, 3, 0], "concept_length": 10000}
    )
    _streams.SEA_CONFIGS.setdefault("sea_nodrift", {"concept_ids": [0], "concept_length": int(n_samples)})
    _streams.SEA_CONFIGS.setdefault("sea_gradual_frequent", sea_base)

    sine_base = dict(
        _streams.SINE_DEFAULT.get("sine_abrupt4")
        or {
            "classification_functions": [0, 1, 2, 3, 0],
            "segment_length": 10000,
            "balance_classes": False,
            "has_noise": False,
        }
    )
    _streams.SINE_DEFAULT.setdefault(
        "sine_nodrift",
        {**sine_base, "classification_functions": [0], "segment_length": int(n_samples)},
    )
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
    dataset_name: str,
    variant: str,
    seed: int,
) -> Path:
    log_dir = (
        logs_root
        / "expA_ema_decay"
        / run_tag
        / f"lr{ratio_tag}"
        / dataset_name
        / variant
        / f"seed{seed}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{dataset_name}__{variant}__seed{seed}.csv"


def ensure_run(
    base_cfg: ExperimentConfig,
    *,
    dataset_name: str,
    seed: int,
    labeled_ratio: float,
    monitor_preset: str,
    signal_set: Optional[str],
    log_path: Path,
    device: str,
    variant_cfg: Dict[str, Any],
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
        signal_set=str(signal_set) if signal_set else None,
        **variant_cfg,
    )
    _ = run_experiment(cfg, device=str(device))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ExpA: EMA decay adapt sweep runner")
    p.add_argument("--dataset_type", type=str, required=True, choices=["sea", "sine", "stagger"])
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--labeled_ratio", type=float, required=True)
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--log_root_suffix", type=str, default="expA")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument(
        "--monitor_preset",
        type=str,
        default="entropy_divergence_ph_meta",
    )
    p.add_argument("--signal_set", type=str, default="proxy", choices=["error", "proxy", "all"])
    p.add_argument("--variants", type=str, default="B1,B2,B3,P1,P2")
    p.add_argument("--gamma_high", type=float, default=0.999)
    p.add_argument("--gamma_mid", type=float, default=0.99)
    p.add_argument("--gamma_low", type=float, default=0.95)
    p.add_argument("--ema_gamma_min", type=float, default=0.95)
    p.add_argument("--ema_gamma_max", type=float, default=0.999)
    p.add_argument("--ema_severity_mode", type=str, default="max", choices=["max", "weighted"])
    p.add_argument("--ema_severity_weights", type=str, default="")
    p.add_argument("--ema_severity_smoothing", type=float, default=0.9)
    p.add_argument("--ema_severity_threshold", type=float, default=0.6)
    p.add_argument("--ema_severity_threshold_off", type=float, default=None)
    p.add_argument("--ema_cooldown_steps", type=int, default=200)
    p.add_argument("--ema_use_candidate", action="store_true")
    p.add_argument("--ema_use_drift_flag", action="store_true")
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
    variants = [v.strip().upper() for v in str(args.variants).split(",") if v.strip()]
    if not variants:
        raise ValueError("--variants 不能为空")
    weights = parse_weights(str(args.ema_severity_weights))

    run_tag = str(args.run_tag or "").strip()
    if not run_tag:
        run_tag = f"ema{args.ema_gamma_min:.3g}-{args.ema_gamma_max:.3g}_s{args.ema_severity_smoothing:.2f}"
        run_tag = run_tag.replace(".", "p")

    cfg_map = build_cfg_map(int(seeds[0]))
    base_cfg = resolve_base_cfg(str(args.dataset_name), cfg_map)

    preset_map: Dict[str, Dict[str, Any]] = {
        "B1": {
            "model_variant": "ema_fixed_high",
            "initial_alpha": float(args.gamma_high),
            "ema_decay_mode": "fixed",
            "ema_gamma_fixed": float(args.gamma_high),
            "ema_gamma_min": float(args.ema_gamma_min),
            "ema_gamma_max": float(args.ema_gamma_max),
        },
        "B2": {
            "model_variant": "ema_fixed_mid",
            "initial_alpha": float(args.gamma_mid),
            "ema_decay_mode": "fixed",
            "ema_gamma_fixed": float(args.gamma_mid),
            "ema_gamma_min": float(args.ema_gamma_min),
            "ema_gamma_max": float(args.ema_gamma_max),
        },
        "B3": {
            "model_variant": "ema_fixed_fast",
            "initial_alpha": float(args.gamma_low),
            "ema_decay_mode": "fixed",
            "ema_gamma_fixed": float(args.gamma_low),
            "ema_gamma_min": float(args.ema_gamma_min),
            "ema_gamma_max": float(args.ema_gamma_max),
        },
        "P1": {
            "model_variant": "ema_adapt_cont",
            "initial_alpha": float(args.ema_gamma_max),
            "ema_decay_mode": "severity_continuous",
            "ema_gamma_min": float(args.ema_gamma_min),
            "ema_gamma_max": float(args.ema_gamma_max),
            "ema_severity_mode": str(args.ema_severity_mode),
            "ema_severity_weights": weights,
            "ema_severity_smoothing": float(args.ema_severity_smoothing),
        },
        "P2": {
            "model_variant": "ema_adapt_event",
            "initial_alpha": float(args.ema_gamma_max),
            "ema_decay_mode": "severity_event",
            "ema_gamma_min": float(args.ema_gamma_min),
            "ema_gamma_max": float(args.ema_gamma_max),
            "ema_severity_mode": str(args.ema_severity_mode),
            "ema_severity_weights": weights,
            "ema_severity_smoothing": float(args.ema_severity_smoothing),
            "ema_severity_threshold": float(args.ema_severity_threshold),
            "ema_severity_threshold_off": float(args.ema_severity_threshold_off) if args.ema_severity_threshold_off is not None else None,
            "ema_cooldown_steps": int(args.ema_cooldown_steps),
            "ema_use_candidate": bool(args.ema_use_candidate),
            "ema_use_drift_flag": bool(args.ema_use_drift_flag),
        },
        "P3": {
            "model_variant": "ema_adapt_reverse",
            "initial_alpha": float(args.ema_gamma_min),
            "ema_decay_mode": "severity_event_reverse",
            "ema_gamma_min": float(args.ema_gamma_min),
            "ema_gamma_max": float(args.ema_gamma_max),
            "ema_severity_mode": str(args.ema_severity_mode),
            "ema_severity_weights": weights,
            "ema_severity_smoothing": float(args.ema_severity_smoothing),
            "ema_severity_threshold": float(args.ema_severity_threshold),
            "ema_severity_threshold_off": float(args.ema_severity_threshold_off) if args.ema_severity_threshold_off is not None else None,
            "ema_cooldown_steps": int(args.ema_cooldown_steps),
            "ema_use_candidate": bool(args.ema_use_candidate),
            "ema_use_drift_flag": bool(args.ema_use_drift_flag),
        },
    }

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        if variant not in preset_map:
            raise ValueError(f"未知 variant: {variant}")
        vcfg = preset_map[variant]
        for seed in seeds:
            log_path = build_log_path(
                logs_root=logs_root,
                run_tag=run_tag,
                ratio_tag=ratio_tag,
                dataset_name=str(args.dataset_name),
                variant=str(vcfg["model_variant"]),
                seed=int(seed),
            )
            ensure_run(
                base_cfg,
                dataset_name=str(args.dataset_name),
                seed=int(seed),
                labeled_ratio=labeled_ratio,
                monitor_preset=str(args.monitor_preset),
                signal_set=str(args.signal_set) if args.signal_set else None,
                log_path=log_path,
                device=str(args.device),
                variant_cfg=vcfg,
            )
            rows.append(
                {
                    "dataset_name": str(args.dataset_name),
                    "dataset_type": str(args.dataset_type),
                    "dataset_kind": "nodrift" if "nodrift" in str(args.dataset_name).lower() else "drift",
                    "labeled_ratio": float(labeled_ratio),
                    "variant": str(variant),
                    "model_variant": str(vcfg["model_variant"]),
                    "seed": int(seed),
                    "log_path": str(log_path),
                    "monitor_preset": str(args.monitor_preset),
                    "signal_set": str(args.signal_set),
                    "gamma_high": float(args.gamma_high),
                    "gamma_mid": float(args.gamma_mid),
                    "gamma_low": float(args.gamma_low),
                    "ema_gamma_min": float(args.ema_gamma_min),
                    "ema_gamma_max": float(args.ema_gamma_max),
                    "ema_severity_mode": str(args.ema_severity_mode),
                    "ema_severity_weights": str(args.ema_severity_weights or ""),
                    "ema_severity_smoothing": float(args.ema_severity_smoothing),
                    "ema_severity_threshold": float(args.ema_severity_threshold),
                    "ema_severity_threshold_off": float(args.ema_severity_threshold_off) if args.ema_severity_threshold_off is not None else "",
                    "ema_cooldown_steps": int(args.ema_cooldown_steps),
                    "ema_use_candidate": int(bool(args.ema_use_candidate)),
                    "ema_use_drift_flag": int(bool(args.ema_use_drift_flag)),
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
