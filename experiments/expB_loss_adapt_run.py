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
            raise ValueError("--loss_severity_weights expects 3 values or key=value")
        return tuple(float(x) for x in parts)  # type: ignore[return-value]
    weights = {"error_rate": 0.0, "divergence": 0.0, "teacher_entropy": 0.0}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--loss_severity_weights bad token: {token} (expected key=value)")
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
            raise ValueError(f"--loss_severity_weights unknown key: {key}")
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
        raise KeyError(f"base_cfg not found: {dataset_name}")
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
        / "expB_loss_adapt"
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
    p = argparse.ArgumentParser(description="ExpB: adaptive lambda/tau runner")
    p.add_argument("--dataset_type", type=str, required=True, choices=["sea", "sine", "stagger"])
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--labeled_ratio", type=float, required=True)
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--log_root_suffix", type=str, default="expB")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--monitor_preset", type=str, default="entropy_divergence_ph_meta")
    p.add_argument("--signal_set", type=str, default="proxy", choices=["error", "proxy", "all"])
    p.add_argument("--variants", type=str, default="B0,B1,B2,P1,P2,P1L,P1T")
    p.add_argument("--lambda_min_scale", type=float, default=0.5)
    p.add_argument("--lambda_max_scale", type=float, default=1.5)
    p.add_argument("--tau_delta", type=float, default=0.05)
    p.add_argument("--loss_severity_mode", type=str, default="max", choices=["max", "weighted"])
    p.add_argument("--loss_severity_weights", type=str, default="")
    p.add_argument("--loss_severity_momentum", type=float, default=0.99)
    p.add_argument("--loss_severity_smoothing", type=float, default=0.9)
    p.add_argument("--loss_severity_low", type=float, default=0.0)
    p.add_argument("--loss_severity_high", type=float, default=2.0)
    p.add_argument("--loss_severity_use_error", action="store_true")
    p.add_argument("--loss_event_on", type=float, default=0.6)
    p.add_argument("--loss_event_off", type=float, default=0.4)
    p.add_argument("--loss_cooldown_steps", type=int, default=200)
    p.add_argument("--loss_use_candidate", action="store_true")
    p.add_argument("--loss_use_drift_flag", action="store_true")
    p.add_argument("--safety_tau", type=float, default=0.95)
    p.add_argument("--safety_lambda", type=float, default=0.5)
    p.add_argument("--safety_cooldown_steps", type=int, default=200)
    p.add_argument("--safety_use_severity", action="store_true")
    p.add_argument("--safety_severity_threshold", type=float, default=0.6)
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
        raise ValueError("--seeds must not be empty")
    labeled_ratio = float(args.labeled_ratio)
    ratio_tag = format_ratio_tag(labeled_ratio)
    variants = [v.strip().upper() for v in str(args.variants).split(",") if v.strip()]
    if not variants:
        raise ValueError("--variants must not be empty")

    run_tag = str(args.run_tag or "").strip()
    if not run_tag:
        run_tag = f"loss_s{float(args.loss_severity_smoothing):.2f}"
        run_tag = run_tag.replace(".", "p")

    cfg_map = build_cfg_map(int(seeds[0]))
    base_cfg = resolve_base_cfg(str(args.dataset_name), cfg_map)

    base_lambda = float(base_cfg.lambda_u)
    base_tau = float(base_cfg.tau)
    lambda_min = base_lambda * float(args.lambda_min_scale)
    lambda_max = base_lambda * float(args.lambda_max_scale)
    tau_min = max(0.0, min(1.0, base_tau - float(args.tau_delta)))
    tau_max = max(0.0, min(1.0, base_tau + float(args.tau_delta)))
    if lambda_min > lambda_max:
        lambda_min, lambda_max = lambda_max, lambda_min
    if tau_min > tau_max:
        tau_min, tau_max = tau_max, tau_min

    weights = parse_weights(str(args.loss_severity_weights))

    common_loss_cfg = {
        "loss_severity_mode": str(args.loss_severity_mode),
        "loss_severity_weights": weights,
        "loss_severity_momentum": float(args.loss_severity_momentum),
        "loss_severity_smoothing": float(args.loss_severity_smoothing),
        "loss_severity_low": float(args.loss_severity_low),
        "loss_severity_high": float(args.loss_severity_high),
        "loss_severity_use_error": bool(args.loss_severity_use_error),
        "loss_event_on": float(args.loss_event_on),
        "loss_event_off": float(args.loss_event_off),
        "loss_cooldown_steps": int(args.loss_cooldown_steps),
        "loss_use_candidate": bool(args.loss_use_candidate),
        "loss_use_drift_flag": bool(args.loss_use_drift_flag),
    }

    ema_fixed_cfg = {
        "ema_decay_mode": "fixed",
        "ema_gamma_fixed": float(base_cfg.initial_alpha),
    }

    preset_map: Dict[str, Dict[str, Any]] = {
        "B0": {
            "model_variant": "loss_fixed_default",
            "loss_scheduler_mode": "fixed",
            "loss_lambda_fixed": float(base_lambda),
            "loss_tau_fixed": float(base_tau),
            "loss_apply_lambda": True,
            "loss_apply_tau": True,
        },
        "B1": {
            "model_variant": "loss_fixed_conservative",
            "loss_scheduler_mode": "fixed",
            "loss_lambda_fixed": float(lambda_min),
            "loss_tau_fixed": float(tau_max),
            "loss_apply_lambda": True,
            "loss_apply_tau": True,
        },
        "B2": {
            "model_variant": "loss_fixed_aggressive",
            "loss_scheduler_mode": "fixed",
            "loss_lambda_fixed": float(lambda_max),
            "loss_tau_fixed": float(tau_min),
            "loss_apply_lambda": True,
            "loss_apply_tau": True,
        },
        "P1": {
            "model_variant": "loss_adapt_cont",
            "loss_scheduler_mode": "severity_continuous",
            "loss_lambda_min": float(lambda_min),
            "loss_lambda_max": float(lambda_max),
            "loss_tau_min": float(tau_min),
            "loss_tau_max": float(tau_max),
            "loss_apply_lambda": True,
            "loss_apply_tau": True,
        },
        "P2": {
            "model_variant": "loss_adapt_event",
            "loss_scheduler_mode": "severity_event",
            "loss_lambda_min": float(lambda_min),
            "loss_lambda_max": float(lambda_max),
            "loss_tau_min": float(tau_min),
            "loss_tau_max": float(tau_max),
            "loss_apply_lambda": True,
            "loss_apply_tau": True,
        },
        "P1L": {
            "model_variant": "loss_adapt_cont_lambda",
            "loss_scheduler_mode": "severity_continuous",
            "loss_lambda_min": float(lambda_min),
            "loss_lambda_max": float(lambda_max),
            "loss_tau_fixed": float(base_tau),
            "loss_apply_lambda": True,
            "loss_apply_tau": False,
        },
        "P1T": {
            "model_variant": "loss_adapt_cont_tau",
            "loss_scheduler_mode": "severity_continuous",
            "loss_tau_min": float(tau_min),
            "loss_tau_max": float(tau_max),
            "loss_lambda_fixed": float(base_lambda),
            "loss_apply_lambda": False,
            "loss_apply_tau": True,
        },
        "P1T_SV": {
            "model_variant": "loss_adapt_cont_tau_sv",
            "loss_scheduler_mode": "severity_continuous",
            "loss_tau_min": float(tau_min),
            "loss_tau_max": float(tau_max),
            "loss_lambda_fixed": float(base_lambda),
            "loss_apply_lambda": False,
            "loss_apply_tau": True,
            "loss_safety_enabled": True,
            "loss_safety_use_candidate": True,
            "loss_safety_use_severity": bool(args.safety_use_severity),
            "loss_safety_severity_threshold": float(args.safety_severity_threshold),
            "loss_safety_cooldown_steps": int(args.safety_cooldown_steps),
            "loss_safety_tau": float(max(base_tau, float(args.safety_tau))),
            "loss_safety_lambda": float(min(base_lambda, float(args.safety_lambda))),
        },
    }

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        if variant not in preset_map:
            raise ValueError(f"unknown variant: {variant}")
        vcfg = {**preset_map[variant], **common_loss_cfg, **ema_fixed_cfg}
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
                    "lambda_base": float(base_lambda),
                    "tau_base": float(base_tau),
                    "lambda_min": float(lambda_min),
                    "lambda_max": float(lambda_max),
                    "tau_min": float(tau_min),
                    "tau_max": float(tau_max),
                    "loss_severity_mode": str(args.loss_severity_mode),
                    "loss_severity_weights": str(args.loss_severity_weights or ""),
                    "loss_severity_smoothing": float(args.loss_severity_smoothing),
                    "loss_event_on": float(args.loss_event_on),
                    "loss_event_off": float(args.loss_event_off),
                    "loss_cooldown_steps": int(args.loss_cooldown_steps),
                    "loss_use_candidate": int(bool(args.loss_use_candidate)),
                    "loss_use_drift_flag": int(bool(args.loss_use_drift_flag)),
                    "safety_tau": float(args.safety_tau),
                    "safety_lambda": float(args.safety_lambda),
                    "safety_cooldown_steps": int(args.safety_cooldown_steps),
                    "safety_use_severity": int(bool(args.safety_use_severity)),
                    "safety_severity_threshold": float(args.safety_severity_threshold),
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
