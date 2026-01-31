"""Step-1 实验脚本，实现多数据集批量运行与统一日志。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
from river import metrics

from data import streams
from drift import detectors
from models.teacher_student import TeacherStudentModel
from scheduler.hparam_scheduler import HParams, SchedulerState
from soft_drift.utils.run_paths import create_experiment_run
from training.loop import FeatureVectorizer, LabelEncoder, TrainingConfig, run_training_loop


def _use_severity_scheduler(model_variant: str) -> bool:
    return "_severity" in model_variant


def _format_trigger_weights(weights: Optional[Dict[str, Any]]) -> str:
    if not weights:
        return ""
    parts: List[str] = []
    for k in sorted(weights.keys()):
        v = weights.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{k}={float(v):.6g}")
        else:
            parts.append(f"{k}={v}")
    return ",".join(parts)


_SIGNAL_SET_PRESET_MAP: Dict[str, str] = {
    "error": "error_ph_meta",
    "proxy": "entropy_divergence_ph_meta",
    "all": "all_signals_ph_meta",
}


def _normalize_signal_set(signal_set: Optional[str]) -> Optional[str]:
    if signal_set is None:
        return None
    value = str(signal_set).strip().lower()
    if not value:
        return None
    return value if value in _SIGNAL_SET_PRESET_MAP else None


def _apply_signal_set_to_preset(monitor_preset: str, signal_set: Optional[str]) -> str:
    normalized = _normalize_signal_set(signal_set)
    if not normalized:
        return monitor_preset
    preset = str(monitor_preset or "").strip()
    spec = ""
    if "@" in preset:
        _, spec = preset.split("@", 1)
        spec = spec.strip()
    target = _SIGNAL_SET_PRESET_MAP[normalized]
    return f"{target}@{spec}" if spec else target


@dataclass
class ExperimentConfig:
    """单次实验的高层配置。"""

    dataset_type: str
    dataset_name: str
    csv_path: Optional[str] = None
    label_col: Optional[str] = None
    model_variant: str = "ts_drift_adapt"
    n_steps: int = 500
    batch_size: int = 64
    labeled_ratio: float = 0.1
    initial_alpha: float = 0.99
    initial_lr: float = 1e-3
    lambda_u: float = 1.0
    tau: float = 0.9
    seed: int = 42
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    stream_kwargs: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[str] = None
    monitor_preset: str = "none"
    signal_set: Optional[str] = None
    trigger_mode: str = "or"
    trigger_k: int = 2
    trigger_threshold: float = 0.5
    trigger_weights: Optional[Dict[str, Any]] = None
    confirm_window: int = 200
    severity_scheduler_scale: float = 1.0
    use_severity_v2: bool = False
    severity_gate: str = "none"
    severity_gate_min_streak: int = 1
    entropy_mode: str = "overconfident"
    severity_decay: float = 0.95
    freeze_baseline_steps: int = 0
    ema_decay_mode: str = "none"
    ema_gamma_min: float = 0.95
    ema_gamma_max: float = 0.999
    ema_gamma_fixed: Optional[float] = None
    ema_severity_mode: str = "max"
    ema_severity_weights: Optional[Tuple[float, float, float]] = None
    ema_severity_smoothing: float = 0.9
    ema_severity_threshold: float = 0.6
    ema_severity_threshold_off: Optional[float] = None
    ema_cooldown_steps: int = 200
    ema_use_candidate: bool = False
    ema_use_drift_flag: bool = False
    loss_scheduler_mode: str = "none"
    loss_lambda_min: Optional[float] = None
    loss_lambda_max: Optional[float] = None
    loss_tau_min: Optional[float] = None
    loss_tau_max: Optional[float] = None
    loss_lambda_fixed: Optional[float] = None
    loss_tau_fixed: Optional[float] = None
    loss_severity_mode: str = "max"
    loss_severity_weights: Optional[Tuple[float, float, float]] = None
    loss_severity_momentum: float = 0.99
    loss_severity_smoothing: float = 0.9
    loss_severity_low: float = 0.0
    loss_severity_high: float = 2.0
    loss_severity_eps: float = 1e-6
    loss_severity_use_error: bool = False
    loss_event_on: float = 0.6
    loss_event_off: float = 0.4
    loss_cooldown_steps: int = 200
    loss_use_candidate: bool = False
    loss_use_drift_flag: bool = False
    loss_apply_lambda: bool = True
    loss_apply_tau: bool = True
    loss_safety_enabled: bool = False
    loss_safety_use_candidate: bool = True
    loss_safety_use_severity: bool = False
    loss_safety_severity_threshold: float = 0.6
    loss_safety_cooldown_steps: int = 200
    loss_safety_tau: Optional[float] = None
    loss_safety_lambda: Optional[float] = None


def run_experiment(config: ExperimentConfig, device: str = "cpu") -> pd.DataFrame:
    """
    构建数据流、模型与监控组件，运行在线训练，并返回日志 DataFrame。
    同时依据 config.log_path 写入 CSV（若提供）。
    """
    signal_set = _normalize_signal_set(getattr(config, "signal_set", None))
    monitor_preset = _apply_signal_set_to_preset(config.monitor_preset, signal_set)
    if monitor_preset != config.monitor_preset or signal_set != getattr(config, "signal_set", None):
        config = replace(config, monitor_preset=monitor_preset, signal_set=signal_set)

    stream_info = streams.build_stream(
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        csv_path=config.csv_path,
        label_col=config.label_col,
        seed=config.seed,
        **config.stream_kwargs,
    )
    if stream_info.n_classes <= 0:
        raise ValueError("需要提供有效的类别数以构建模型。")
    vectorizer = FeatureVectorizer(
        input_dim=stream_info.n_features,
        feature_order=stream_info.feature_names,
    )
    label_encoder = LabelEncoder(classes=stream_info.classes)
    model = TeacherStudentModel(
        input_dim=stream_info.n_features,
        num_classes=stream_info.n_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        device=torch.device(device),
    )
    optimizer = torch.optim.Adam(model.student.parameters(), lr=config.initial_lr)
    monitor = detectors.build_default_monitor(
        preset=config.monitor_preset,
        trigger_mode=config.trigger_mode,
        trigger_k=config.trigger_k,
        trigger_weights=config.trigger_weights,
        trigger_threshold=config.trigger_threshold,
        confirm_window=config.confirm_window,
    )
    initial_hparams = HParams(
        alpha=config.initial_alpha,
        lr=config.initial_lr,
        lambda_u=config.lambda_u,
        tau=config.tau,
    )
    scheduler_state = SchedulerState(base_hparams=initial_hparams)
    metric = metrics.Accuracy()
    batch_iter = streams.batch_stream(
        stream=stream_info,
        batch_size=config.batch_size,
        labeled_ratio=config.labeled_ratio,
        seed=config.seed,
    )
    training_config = TrainingConfig(
        n_steps=config.n_steps,
        device=device,
        log_path=config.log_path,
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        model_variant=config.model_variant,
        seed=config.seed,
        monitor_preset=config.monitor_preset,
        signal_set=config.signal_set,
        trigger_mode=config.trigger_mode,
        trigger_k=config.trigger_k,
        trigger_threshold=config.trigger_threshold,
        trigger_weights=_format_trigger_weights(config.trigger_weights),
        confirm_window=int(config.confirm_window),
        use_severity_scheduler=_use_severity_scheduler(config.model_variant),
        use_severity_v2=bool(config.use_severity_v2),
        severity_gate=str(config.severity_gate),
        severity_gate_min_streak=int(getattr(config, "severity_gate_min_streak", 1) or 1),
        entropy_mode=str(config.entropy_mode),
        severity_decay=float(config.severity_decay),
        freeze_baseline_steps=int(config.freeze_baseline_steps),
        severity_scheduler_scale=config.severity_scheduler_scale,
        ema_decay_mode=str(config.ema_decay_mode),
        ema_gamma_min=float(config.ema_gamma_min),
        ema_gamma_max=float(config.ema_gamma_max),
        ema_gamma_fixed=float(config.ema_gamma_fixed) if config.ema_gamma_fixed is not None else None,
        ema_severity_mode=str(config.ema_severity_mode),
        ema_severity_weights=config.ema_severity_weights,
        ema_severity_smoothing=float(config.ema_severity_smoothing),
        ema_severity_threshold=float(config.ema_severity_threshold),
        ema_severity_threshold_off=float(config.ema_severity_threshold_off) if config.ema_severity_threshold_off is not None else None,
        ema_cooldown_steps=int(config.ema_cooldown_steps),
        ema_use_candidate=bool(config.ema_use_candidate),
        ema_use_drift_flag=bool(config.ema_use_drift_flag),
        loss_scheduler_mode=str(config.loss_scheduler_mode),
        loss_lambda_min=config.loss_lambda_min,
        loss_lambda_max=config.loss_lambda_max,
        loss_tau_min=config.loss_tau_min,
        loss_tau_max=config.loss_tau_max,
        loss_lambda_fixed=config.loss_lambda_fixed,
        loss_tau_fixed=config.loss_tau_fixed,
        loss_severity_mode=str(config.loss_severity_mode),
        loss_severity_weights=config.loss_severity_weights,
        loss_severity_momentum=float(config.loss_severity_momentum),
        loss_severity_smoothing=float(config.loss_severity_smoothing),
        loss_severity_low=float(config.loss_severity_low),
        loss_severity_high=float(config.loss_severity_high),
        loss_severity_eps=float(config.loss_severity_eps),
        loss_severity_use_error=bool(config.loss_severity_use_error),
        loss_event_on=float(config.loss_event_on),
        loss_event_off=float(config.loss_event_off),
        loss_cooldown_steps=int(config.loss_cooldown_steps),
        loss_use_candidate=bool(config.loss_use_candidate),
        loss_use_drift_flag=bool(config.loss_use_drift_flag),
        loss_apply_lambda=bool(config.loss_apply_lambda),
        loss_apply_tau=bool(config.loss_apply_tau),
        loss_safety_enabled=bool(config.loss_safety_enabled),
        loss_safety_use_candidate=bool(config.loss_safety_use_candidate),
        loss_safety_use_severity=bool(config.loss_safety_use_severity),
        loss_safety_severity_threshold=float(config.loss_safety_severity_threshold),
        loss_safety_cooldown_steps=int(config.loss_safety_cooldown_steps),
        loss_safety_tau=config.loss_safety_tau,
        loss_safety_lambda=config.loss_safety_lambda,
    )
    logs_df = run_training_loop(
        batch_iter=batch_iter,
        model=model,
        optimizer=optimizer,
        drift_monitor=monitor,
        scheduler_state=scheduler_state,
        metric=metric,
        initial_hparams=initial_hparams,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        config=training_config,
    )
    return logs_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1 师生 EMA 实验批处理")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        type=str,
        default="sea_abrupt4,hyperplane_slow,electricity,airlines,insects_abrupt_balanced",
        help="逗号分隔的数据集名称列表，或 all",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ts_drift_adapt",
        help="逗号分隔的模型变体列表，或 all",
    )
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="none",
        choices=[
            "none",
            "error_ph_meta",
            "error_only_ph_meta",
            "entropy_only_ph_meta",
            "divergence_ph_meta",
            "divergence_only_ph_meta",
            "error_entropy_ph_meta",
            "error_divergence_ph_meta",
            "entropy_divergence_ph_meta",
            "all_signals_ph_meta",
        ],
        help="漂移检测器预设（none 表示不启用）",
    )
    parser.add_argument(
        "--signal_set",
        type=str,
        default=None,
        choices=["error", "proxy", "all"],
        help="信号组合（可选）：error=仅监督，proxy=entropy+divergence，all=三者；为空则沿用 monitor_preset",
    )
    parser.add_argument("--results_root", type=str, default="results", help="结果输出根目录")
    parser.add_argument("--logs_root", type=str, default="logs", help="日志输出根目录")
    parser.add_argument("--run_name", type=str, default=None, help="附加到 run_id 的别名")
    parser.add_argument("--run_id", type=str, default=None, help="自定义 run_id，谨慎使用")
    args = parser.parse_args()

    experiment_run = create_experiment_run(
        experiment_name="first_stage_experiments",
        results_root=args.results_root,
        logs_root=args.logs_root,
        run_name=args.run_name,
        run_id=args.run_id,
    )
    print(f"[run] first_stage_experiments run_id={experiment_run.run_id}")

    dataset_filters = _parse_list(args.datasets)
    model_filters = _parse_list(args.models)

    base_configs = _default_experiment_configs(args.seed)
    selected_datasets = [
        cfg for cfg in base_configs if not dataset_filters or cfg.dataset_name in dataset_filters
    ]
    if not selected_datasets:
        raise ValueError("未匹配到任何数据集配置")

    available_models = ["baseline_student", "mean_teacher", "ts_drift_adapt", "ts_drift_adapt_severity"]
    model_list = model_filters or available_models

    for base_cfg in selected_datasets:
        for model_variant in model_list:
            run_paths = experiment_run.prepare_dataset_run(base_cfg.dataset_name, model_variant, args.seed)
            log_path = run_paths.log_csv_path()
            cfg = replace(
                base_cfg,
                model_variant=model_variant,
                seed=args.seed,
                log_path=str(log_path),
                monitor_preset=args.monitor_preset,
                signal_set=args.signal_set,
            )
            logs = run_experiment(cfg, device=args.device)
            final_acc = logs["metric_accuracy"].iloc[-1] if not logs.empty else float("nan")
            run_paths.update_legacy_pointer()
            print(
                f"[{cfg.dataset_name}][{cfg.model_variant}][seed={cfg.seed}] final accuracy = {final_acc:.4f}"
            )


def _default_log_path(dataset_name: str, model_variant: str, seed: int) -> str:
    log_dir = os.path.join("logs", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{dataset_name}__{model_variant}__seed{seed}.csv"
    return os.path.join(log_dir, filename)


def _parse_list(spec: str) -> Optional[List[str]]:
    if not spec or spec.lower() == "all":
        return None
    return [item.strip() for item in spec.split(",") if item.strip()]


def _default_experiment_configs(seed: int) -> List[ExperimentConfig]:
    """构造 Stage-1 默认实验集合。"""
    return [
        ExperimentConfig(
            dataset_type="sea",
            dataset_name="sea_abrupt4",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.99,
            initial_lr=1e-3,
            lambda_u=1.0,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="sine",
            dataset_name="sine_abrupt4",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.99,
            initial_lr=1e-3,
            lambda_u=1.0,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="stagger",
            dataset_name="stagger_abrupt3",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.99,
            initial_lr=1e-3,
            lambda_u=1.0,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="hyperplane",
            dataset_name="hyperplane_slow",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.98,
            initial_lr=5e-4,
            lambda_u=0.8,
            tau=0.85,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="uspds_csv",
            dataset_name="electricity",
            csv_path="data/uspds/electricity.csv",
            label_col=None,
            n_steps=1000,
            batch_size=256,
            labeled_ratio=0.05,
            initial_alpha=0.995,
            initial_lr=5e-4,
            lambda_u=0.7,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="uspds_csv",
            dataset_name="airlines",
            csv_path="data/uspds/airlines.csv",
            label_col=None,
            n_steps=1000,
            batch_size=256,
            labeled_ratio=0.05,
            initial_alpha=0.995,
            initial_lr=5e-4,
            lambda_u=0.7,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="insects_real",
            dataset_name="INSECTS_abrupt_balanced",
            csv_path="datasets/real/INSECTS_abrupt_balanced.csv",
            n_steps=1200,
            batch_size=256,
            labeled_ratio=0.05,
            initial_alpha=0.97,
            initial_lr=1e-3,
            lambda_u=0.5,
            tau=0.8,
            seed=seed,
        ),
    ]


if __name__ == "__main__":
    main()
