"""在线训练循环实现。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from river import metrics
from torch import Tensor

from drift.detectors import DriftMonitor
from drift.signals import compute_signals
from models.teacher_student import LossOutputs, TeacherStudentModel
from scheduler.hparam_scheduler import (
    HParams,
    SchedulerState,
    SeveritySchedulerConfig,
    update_hparams,
    update_hparams_with_severity,
)
from scheduler.ema_adapt import EmaDecayConfig, EmaDecayScheduler
from scheduler.loss_adapt import LossAdaptiveConfig, LossAdaptiveScheduler
from scheduler.step_hook import StepScheduler
from soft_drift.severity import SeverityCalibrator


@dataclass
class TrainingConfig:
    """训练循环配置。"""

    n_steps: int
    device: str = "cpu"
    log_path: Optional[str] = None
    dataset_type: str = "unknown"
    dataset_name: str = "unknown"
    model_variant: str = "ts_drift_adapt"
    seed: int = 0
    monitor_preset: str = "none"
    signal_set: Optional[str] = None
    trigger_mode: str = "or"
    trigger_k: int = 2
    trigger_threshold: float = 0.5
    trigger_weights: str = ""
    confirm_window: int = 200
    use_severity_scheduler: bool = False
    use_severity_v2: bool = False
    severity_gate: str = "none"
    severity_gate_min_streak: int = 1
    entropy_mode: str = "overconfident"
    severity_decay: float = 0.95
    freeze_baseline_steps: int = 0
    severity_ema_momentum: float = 0.99
    severity_eps: float = 1e-6
    severity_norm_low: float = 0.0
    severity_norm_high: float = 2.0
    severity_scheduler_scale: float = 1.0
    severity_scheduler_config: Optional[SeveritySchedulerConfig] = None
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


class FeatureVectorizer:
    """将 dict 或数组样本转为统一向量。"""

    def __init__(self, input_dim: int, feature_order: Optional[Sequence[str]] = None) -> None:
        self.input_dim = input_dim
        self.feature_order: Optional[List[str]] = list(feature_order) if feature_order else None

    def transform(self, sample: Any) -> np.ndarray:
        if isinstance(sample, np.ndarray):
            return sample.astype(np.float32)
        if isinstance(sample, dict):
            if self.feature_order is None:
                self.feature_order = list(sorted(sample.keys()))
            vec = np.array([float(sample.get(k, 0.0)) for k in self.feature_order], dtype=np.float32)
            if len(vec) != self.input_dim:
                raise ValueError("特征维度与模型输入不一致")
            return vec
        if isinstance(sample, (list, tuple)):
            return np.asarray(sample, dtype=np.float32)
        raise TypeError(f"不支持的样本类型: {type(sample)}")

    def transform_many(self, samples: Sequence[Any]) -> Optional[np.ndarray]:
        if not samples:
            return None
        return np.stack([self.transform(s) for s in samples], axis=0)


class LabelEncoder:
    """标签编码器。"""

    def __init__(self, classes: Optional[Sequence[Any]] = None) -> None:
        self.class_to_idx: Dict[Any, int] = {}
        self.idx_to_class: List[Any] = []
        if classes:
            for c in classes:
                self._add_class(c)

    def _add_class(self, label: Any) -> int:
        if label not in self.class_to_idx:
            idx = len(self.idx_to_class)
            self.class_to_idx[label] = idx
            self.idx_to_class.append(label)
        return self.class_to_idx[label]

    def encode(self, label: Any) -> int:
        return self._add_class(label)

    def encode_many(self, labels: Sequence[Any]) -> np.ndarray:
        return np.asarray([self.encode(lbl) for lbl in labels], dtype=np.int64)

    def decode(self, idx: int) -> Any:
        return self.idx_to_class[idx]

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_class)


def run_training_loop(
    batch_iter: Iterator[Tuple[List[Any], List[Any], List[Any], List[Any]]],
    model: TeacherStudentModel,
    optimizer: torch.optim.Optimizer,
    drift_monitor: DriftMonitor,
    scheduler_state: SchedulerState,
    metric: metrics.base.Metric,
    initial_hparams: HParams,
    vectorizer: FeatureVectorizer,
    label_encoder: LabelEncoder,
    config: TrainingConfig,
) -> pd.DataFrame:
    """运行在线训练，返回逐批日志 DataFrame。"""
    device = torch.device(config.device)
    model.to(device)
    current_hparams = initial_hparams
    kappa_metric = metrics.CohenKappa()
    logs: List[Dict[str, Any]] = []
    seen_samples = 0
    sample_idx = -1
    candidate_sample_idxs: List[int] = []
    confirmed_sample_idxs: List[int] = []
    acc_series: List[Tuple[int, float]] = []
    monitor_preset_base = str(getattr(drift_monitor, "preset_base", config.monitor_preset) or "")
    monitor_ph_params = getattr(drift_monitor, "ph_params", None) or {}
    monitor_ph_overrides = getattr(drift_monitor, "ph_overrides", None) or {}
    # confirm_cooldown 通过 trigger_weights 透传（避免改动 ExperimentConfig）；单位为 sample_idx。
    confirm_cooldown = 0
    trigger_weights_obj = getattr(config, "trigger_weights", None)
    if isinstance(trigger_weights_obj, dict):
        for key in ("confirm_cooldown", "__confirm_cooldown", "cooldown"):
            if key not in trigger_weights_obj:
                continue
            try:
                confirm_cooldown = max(0, int(float(trigger_weights_obj[key])))  # type: ignore[arg-type]
                break
            except Exception:
                continue
    setattr(drift_monitor, "confirm_cooldown", int(confirm_cooldown))
    # 自适应 cooldown 参数同样通过 trigger_weights 透传；这里只做日志记录/显式注入（便于复现）。
    adaptive_enabled = False
    adaptive_window = 10000
    adaptive_lower_per10k = 10.0
    adaptive_upper_per10k = 25.0
    adaptive_cooldown_low = 200
    adaptive_cooldown_high = 500
    if isinstance(trigger_weights_obj, dict):
        def _get(keys: tuple[str, ...]) -> Optional[float]:
            for k in keys:
                if k not in trigger_weights_obj:
                    continue
                try:
                    return float(trigger_weights_obj[k])  # type: ignore[arg-type]
                except Exception:
                    return None
            return None

        v = _get(("adaptive_cooldown", "adaptive_cd", "adaptive"))
        if v is not None:
            adaptive_enabled = bool(int(v))
        v = _get(("adaptive_window", "adaptive_win"))
        if v is not None:
            adaptive_window = max(1, int(v))
        v = _get(("adaptive_lower_per10k", "adaptive_lower"))
        if v is not None:
            adaptive_lower_per10k = max(0.0, float(v))
        v = _get(("adaptive_upper_per10k", "adaptive_upper"))
        if v is not None:
            adaptive_upper_per10k = max(adaptive_lower_per10k, float(v))
        v = _get(("adaptive_cooldown_low", "adaptive_low"))
        if v is not None:
            adaptive_cooldown_low = max(0, int(v))
        v = _get(("adaptive_cooldown_high", "adaptive_high"))
        if v is not None:
            adaptive_cooldown_high = max(0, int(v))

    setattr(drift_monitor, "adaptive_cooldown_enabled", bool(adaptive_enabled))
    setattr(drift_monitor, "adaptive_window", int(adaptive_window))
    setattr(drift_monitor, "adaptive_lower_per10k", float(adaptive_lower_per10k))
    setattr(drift_monitor, "adaptive_upper_per10k", float(adaptive_upper_per10k))
    setattr(drift_monitor, "adaptive_cooldown_low", int(adaptive_cooldown_low))
    setattr(drift_monitor, "adaptive_cooldown_high", int(adaptive_cooldown_high))
    try:
        monitor_ph_params_json = json.dumps(monitor_ph_params, ensure_ascii=False, sort_keys=True)
        monitor_ph_overrides_json = json.dumps(monitor_ph_overrides, ensure_ascii=False, sort_keys=True)
    except Exception:
        monitor_ph_params_json = ""
        monitor_ph_overrides_json = ""
    ph_error = monitor_ph_params.get("error_rate", {}) if isinstance(monitor_ph_params, dict) else {}
    ph_div = monitor_ph_params.get("divergence", {}) if isinstance(monitor_ph_params, dict) else {}
    ph_ent = monitor_ph_params.get("teacher_entropy", {}) if isinstance(monitor_ph_params, dict) else {}
    ph_error_threshold = float(ph_error.get("threshold")) if isinstance(ph_error, dict) and ph_error.get("threshold") is not None else float("nan")
    ph_error_delta = float(ph_error.get("delta")) if isinstance(ph_error, dict) and ph_error.get("delta") is not None else float("nan")
    ph_error_alpha = float(ph_error.get("alpha")) if isinstance(ph_error, dict) and ph_error.get("alpha") is not None else float("nan")
    ph_error_min_instances = int(float(ph_error.get("min_instances"))) if isinstance(ph_error, dict) and ph_error.get("min_instances") is not None else -1
    ph_div_threshold = float(ph_div.get("threshold")) if isinstance(ph_div, dict) and ph_div.get("threshold") is not None else float("nan")
    ph_div_delta = float(ph_div.get("delta")) if isinstance(ph_div, dict) and ph_div.get("delta") is not None else float("nan")
    ph_div_alpha = float(ph_div.get("alpha")) if isinstance(ph_div, dict) and ph_div.get("alpha") is not None else float("nan")
    ph_div_min_instances = int(float(ph_div.get("min_instances"))) if isinstance(ph_div, dict) and ph_div.get("min_instances") is not None else -1
    ph_ent_threshold = float(ph_ent.get("threshold")) if isinstance(ph_ent, dict) and ph_ent.get("threshold") is not None else float("nan")
    ph_ent_delta = float(ph_ent.get("delta")) if isinstance(ph_ent, dict) and ph_ent.get("delta") is not None else float("nan")
    ph_ent_alpha = float(ph_ent.get("alpha")) if isinstance(ph_ent, dict) and ph_ent.get("alpha") is not None else float("nan")
    ph_ent_min_instances = int(float(ph_ent.get("min_instances"))) if isinstance(ph_ent, dict) and ph_ent.get("min_instances") is not None else -1
    severity_calibrator: Optional[SeverityCalibrator] = None
    severity_carry = 0.0
    baseline_freeze_remaining = 0
    severity_confirmed_streak = 0
    severity_scheduler_cfg = config.severity_scheduler_config or SeveritySchedulerConfig()
    severity_scheduler_cfg = replace(
        severity_scheduler_cfg,
        severity_scale=max(0.0, config.severity_scheduler_scale),
    )
    if config.use_severity_scheduler:
        severity_calibrator = SeverityCalibrator(
            ema_momentum=config.severity_ema_momentum,
            eps=config.severity_eps,
            severity_low=config.severity_norm_low,
            severity_high=config.severity_norm_high,
            entropy_mode=config.entropy_mode,
        )
    ema_decay_mode = str(getattr(config, "ema_decay_mode", "none") or "none").strip().lower()
    if ema_decay_mode not in {"none", "fixed", "severity_continuous", "severity_event", "severity_event_reverse"}:
        ema_decay_mode = "none"
    ema_gamma_min = float(getattr(config, "ema_gamma_min", 0.95) or 0.95)
    ema_gamma_max = float(getattr(config, "ema_gamma_max", 0.999) or 0.999)
    ema_gamma_fixed = getattr(config, "ema_gamma_fixed", None)
    def _clamp_alpha(value: float) -> float:
        if math.isnan(value):
            return 0.0
        return min(0.999999, max(0.0, float(value)))
    ema_gamma_min = _clamp_alpha(ema_gamma_min)
    ema_gamma_max = _clamp_alpha(ema_gamma_max)
    if ema_gamma_min > ema_gamma_max:
        ema_gamma_min, ema_gamma_max = ema_gamma_max, ema_gamma_min
    if ema_gamma_fixed is None and ema_decay_mode == "fixed":
        ema_gamma_fixed = float(ema_gamma_max)
    ema_severity_mode = str(getattr(config, "ema_severity_mode", "max") or "max").strip().lower()
    ema_severity_weights = getattr(config, "ema_severity_weights", None)
    if ema_severity_weights is None:
        ema_severity_weights = (0.0, 0.5, 0.5)
    ema_severity_smoothing = float(getattr(config, "ema_severity_smoothing", 0.9) or 0.9)
    if not (0.0 <= ema_severity_smoothing <= 1.0):
        ema_severity_smoothing = min(1.0, max(0.0, ema_severity_smoothing))
    ema_severity_threshold = float(getattr(config, "ema_severity_threshold", 0.6) or 0.6)
    ema_severity_threshold_off = getattr(config, "ema_severity_threshold_off", None)
    ema_cooldown_steps = max(0, int(getattr(config, "ema_cooldown_steps", 200) or 0))
    ema_use_candidate = bool(getattr(config, "ema_use_candidate", False))
    ema_use_drift_flag = bool(getattr(config, "ema_use_drift_flag", False))
    ema_cfg = EmaDecayConfig(
        mode=str(ema_decay_mode),
        gamma_min=float(ema_gamma_min),
        gamma_max=float(ema_gamma_max),
        gamma_fixed=float(ema_gamma_fixed) if ema_gamma_fixed is not None else None,
        severity_mode=str(ema_severity_mode),
        severity_weights=tuple(float(x) for x in ema_severity_weights),
        severity_smoothing=float(ema_severity_smoothing),
        severity_threshold=float(ema_severity_threshold),
        severity_threshold_off=float(ema_severity_threshold_off) if ema_severity_threshold_off is not None else None,
        cooldown_steps=int(ema_cooldown_steps),
        use_candidate=bool(ema_use_candidate),
        use_drift_flag=bool(ema_use_drift_flag),
        ema_momentum=float(config.severity_ema_momentum),
        eps=float(config.severity_eps),
        severity_low=float(config.severity_norm_low),
        severity_high=float(config.severity_norm_high),
        entropy_mode=str(config.entropy_mode),
    )
    ema_scheduler = EmaDecayScheduler(ema_cfg, default_decay=float(initial_hparams.alpha))

    loss_scheduler_mode = str(getattr(config, "loss_scheduler_mode", "none") or "none").strip().lower()
    loss_severity_mode = str(getattr(config, "loss_severity_mode", "max") or "max").strip().lower()
    loss_severity_weights = getattr(config, "loss_severity_weights", None)
    if loss_severity_weights is None:
        loss_severity_weights = (0.0, 0.5, 0.5)
    loss_cfg = LossAdaptiveConfig(
        mode=str(loss_scheduler_mode),
        lambda_min=getattr(config, "loss_lambda_min", None),
        lambda_max=getattr(config, "loss_lambda_max", None),
        tau_min=getattr(config, "loss_tau_min", None),
        tau_max=getattr(config, "loss_tau_max", None),
        lambda_fixed=getattr(config, "loss_lambda_fixed", None),
        tau_fixed=getattr(config, "loss_tau_fixed", None),
        severity_mode=str(loss_severity_mode),
        severity_weights=tuple(float(x) for x in loss_severity_weights),
        severity_momentum=float(getattr(config, "loss_severity_momentum", 0.99) or 0.99),
        severity_smoothing=float(getattr(config, "loss_severity_smoothing", 0.9) or 0.9),
        severity_low=float(getattr(config, "loss_severity_low", 0.0) or 0.0),
        severity_high=float(getattr(config, "loss_severity_high", 2.0) or 2.0),
        severity_eps=float(getattr(config, "loss_severity_eps", 1e-6) or 1e-6),
        severity_use_error=bool(getattr(config, "loss_severity_use_error", False)),
        event_threshold_on=float(getattr(config, "loss_event_on", 0.6) or 0.6),
        event_threshold_off=float(getattr(config, "loss_event_off", 0.4) or 0.4),
        cooldown_steps=int(getattr(config, "loss_cooldown_steps", 200) or 0),
        use_candidate=bool(getattr(config, "loss_use_candidate", False)),
        use_drift_flag=bool(getattr(config, "loss_use_drift_flag", False)),
        adapt_lambda=bool(getattr(config, "loss_apply_lambda", True)),
        adapt_tau=bool(getattr(config, "loss_apply_tau", True)),
        base_lambda=float(initial_hparams.lambda_u),
        base_tau=float(initial_hparams.tau),
        safety_enabled=bool(getattr(config, "loss_safety_enabled", False)),
        safety_use_candidate=bool(getattr(config, "loss_safety_use_candidate", True)),
        safety_use_severity=bool(getattr(config, "loss_safety_use_severity", False)),
        safety_severity_threshold=float(getattr(config, "loss_safety_severity_threshold", 0.6) or 0.6),
        safety_cooldown_steps=int(getattr(config, "loss_safety_cooldown_steps", 200) or 0),
        safety_tau=getattr(config, "loss_safety_tau", None),
        safety_lambda=getattr(config, "loss_safety_lambda", None),
    )
    loss_scheduler = LossAdaptiveScheduler(loss_cfg)
    step_scheduler = StepScheduler(ema_scheduler, loss_scheduler)
    for step, batch in enumerate(batch_iter, start=1):
        if step > config.n_steps:
            break
        scheduler_state.step = step
        (
            x_labeled_raw,
            y_labeled_raw,
            x_unlabeled_raw,
            y_unlabeled_raw,
        ) = batch
        batch_sample_count = len(x_labeled_raw) + len(x_unlabeled_raw)
        seen_samples += batch_sample_count
        sample_idx += batch_sample_count
        x_labeled_np = vectorizer.transform_many(x_labeled_raw)
        x_unlabeled_np = vectorizer.transform_many(x_unlabeled_raw)
        y_labeled_np = label_encoder.encode_many(y_labeled_raw) if y_labeled_raw else None
        y_unlabeled_np = label_encoder.encode_many(y_unlabeled_raw) if y_unlabeled_raw else None

        x_labeled = _to_tensor(x_labeled_np, device)
        y_labeled = _to_label_tensor(y_labeled_np, device)
        x_unlabeled = _to_tensor(x_unlabeled_np, device)

        optimizer.zero_grad(set_to_none=True)
        losses = model.compute_losses(
            x_labeled=x_labeled,
            y_labeled=y_labeled,
            x_unlabeled=x_unlabeled,
            hparams=current_hparams,
        )
        losses.total.backward()
        optimizer.step()

        stats = _collect_statistics(losses, y_labeled_np, y_unlabeled_np)
        signals = stats["signals"]
        pl_accept_rate, pl_precision, pl_accept_count, pl_total = _compute_pseudo_label_metrics(
            stats.get("teacher_probs"),
            y_unlabeled_np,
            current_hparams.tau,
        )
        unsup_weighted = float(current_hparams.lambda_u * losses.unsupervised.detach().cpu().item())
        sup_loss_val = float(losses.supervised.detach().cpu().item())
        loss_total_val = sup_loss_val + unsup_weighted
        unsup_ratio = float(unsup_weighted / loss_total_val) if loss_total_val > 0 else float("nan")
        drift_flag, monitor_severity = drift_monitor.update(signals, step, sample_idx=int(sample_idx))
        monitor_vote_count = getattr(drift_monitor, "last_vote_count", None)
        monitor_vote_score = getattr(drift_monitor, "last_vote_score", None)
        monitor_fused_severity = getattr(drift_monitor, "last_fused_severity", None)
        candidate_flag = bool(getattr(drift_monitor, "last_candidate_flag", drift_flag))
        confirm_delay = int(getattr(drift_monitor, "last_confirm_delay", -1))
        candidate_count_total = int(getattr(drift_monitor, "candidate_count_total", 0))
        confirmed_count_total = int(getattr(drift_monitor, "confirmed_count_total", 0))
        confirm_rule_effective = str(getattr(drift_monitor, "last_confirm_rule", ""))
        last_perm_pvalue = float(getattr(drift_monitor, "last_perm_pvalue", float("nan")))
        last_perm_effect = float(getattr(drift_monitor, "last_perm_effect", float("nan")))
        perm_test_count_total = int(getattr(drift_monitor, "perm_test_count_total", 0))
        perm_accept_count_total = int(getattr(drift_monitor, "perm_accept_count_total", 0))
        perm_reject_count_total = int(getattr(drift_monitor, "perm_reject_count_total", 0))
        cooldown_active = bool(getattr(drift_monitor, "last_cooldown_active", False))
        cooldown_remaining = int(getattr(drift_monitor, "last_cooldown_remaining", 0) or 0)
        effective_cooldown = int(getattr(drift_monitor, "last_effective_confirm_cooldown", confirm_cooldown) or 0)
        confirm_rate_per10k = float(getattr(drift_monitor, "last_confirm_rate_per10k", 0.0) or 0.0)
        adaptive_active = bool(getattr(drift_monitor, "last_adaptive_cooldown_enabled", False))

        if severity_calibrator:
            if baseline_freeze_remaining > 0:
                baseline_freeze_remaining -= 1
            else:
                severity_calibrator.update_baselines(
                    signals["error_rate"],
                    signals["divergence"],
                    signals["teacher_entropy"],
                )
        severity_raw = 0.0
        severity_norm = 0.0
        severity_confirmed = True
        gate_mode = str(config.severity_gate).lower()
        if gate_mode in {"confirmed_only", "confirmed_streak"}:
            vote = float(monitor_vote_score) if monitor_vote_score is not None else 0.0
            severity_event = bool(drift_flag) and vote >= float(config.trigger_threshold)
            severity_confirmed_streak = severity_confirmed_streak + 1 if severity_event else 0
            if gate_mode == "confirmed_only":
                severity_confirmed = severity_event
            else:
                min_streak = max(1, int(getattr(config, "severity_gate_min_streak", 1) or 1))
                severity_confirmed = severity_confirmed_streak >= min_streak
        else:
            severity_confirmed_streak = 0
        if severity_calibrator and drift_flag:
            severity_raw, severity_norm = severity_calibrator.compute_severity(
                signals["error_rate"],
                signals["divergence"],
                signals["teacher_entropy"],
            )
        severity_norm_apply = severity_norm if severity_confirmed else 0.0
        if drift_flag and severity_calibrator and config.freeze_baseline_steps > 0 and severity_norm_apply > 0.0:
            baseline_freeze_remaining = max(baseline_freeze_remaining, int(config.freeze_baseline_steps))

        if config.use_severity_v2:
            decay = float(config.severity_decay)
            if not (0.0 <= decay <= 1.0):
                decay = min(1.0, max(0.0, decay))
            severity_carry = max(severity_carry * decay, severity_norm_apply)
            severity_for_scheduler = severity_carry
        else:
            severity_carry = severity_norm_apply
            severity_for_scheduler = severity_norm_apply
        if config.use_severity_scheduler:
            current_hparams, regime = update_hparams_with_severity(
                scheduler_state,
                current_hparams,
                drift_flag,
                monitor_severity,
                severity_for_scheduler,
                severity_scheduler_cfg,
            )
        else:
            current_hparams, regime = update_hparams(
                scheduler_state, current_hparams, drift_flag, monitor_severity
            )
        scheduler_out = step_scheduler.step(
            signals,
            drift_flag=drift_flag,
            candidate_flag=candidate_flag,
        )
        if scheduler_out.overrides:
            current_hparams = replace(current_hparams, **scheduler_out.overrides)
        model.update_teacher(current_hparams.alpha)
        _set_optimizer_lr(optimizer, current_hparams.lr)

        acc_value, kappa_value = _update_metrics(
            metric,
            kappa_metric,
            stats["student_probs_labeled"],
            y_labeled_np,
            stats["student_probs_unlabeled"],
            y_unlabeled_np,
        )
        acc_series.append((int(sample_idx), float(acc_value)))
        if candidate_flag:
            candidate_sample_idxs.append(int(sample_idx))
        if drift_flag:
            confirmed_sample_idxs.append(int(sample_idx))
        log_entry = {
                "step": step,
                "seen_samples": seen_samples,
                "sample_idx": sample_idx,
                "dataset_name": config.dataset_name,
                "dataset_type": config.dataset_type,
                "model_variant": config.model_variant,
                "seed": config.seed,
                "monitor_preset": config.monitor_preset,
                "monitor_preset_base": monitor_preset_base,
                "signal_set": config.signal_set,
                "monitor_ph_params": monitor_ph_params_json,
                "monitor_ph_overrides": monitor_ph_overrides_json,
                "ph_error_threshold": ph_error_threshold,
                "ph_error_delta": ph_error_delta,
                "ph_error_alpha": ph_error_alpha,
                "ph_error_min_instances": ph_error_min_instances,
                "ph_divergence_threshold": ph_div_threshold,
                "ph_divergence_delta": ph_div_delta,
                "ph_divergence_alpha": ph_div_alpha,
                "ph_divergence_min_instances": ph_div_min_instances,
                "ph_entropy_threshold": ph_ent_threshold,
                "ph_entropy_delta": ph_ent_delta,
                "ph_entropy_alpha": ph_ent_alpha,
                "ph_entropy_min_instances": ph_ent_min_instances,
                "trigger_mode": config.trigger_mode,
                "trigger_k": int(config.trigger_k),
                "trigger_threshold": float(config.trigger_threshold),
                "trigger_weights": config.trigger_weights,
                "confirm_window": int(config.confirm_window),
                "confirm_cooldown": int(confirm_cooldown),
                "effective_confirm_cooldown": int(effective_cooldown),
                "confirm_rate_per10k": float(confirm_rate_per10k),
                "adaptive_cooldown_enabled": int(bool(adaptive_active)),
                "adaptive_window": int(adaptive_window),
                "adaptive_lower_per10k": float(adaptive_lower_per10k),
                "adaptive_upper_per10k": float(adaptive_upper_per10k),
                "adaptive_cooldown_low": int(adaptive_cooldown_low),
                "adaptive_cooldown_high": int(adaptive_cooldown_high),
                "confirm_cooldown_active": int(bool(cooldown_active)),
                "confirm_cooldown_remaining": int(cooldown_remaining),
                "severity_scheduler_scale": float(config.severity_scheduler_scale),
                "metric_accuracy": acc_value,
                "metric_kappa": kappa_value,
                "student_error_rate": signals["error_rate"],
                "teacher_entropy": signals["teacher_entropy"],
                "divergence_js": signals["divergence"],
                "drift_flag": int(drift_flag),
                "candidate_flag": int(candidate_flag),
                "candidate_count_total": candidate_count_total,
                "confirmed_count_total": confirmed_count_total,
                "confirm_delay": confirm_delay,
                "confirm_rule_effective": confirm_rule_effective,
                "last_perm_pvalue": last_perm_pvalue,
                "last_perm_effect": last_perm_effect,
                "perm_test_count_total": perm_test_count_total,
                "perm_accept_count_total": perm_accept_count_total,
                "perm_reject_count_total": perm_reject_count_total,
                "monitor_severity": monitor_severity,
                "monitor_fused_severity": monitor_fused_severity if monitor_fused_severity is not None else 0.0,
                "monitor_vote_count": int(monitor_vote_count) if monitor_vote_count is not None else 0,
                "monitor_vote_score": float(monitor_vote_score) if monitor_vote_score is not None else 0.0,
                "drift_severity_raw": severity_raw if drift_flag else 0.0,
                "drift_severity": severity_norm if drift_flag else 0.0,
                "severity_carry": severity_carry,
                "use_severity_v2": int(bool(config.use_severity_v2)),
                "severity_gate": str(config.severity_gate),
                "severity_confirmed": int(bool(severity_confirmed)) if drift_flag else 0,
                "severity_gate_min_streak": int(getattr(config, "severity_gate_min_streak", 1) or 1),
                "severity_confirmed_streak": int(severity_confirmed_streak) if drift_flag else 0,
                "entropy_mode": config.entropy_mode,
                "decay": float(config.severity_decay),
                "freeze_baseline_steps": int(config.freeze_baseline_steps),
                "ema_decay_mode": ema_decay_mode,
                "ema_gamma_min": float(ema_gamma_min),
                "ema_gamma_max": float(ema_gamma_max),
                "ema_gamma_fixed": float(ema_gamma_fixed) if ema_gamma_fixed is not None else float("nan"),
                "ema_severity_mode": str(ema_severity_mode),
                "ema_severity_weights": ",".join(
                    f"{w:.6g}" for w in (ema_severity_weights or (0.0, 0.0, 0.0))
                ),
                "ema_severity_smoothing": float(ema_severity_smoothing),
                "ema_severity_threshold": float(ema_severity_threshold),
                "ema_severity_threshold_off": float(ema_severity_threshold_off) if ema_severity_threshold_off is not None else float("nan"),
                "ema_cooldown_steps": int(ema_cooldown_steps),
                "ema_use_candidate": int(bool(ema_use_candidate)),
                "ema_use_drift_flag": int(bool(ema_use_drift_flag)),
                "regime": regime,
                "alpha": current_hparams.alpha,
                "lr": current_hparams.lr,
                "lambda_u": current_hparams.lambda_u,
                "tau": current_hparams.tau,
                "timestamp": time.time(),
                "supervised_loss": float(losses.supervised.detach().cpu().item()),
                "unsupervised_loss": float(losses.unsupervised.detach().cpu().item()),
                "unsupervised_loss_weighted": float(unsup_weighted),
                "unsupervised_loss_ratio": float(unsup_ratio),
                "pl_accept_rate": float(pl_accept_rate),
                "pl_precision": float(pl_precision),
                "pl_accept_count": int(pl_accept_count),
                "pl_total": int(pl_total),
            }
        log_entry.update(scheduler_out.logs)
        logs.append(log_entry)
    df = pd.DataFrame(logs)
    if config.log_path:
        log_dir = os.path.dirname(config.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        df.to_csv(config.log_path, index=False)
        # sidecar summary：用于后续指标统计，避免对大日志做全量扫描
        try:
            summary_path = Path(config.log_path).with_suffix(".summary.json")
            horizon = int(df["sample_idx"].iloc[-1] + 1) if not df.empty else 0
            acc_final = float(df["metric_accuracy"].iloc[-1]) if not df.empty else float("nan")
            mean_acc = float(df["metric_accuracy"].mean()) if not df.empty else float("nan")
            acc_min = float(df["metric_accuracy"].min()) if not df.empty else float("nan")
            last = df.iloc[-1] if not df.empty else None
            # perm_test 诊断（用于 Track AM；尽量从 monitor 内存状态读取，避免扫描大日志）
            perm_pvalues = list(getattr(drift_monitor, "_perm_pvalues", []) or [])
            perm_effects = list(getattr(drift_monitor, "_perm_effects", []) or [])
            perm_alpha = float(getattr(drift_monitor, "perm_alpha", 0.01) or 0.01)
            tw_obj = getattr(drift_monitor, "trigger_weights", None)
            if isinstance(tw_obj, dict):
                for k in ("__perm_alpha", "perm_alpha"):
                    if k in tw_obj:
                        try:
                            perm_alpha = float(tw_obj[k])  # type: ignore[arg-type]
                        except Exception:
                            pass
            def _q(xs: list[float], q: float) -> float:
                if not xs:
                    return float("nan")
                arr = np.asarray(xs, dtype=np.float64)
                return float(np.quantile(arr, q))
            perm_pvalue_p50 = _q(perm_pvalues, 0.50)
            perm_pvalue_p90 = _q(perm_pvalues, 0.90)
            perm_pvalue_p99 = _q(perm_pvalues, 0.99)
            perm_effect_p50 = _q(perm_effects, 0.50)
            perm_effect_p90 = _q(perm_effects, 0.90)
            perm_effect_p99 = _q(perm_effects, 0.99)
            perm_pvalue_le_alpha_ratio = (float(sum(1 for p in perm_pvalues if p <= perm_alpha)) / float(len(perm_pvalues))) if perm_pvalues else float("nan")
            payload: Dict[str, Any] = {
                "dataset_name": config.dataset_name,
                "dataset_type": config.dataset_type,
                "model_variant": config.model_variant,
                "seed": int(config.seed),
                "monitor_preset": config.monitor_preset,
                "monitor_preset_base": monitor_preset_base,
                "signal_set": config.signal_set,
                "monitor_ph_params": monitor_ph_params,
                "monitor_ph_overrides": monitor_ph_overrides,
                "trigger_mode": config.trigger_mode,
                "trigger_k": int(config.trigger_k),
                "trigger_threshold": float(config.trigger_threshold),
                "trigger_weights": config.trigger_weights,
                "confirm_window": int(config.confirm_window),
                "confirm_cooldown": int(confirm_cooldown),
                "adaptive_cooldown_enabled": int(bool(adaptive_enabled)),
                "adaptive_window": int(adaptive_window),
                "adaptive_lower_per10k": float(adaptive_lower_per10k),
                "adaptive_upper_per10k": float(adaptive_upper_per10k),
                "adaptive_cooldown_low": int(adaptive_cooldown_low),
                "adaptive_cooldown_high": int(adaptive_cooldown_high),
                "severity_scheduler_scale": float(config.severity_scheduler_scale),
                "use_severity_v2": int(bool(config.use_severity_v2)),
                "severity_gate": str(config.severity_gate),
                "severity_gate_min_streak": int(getattr(config, "severity_gate_min_streak", 1) or 1),
                "entropy_mode": str(config.entropy_mode),
                "severity_decay": float(config.severity_decay),
                "freeze_baseline_steps": int(config.freeze_baseline_steps),
                "ema_decay_mode": ema_decay_mode,
                "ema_gamma_min": float(ema_gamma_min),
                "ema_gamma_max": float(ema_gamma_max),
                "ema_gamma_fixed": float(ema_gamma_fixed) if ema_gamma_fixed is not None else float("nan"),
                "ema_severity_mode": str(ema_severity_mode),
                "ema_severity_weights": ",".join(
                    f"{w:.6g}" for w in (ema_severity_weights or (0.0, 0.0, 0.0))
                ),
                "ema_severity_smoothing": float(ema_severity_smoothing),
                "ema_severity_threshold": float(ema_severity_threshold),
                "ema_severity_threshold_off": float(ema_severity_threshold_off) if ema_severity_threshold_off is not None else float("nan"),
                "ema_cooldown_steps": int(ema_cooldown_steps),
                "ema_use_candidate": int(bool(ema_use_candidate)),
                "ema_use_drift_flag": int(bool(ema_use_drift_flag)),
                "n_steps": int(config.n_steps),
                "horizon": horizon,
                "acc_final": acc_final,
                "mean_acc": mean_acc,
                "acc_min": acc_min,
                "candidate_sample_idxs": candidate_sample_idxs,
                "confirmed_sample_idxs": confirmed_sample_idxs,
                "acc_series": [[int(x), float(a)] for x, a in acc_series],
                "candidate_count_total": int(last["candidate_count_total"]) if last is not None else 0,
                "confirmed_count_total": int(last["confirmed_count_total"]) if last is not None else 0,
                "created_at": float(time.time()),
                "confirm_rule_effective": str(getattr(drift_monitor, "last_confirm_rule", "")),
                "last_perm_pvalue": float(getattr(drift_monitor, "last_perm_pvalue", float("nan"))),
                "last_perm_effect": float(getattr(drift_monitor, "last_perm_effect", float("nan"))),
                "perm_test_count_total": int(getattr(drift_monitor, "perm_test_count_total", 0)),
                "perm_accept_count_total": int(getattr(drift_monitor, "perm_accept_count_total", 0)),
                "perm_reject_count_total": int(getattr(drift_monitor, "perm_reject_count_total", 0)),
                "perm_alpha": float(perm_alpha),
                "perm_pvalue_p50": float(perm_pvalue_p50),
                "perm_pvalue_p90": float(perm_pvalue_p90),
                "perm_pvalue_p99": float(perm_pvalue_p99),
                "perm_pvalue_le_alpha_ratio": float(perm_pvalue_le_alpha_ratio),
                "perm_effect_p50": float(perm_effect_p50),
                "perm_effect_p90": float(perm_effect_p90),
                "perm_effect_p99": float(perm_effect_p99),
            }
            if "ema_decay" in df.columns:
                ema_vals = df["ema_decay"].astype(float).to_numpy()
                if ema_vals.size > 0:
                    payload["ema_decay_mean"] = float(np.mean(ema_vals))
                    payload["ema_decay_p10"] = float(np.quantile(ema_vals, 0.10))
                    payload["ema_decay_p50"] = float(np.quantile(ema_vals, 0.50))
                    payload["ema_decay_p90"] = float(np.quantile(ema_vals, 0.90))
            if "ema_severity_ema" in df.columns:
                sev_vals = df["ema_severity_ema"].astype(float).to_numpy()
                if sev_vals.size > 0:
                    payload["ema_severity_mean"] = float(np.mean(sev_vals))
            if "lambda_u" in df.columns:
                lambda_vals = df["lambda_u"].astype(float).to_numpy()
                if lambda_vals.size > 0:
                    payload["lambda_u_mean"] = float(np.mean(lambda_vals))
                    payload["lambda_u_p50"] = float(np.quantile(lambda_vals, 0.50))
                    payload["lambda_u_p90"] = float(np.quantile(lambda_vals, 0.90))
                    payload["lambda_u_p99"] = float(np.quantile(lambda_vals, 0.99))
            if "tau" in df.columns:
                tau_vals = df["tau"].astype(float).to_numpy()
                if tau_vals.size > 0:
                    payload["tau_mean"] = float(np.mean(tau_vals))
                    payload["tau_p50"] = float(np.quantile(tau_vals, 0.50))
                    payload["tau_p90"] = float(np.quantile(tau_vals, 0.90))
                    payload["tau_p99"] = float(np.quantile(tau_vals, 0.99))
            if "pl_accept_rate" in df.columns:
                accept_vals = df["pl_accept_rate"].astype(float).to_numpy()
                finite = accept_vals[np.isfinite(accept_vals)]
                if finite.size > 0:
                    payload["pl_accept_rate_mean"] = float(np.mean(finite))
            if "pl_precision" in df.columns:
                prec_vals = df["pl_precision"].astype(float).to_numpy()
                finite = prec_vals[np.isfinite(prec_vals)]
                if finite.size > 0:
                    payload["pl_precision_mean"] = float(np.mean(finite))
            summary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            # summary 写失败不应影响主流程
            pass
    return df


def _to_tensor(arr: Optional[np.ndarray], device: torch.device) -> Optional[Tensor]:
    if arr is None:
        return None
    return torch.from_numpy(arr).to(device=device)


def _to_label_tensor(arr: Optional[np.ndarray], device: torch.device) -> Optional[Tensor]:
    if arr is None:
        return None
    return torch.from_numpy(arr.astype(np.int64)).to(device=device)


def _collect_statistics(
    losses: LossOutputs,
    y_labeled: Optional[np.ndarray],
    y_unlabeled: Optional[np.ndarray],
) -> Dict[str, Any]:
    """提取漂移信号与概率。"""
    student_probs_labeled = None
    if "student_logits_labeled" in losses.details and y_labeled is not None:
        probs = torch.softmax(losses.details["student_logits_labeled"], dim=1)
        student_probs_labeled = probs.detach().cpu().numpy()
    teacher_probs = None
    student_probs_unlabeled = None
    if "teacher_probs" in losses.details:
        teacher_probs = losses.details["teacher_probs"].detach().cpu().numpy()
    if "student_probs_unlabeled" in losses.details:
        student_probs_unlabeled = losses.details["student_probs_unlabeled"].detach().cpu().numpy()
    signals = compute_signals(
        student_probs_labeled=student_probs_labeled,
        y_labeled=y_labeled,
        teacher_probs_unlabeled=teacher_probs,
        student_probs_unlabeled=student_probs_unlabeled,
    )
    return {
        "signals": signals,
        "teacher_probs": teacher_probs,
        "student_probs_labeled": student_probs_labeled,
        "student_probs_unlabeled": student_probs_unlabeled,
    }


def _update_metrics(
    metric: metrics.base.Metric,
    kappa_metric: metrics.base.Metric,
    student_probs_labeled: Optional[np.ndarray],
    y_labeled: Optional[np.ndarray],
    student_probs_unlabeled: Optional[np.ndarray],
    y_unlabeled: Optional[np.ndarray],
) -> Tuple[float, float]:
    """逐样本更新准确率与 Kappa。"""
    if student_probs_labeled is not None and y_labeled is not None:
        preds = student_probs_labeled.argmax(axis=1)
        for y_true, y_pred in zip(y_labeled, preds):
            metric.update(y_true=y_true, y_pred=y_pred)
            kappa_metric.update(y_true=y_true, y_pred=y_pred)
    if student_probs_unlabeled is not None and y_unlabeled is not None:
        preds_u = student_probs_unlabeled.argmax(axis=1)
        for y_true, y_pred in zip(y_unlabeled, preds_u):
            metric.update(y_true=y_true, y_pred=y_pred)
            kappa_metric.update(y_true=y_true, y_pred=y_pred)
    return metric.get(), kappa_metric.get()


def _compute_pseudo_label_metrics(
    teacher_probs: Optional[np.ndarray],
    y_unlabeled: Optional[np.ndarray],
    tau: float,
) -> Tuple[float, float, int, int]:
    """Compute pseudo-label accept rate and precision (for logging only)."""
    if teacher_probs is None or y_unlabeled is None or len(y_unlabeled) == 0:
        return float("nan"), float("nan"), 0, 0
    max_probs = teacher_probs.max(axis=1)
    pseudo_labels = teacher_probs.argmax(axis=1)
    if max_probs.size == 0:
        return float("nan"), float("nan"), 0, 0
    mask = max_probs >= float(tau)
    accept_count = int(mask.sum())
    total = int(max_probs.size)
    accept_rate = float(accept_count) / float(total) if total > 0 else float("nan")
    if accept_count <= 0:
        return accept_rate, float("nan"), accept_count, total
    precision = float(np.mean(pseudo_labels[mask] == y_unlabeled[mask]))
    return accept_rate, precision, accept_count, total


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
