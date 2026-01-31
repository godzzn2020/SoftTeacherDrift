"""Adaptive lambda_u/tau scheduler for experiment B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple


def _clean_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def _clamp_unit(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _clamp_nonneg(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, float(value))


@dataclass
class RunningStat:
    mean: float = 0.0
    var: float = 1.0
    initialized: bool = False

    def score_and_update(self, x: Optional[float], *, momentum: float, eps: float) -> Optional[float]:
        if x is None:
            return None
        z = None
        if self.initialized:
            std = (self.var + eps) ** 0.5
            if std > 0.0:
                z = (float(x) - self.mean) / std
        if not self.initialized:
            self.mean = float(x)
            self.var = 1.0
            self.initialized = True
        else:
            self.mean = momentum * self.mean + (1.0 - momentum) * float(x)
            self.var = momentum * self.var + (1.0 - momentum) * (float(x) - self.mean) ** 2
        return z


@dataclass
class LossAdaptiveConfig:
    mode: str = "none"
    lambda_min: Optional[float] = None
    lambda_max: Optional[float] = None
    tau_min: Optional[float] = None
    tau_max: Optional[float] = None
    lambda_fixed: Optional[float] = None
    tau_fixed: Optional[float] = None
    severity_mode: str = "max"
    severity_weights: Tuple[float, float, float] = (0.0, 0.5, 0.5)
    severity_momentum: float = 0.99
    severity_smoothing: float = 0.9
    severity_low: float = 0.0
    severity_high: float = 2.0
    severity_eps: float = 1e-6
    severity_use_error: bool = False
    event_threshold_on: float = 0.6
    event_threshold_off: float = 0.4
    cooldown_steps: int = 200
    use_candidate: bool = False
    use_drift_flag: bool = False
    adapt_lambda: bool = True
    adapt_tau: bool = True
    base_lambda: float = 1.0
    base_tau: float = 0.9
    safety_enabled: bool = False
    safety_use_candidate: bool = True
    safety_use_severity: bool = False
    safety_severity_threshold: float = 0.6
    safety_cooldown_steps: int = 200
    safety_tau: Optional[float] = None
    safety_lambda: Optional[float] = None


@dataclass
class LossAdaptStep:
    overrides: Dict[str, float]
    logs: Dict[str, float | int | str]


class LossAdaptiveScheduler:
    """Signal-driven scheduler for lambda_u and tau."""

    def __init__(self, config: LossAdaptiveConfig) -> None:
        self.config = config
        self.mode = str(config.mode or "none").strip().lower()
        if self.mode not in {"none", "fixed", "severity_continuous", "severity_event"}:
            self.mode = "none"
        self.severity_mode = str(config.severity_mode or "max").strip().lower()
        if self.severity_mode not in {"max", "weighted"}:
            self.severity_mode = "max"
        self.severity_weights = tuple(float(x) for x in (config.severity_weights or (0.0, 0.0, 0.0)))
        smoothing = float(config.severity_smoothing)
        if not (0.0 <= smoothing <= 1.0):
            smoothing = min(1.0, max(0.0, smoothing))
        self.severity_smoothing = smoothing
        self.severity_momentum = float(config.severity_momentum)
        self.severity_low = float(config.severity_low)
        self.severity_high = max(float(config.severity_high), self.severity_low + 1e-6)
        self.severity_eps = float(config.severity_eps)
        self.use_error = bool(config.severity_use_error)
        self.event_on = _clamp_unit(float(config.event_threshold_on))
        self.event_off = _clamp_unit(float(config.event_threshold_off))
        if self.event_off > self.event_on:
            self.event_off, self.event_on = self.event_on, self.event_off
        self.cooldown_steps = max(0, int(config.cooldown_steps))
        self.use_candidate = bool(config.use_candidate)
        self.use_drift_flag = bool(config.use_drift_flag)
        self.adapt_lambda = bool(config.adapt_lambda)
        self.adapt_tau = bool(config.adapt_tau)
        self.base_lambda = _clamp_nonneg(float(config.base_lambda))
        self.base_tau = _clamp_unit(float(config.base_tau))
        self.safety_enabled = bool(config.safety_enabled)
        self.safety_use_candidate = bool(config.safety_use_candidate)
        self.safety_use_severity = bool(config.safety_use_severity)
        self.safety_severity_threshold = _clamp_unit(float(config.safety_severity_threshold))
        self.safety_cooldown_steps = max(0, int(config.safety_cooldown_steps))

        self.lambda_min = (
            _clamp_nonneg(float(config.lambda_min))
            if config.lambda_min is not None
            else _clamp_nonneg(self.base_lambda * 0.5)
        )
        self.lambda_max = (
            _clamp_nonneg(float(config.lambda_max))
            if config.lambda_max is not None
            else _clamp_nonneg(self.base_lambda * 1.5)
        )
        if self.lambda_min > self.lambda_max:
            self.lambda_min, self.lambda_max = self.lambda_max, self.lambda_min
        self.tau_min = (
            _clamp_unit(float(config.tau_min))
            if config.tau_min is not None
            else _clamp_unit(self.base_tau - 0.05)
        )
        self.tau_max = (
            _clamp_unit(float(config.tau_max))
            if config.tau_max is not None
            else _clamp_unit(self.base_tau + 0.05)
        )
        if self.tau_min > self.tau_max:
            self.tau_min, self.tau_max = self.tau_max, self.tau_min

        self.lambda_fixed = (
            _clamp_nonneg(float(config.lambda_fixed))
            if config.lambda_fixed is not None
            else self.base_lambda
        )
        self.tau_fixed = (
            _clamp_unit(float(config.tau_fixed))
            if config.tau_fixed is not None
            else self.base_tau
        )

        if config.safety_lambda is None:
            self.safety_lambda = float(self.lambda_min)
        else:
            self.safety_lambda = _clamp_nonneg(float(config.safety_lambda))
        if config.safety_tau is None:
            self.safety_tau = float(self.tau_max)
        else:
            self.safety_tau = _clamp_unit(float(config.safety_tau))

        self._stat_error = RunningStat()
        self._stat_div = RunningStat()
        self._stat_entropy = RunningStat()
        self._severity_ema = 0.0
        self._severity_ema_initialized = False
        self._risk_active = False
        self._cooldown_remaining = 0
        self._safety_remaining = 0

    def _compute_severity(
        self,
        error_rate: Optional[float],
        divergence: Optional[float],
        entropy: Optional[float],
    ) -> tuple[float, float, float, Optional[float], Optional[float], Optional[float]]:
        m = float(self.severity_momentum)
        eps = float(self.severity_eps)
        err_v = _clean_value(error_rate) if self.use_error else None
        div_v = _clean_value(divergence)
        ent_v = _clean_value(entropy)

        z_err = self._stat_error.score_and_update(err_v, momentum=m, eps=eps) if err_v is not None else None
        z_div = self._stat_div.score_and_update(div_v, momentum=m, eps=eps) if div_v is not None else None
        z_ent = self._stat_entropy.score_and_update(ent_v, momentum=m, eps=eps) if ent_v is not None else None

        z_vals = []
        w_vals = []
        if z_err is not None:
            z_vals.append(abs(float(z_err)))
            w_vals.append(float(self.severity_weights[0]))
        if z_div is not None:
            z_vals.append(abs(float(z_div)))
            w_vals.append(float(self.severity_weights[1]))
        if z_ent is not None:
            z_vals.append(abs(float(z_ent)))
            w_vals.append(float(self.severity_weights[2]))

        if not z_vals:
            severity_raw = 0.0
        elif self.severity_mode == "max":
            severity_raw = float(max(z_vals))
        else:
            weight_sum = sum(w for w in w_vals if w > 0.0)
            if weight_sum <= 0.0:
                severity_raw = 0.0
            else:
                severity_raw = float(sum(w * z for w, z in zip(w_vals, z_vals)) / weight_sum)

        severity_norm = (severity_raw - self.severity_low) / (self.severity_high - self.severity_low)
        severity_norm = _clamp_unit(float(severity_norm))

        if not self._severity_ema_initialized:
            self._severity_ema = float(severity_norm)
            self._severity_ema_initialized = True
        else:
            self._severity_ema = (
                self.severity_smoothing * self._severity_ema
                + (1.0 - self.severity_smoothing) * float(severity_norm)
            )
        severity_ema = float(self._severity_ema)
        return severity_raw, severity_norm, severity_ema, z_err, z_div, z_ent

    def step(
        self,
        signals: Dict[str, float],
        *,
        drift_flag: bool,
        candidate_flag: bool,
    ) -> LossAdaptStep:
        severity_raw = 0.0
        severity_norm = 0.0
        severity_ema = float(self._severity_ema)
        z_err = None
        z_div = None
        z_ent = None

        if self.mode in {"severity_continuous", "severity_event"} or self.safety_use_severity:
            severity_raw, severity_norm, severity_ema, z_err, z_div, z_ent = self._compute_severity(
                signals.get("error_rate"),
                signals.get("divergence"),
                signals.get("teacher_entropy"),
            )

        lambda_val = float(self.lambda_fixed)
        tau_val = float(self.tau_fixed)
        triggered = False
        risk_active = bool(self._risk_active)
        cooldown_remaining = int(self._cooldown_remaining)

        if self.mode == "fixed":
            lambda_val = float(self.lambda_fixed)
            tau_val = float(self.tau_fixed)
        elif self.mode == "severity_continuous":
            sev = _clamp_unit(float(severity_ema))
            lambda_val = self.lambda_max - (self.lambda_max - self.lambda_min) * sev
            tau_val = self.tau_min + (self.tau_max - self.tau_min) * sev
        elif self.mode == "severity_event":
            triggered = float(severity_ema) >= float(self.event_on)
            if self.use_candidate and candidate_flag:
                triggered = True
            if self.use_drift_flag and drift_flag:
                triggered = True
            if triggered:
                risk_active = True
                cooldown_remaining = max(cooldown_remaining, self.cooldown_steps)
            if risk_active:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                if float(severity_ema) <= float(self.event_off) and cooldown_remaining <= 0:
                    risk_active = False
            if risk_active:
                lambda_val = float(self.lambda_min)
                tau_val = float(self.tau_max)
            else:
                lambda_val = float(self.lambda_max)
                tau_val = float(self.tau_min)

        safety_triggered = False
        safety_active = False
        safety_remaining = int(self._safety_remaining)
        if self.safety_enabled:
            if self.safety_use_candidate and candidate_flag:
                safety_triggered = True
            if self.safety_use_severity and float(severity_ema) >= float(self.safety_severity_threshold):
                safety_triggered = True
            if safety_triggered:
                safety_remaining = max(safety_remaining, self.safety_cooldown_steps)
            if safety_remaining > 0:
                safety_active = True
                safety_remaining -= 1

        lambda_before_safe = float(lambda_val)
        tau_before_safe = float(tau_val)
        if safety_active:
            if self.safety_lambda is not None:
                lambda_val = min(lambda_val, float(self.safety_lambda))
            if self.safety_tau is not None:
                tau_val = max(tau_val, float(self.safety_tau))

        lambda_val = _clamp_nonneg(float(lambda_val))
        tau_val = _clamp_unit(float(tau_val))

        overrides: Dict[str, float] = {}
        if self.adapt_lambda:
            overrides["lambda_u"] = float(lambda_val)
        if self.adapt_tau:
            overrides["tau"] = float(tau_val)

        self._risk_active = bool(risk_active)
        self._cooldown_remaining = int(cooldown_remaining)
        self._safety_remaining = int(safety_remaining)

        lambda_safe_applied = int(bool(safety_active and lambda_val < lambda_before_safe - 1e-12))
        tau_safe_applied = int(bool(safety_active and tau_val > tau_before_safe + 1e-12))
        logs: Dict[str, float | int | str] = {
            "loss_scheduler_mode": str(self.mode),
            "loss_severity_raw": float(severity_raw),
            "loss_severity_norm": float(severity_norm),
            "loss_severity_ema": float(severity_ema),
            "loss_severity_z_error": float(z_err) if z_err is not None else float("nan"),
            "loss_severity_z_divergence": float(z_div) if z_div is not None else float("nan"),
            "loss_severity_z_entropy": float(z_ent) if z_ent is not None else float("nan"),
            "loss_severity_mode": str(self.severity_mode),
            "loss_severity_weights": ",".join(f"{w:.6g}" for w in self.severity_weights),
            "loss_event_on": float(self.event_on),
            "loss_event_off": float(self.event_off),
            "loss_cooldown_steps": int(self.cooldown_steps),
            "loss_cooldown_remaining": int(cooldown_remaining),
            "loss_triggered": int(bool(triggered)),
            "loss_risk": int(bool(risk_active)),
            "loss_risk_mode": int(bool(safety_active)),
            "loss_risk_triggered": int(bool(safety_triggered)),
            "loss_risk_remaining": int(safety_remaining),
            "loss_tau_safe": float(self.safety_tau) if self.safety_tau is not None else float("nan"),
            "loss_lambda_safe": float(self.safety_lambda) if self.safety_lambda is not None else float("nan"),
            "loss_tau_safe_applied": int(tau_safe_applied),
            "loss_lambda_safe_applied": int(lambda_safe_applied),
            "risk_mode": int(bool(safety_active)),
            "loss_lambda_u": float(lambda_val),
            "loss_tau": float(tau_val),
            "loss_apply_lambda": int(bool(self.adapt_lambda)),
            "loss_apply_tau": int(bool(self.adapt_tau)),
        }
        return LossAdaptStep(overrides=overrides, logs=logs)
