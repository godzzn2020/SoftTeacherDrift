"""EMA decay scheduler for experiment A (and fixed override)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

from soft_drift.severity import SeverityCalibrator


def _clamp_alpha(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(0.999999, max(0.0, float(value)))


def _clamp_unit(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


@dataclass
class EmaDecayConfig:
    mode: str = "none"
    gamma_min: float = 0.95
    gamma_max: float = 0.999
    gamma_fixed: Optional[float] = None
    severity_mode: str = "max"
    severity_weights: Tuple[float, float, float] = (0.0, 0.5, 0.5)
    severity_smoothing: float = 0.9
    severity_threshold: float = 0.6
    severity_threshold_off: Optional[float] = None
    cooldown_steps: int = 200
    use_candidate: bool = False
    use_drift_flag: bool = False
    ema_momentum: float = 0.99
    eps: float = 1e-6
    severity_low: float = 0.0
    severity_high: float = 2.0
    entropy_mode: str = "overconfident"


@dataclass
class EmaDecayStep:
    overrides: Dict[str, float]
    logs: Dict[str, float | int]


class EmaDecayScheduler:
    """EMA decay scheduler driven by online severity."""

    def __init__(self, config: EmaDecayConfig, *, default_decay: float) -> None:
        self.config = config
        self.mode = str(config.mode or "none").strip().lower()
        if self.mode not in {"none", "fixed", "severity_continuous", "severity_event", "severity_event_reverse"}:
            self.mode = "none"
        gamma_min = _clamp_alpha(float(config.gamma_min))
        gamma_max = _clamp_alpha(float(config.gamma_max))
        if gamma_min > gamma_max:
            gamma_min, gamma_max = gamma_max, gamma_min
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_fixed = (
            _clamp_alpha(float(config.gamma_fixed))
            if config.gamma_fixed is not None
            else None
        )
        if self.gamma_fixed is None and self.mode == "fixed":
            self.gamma_fixed = float(self.gamma_max)
        self.severity_mode = str(config.severity_mode or "max").strip().lower()
        if self.severity_mode not in {"max", "weighted"}:
            self.severity_mode = "max"
        self.severity_weights = tuple(float(x) for x in (config.severity_weights or (0.0, 0.0, 0.0)))
        smoothing = float(config.severity_smoothing)
        if not (0.0 <= smoothing <= 1.0):
            smoothing = min(1.0, max(0.0, smoothing))
        self.severity_smoothing = smoothing
        self.severity_threshold = _clamp_unit(float(config.severity_threshold))
        if config.severity_threshold_off is None:
            self.severity_threshold_off = None
        else:
            thr_off = _clamp_unit(float(config.severity_threshold_off))
            if thr_off > self.severity_threshold:
                thr_off = float(self.severity_threshold)
            self.severity_threshold_off = float(thr_off)
        self.cooldown_steps = max(0, int(config.cooldown_steps))
        self.use_candidate = bool(config.use_candidate)
        self.use_drift_flag = bool(config.use_drift_flag)
        self.default_decay = _clamp_alpha(float(default_decay))

        self._calibrator: Optional[SeverityCalibrator] = None
        if self.mode in {"severity_continuous", "severity_event", "severity_event_reverse"}:
            self._calibrator = SeverityCalibrator(
                ema_momentum=float(config.ema_momentum),
                eps=float(config.eps),
                severity_low=float(config.severity_low),
                severity_high=float(config.severity_high),
                weights=self.severity_weights,
                entropy_mode=str(config.entropy_mode),
                mode=self.severity_mode,
            )
        self._severity_ema = 0.0
        self._severity_ema_initialized = False
        self._cooldown_remaining = 0
        self._event_active = False

    def step(
        self,
        signals: Dict[str, float],
        *,
        drift_flag: bool,
        candidate_flag: bool,
    ) -> EmaDecayStep:
        ema_decay = float(self.default_decay)
        severity_raw = 0.0
        severity_norm = 0.0
        severity_ema = float(self._severity_ema)
        triggered = False
        cooldown_remaining = int(self._cooldown_remaining)

        if self.mode == "fixed":
            ema_decay = float(self.gamma_fixed) if self.gamma_fixed is not None else float(self.gamma_max)
        elif self.mode in {"severity_continuous", "severity_event", "severity_event_reverse"} and self._calibrator is not None:
            self._calibrator.update_baselines(
                float(signals.get("error_rate", float("nan"))),
                float(signals.get("divergence", float("nan"))),
                float(signals.get("teacher_entropy", float("nan"))),
            )
            severity_raw, severity_norm = self._calibrator.compute_severity(
                float(signals.get("error_rate", float("nan"))),
                float(signals.get("divergence", float("nan"))),
                float(signals.get("teacher_entropy", float("nan"))),
            )
            if not self._severity_ema_initialized:
                severity_ema = float(severity_norm)
                self._severity_ema = float(severity_norm)
                self._severity_ema_initialized = True
            else:
                severity_ema = (
                    self.severity_smoothing * self._severity_ema
                    + (1.0 - self.severity_smoothing) * float(severity_norm)
                )
                self._severity_ema = float(severity_ema)
            sev_for_gamma = _clamp_unit(float(severity_ema))
            if self.mode == "severity_continuous":
                ema_decay = self.gamma_max - (self.gamma_max - self.gamma_min) * sev_for_gamma
            else:
                if self.mode == "severity_event_reverse":
                    base_gamma = float(self.gamma_min)
                    event_gamma = float(self.gamma_max)
                else:
                    base_gamma = float(self.gamma_max)
                    event_gamma = float(self.gamma_min)
                triggered = sev_for_gamma >= float(self.severity_threshold)
                if self.use_candidate and candidate_flag:
                    triggered = True
                if self.use_drift_flag and drift_flag:
                    triggered = True
                if self.severity_threshold_off is None:
                    if triggered:
                        cooldown_remaining = max(cooldown_remaining, self.cooldown_steps)
                    if cooldown_remaining > 0:
                        ema_decay = event_gamma
                        cooldown_remaining -= 1
                    else:
                        ema_decay = base_gamma
                else:
                    if self._event_active:
                        if cooldown_remaining > 0:
                            ema_decay = event_gamma
                            cooldown_remaining -= 1
                        else:
                            if sev_for_gamma < float(self.severity_threshold_off):
                                self._event_active = False
                                ema_decay = base_gamma
                            else:
                                ema_decay = event_gamma
                    else:
                        if triggered:
                            self._event_active = True
                            cooldown_remaining = max(cooldown_remaining, self.cooldown_steps)
                            ema_decay = event_gamma
                        else:
                            ema_decay = base_gamma
            ema_decay = _clamp_alpha(float(ema_decay))

        self._cooldown_remaining = int(cooldown_remaining)

        overrides: Dict[str, float] = {}
        if self.mode != "none":
            overrides["alpha"] = float(ema_decay)

        logs: Dict[str, float | int] = {
            "ema_decay": float(ema_decay),
            "ema_severity_raw": float(severity_raw),
            "ema_severity_norm": float(severity_norm),
            "ema_severity_ema": float(severity_ema),
            "ema_severity_threshold": float(self.severity_threshold),
            "ema_severity_threshold_off": float(self.severity_threshold_off) if self.severity_threshold_off is not None else float("nan"),
            "ema_cooldown_remaining": int(cooldown_remaining),
            "ema_triggered": int(bool(triggered)),
            "ema_event_active": int(bool(self._event_active)),
        }
        return EmaDecayStep(overrides=overrides, logs=logs)
