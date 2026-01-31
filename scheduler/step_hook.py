"""Unified per-step scheduler hook (EMA + loss adapt)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from scheduler.ema_adapt import EmaDecayScheduler, EmaDecayStep
from scheduler.loss_adapt import LossAdaptiveScheduler, LossAdaptStep


@dataclass
class StepSchedulerOutput:
    overrides: Dict[str, float]
    logs: Dict[str, Any]


class StepScheduler:
    """Single hook to run multiple schedulers."""

    def __init__(
        self,
        ema_scheduler: EmaDecayScheduler,
        loss_scheduler: LossAdaptiveScheduler,
    ) -> None:
        self.ema_scheduler = ema_scheduler
        self.loss_scheduler = loss_scheduler

    def step(
        self,
        signals: Dict[str, float],
        *,
        drift_flag: bool,
        candidate_flag: bool,
    ) -> StepSchedulerOutput:
        overrides: Dict[str, float] = {}
        logs: Dict[str, Any] = {}

        ema_out: EmaDecayStep = self.ema_scheduler.step(
            signals,
            drift_flag=drift_flag,
            candidate_flag=candidate_flag,
        )
        overrides.update(ema_out.overrides)
        logs.update(ema_out.logs)

        loss_out: LossAdaptStep = self.loss_scheduler.step(
            signals,
            drift_flag=drift_flag,
            candidate_flag=candidate_flag,
        )
        overrides.update(loss_out.overrides)
        logs.update(loss_out.logs)

        return StepSchedulerOutput(overrides=overrides, logs=logs)
