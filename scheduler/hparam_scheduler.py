"""漂移触发的超参调度器。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class HParams:
    """训练关键超参数。"""

    alpha: float
    lr: float
    lambda_u: float
    tau: float


@dataclass
class SchedulerState:
    """调度状态。"""

    base_hparams: HParams
    step: int = 0
    last_drift_step: int = -1

    def time_since_drift(self) -> int:
        if self.last_drift_step < 0:
            return self.step
        return self.step - self.last_drift_step


def update_hparams(
    state: SchedulerState,
    prev_hparams: HParams,
    drift_flag: bool,
    severity: float,
) -> HParams:
    """根据漂移信号更新超参数。"""
    mild_threshold, severe_threshold = 0.05, 0.15
    if math.isnan(severity):
        severity = 0.0
    regime = "stable"
    if drift_flag:
        state.last_drift_step = state.step
        if severity >= severe_threshold:
            regime = "severe_drift"
        elif severity >= mild_threshold:
            regime = "mild_drift"
    else:
        # 若长时间无漂移，则回归稳定设置
        if state.time_since_drift() > 20:
            regime = "stable"
        else:
            regime = "mild_drift"
    target = _regime_target(regime, state.base_hparams)
    return HParams(
        alpha=_blend(prev_hparams.alpha, target.alpha, 0.5),
        lr=_blend(prev_hparams.lr, target.lr, 0.5),
        lambda_u=_blend(prev_hparams.lambda_u, target.lambda_u, 0.5),
        tau=_blend(prev_hparams.tau, target.tau, 0.5),
    )


def _regime_target(regime: str, base: HParams) -> HParams:
    """根据策略返回目标超参。"""
    if regime == "stable":
        return HParams(
            alpha=min(0.999, max(base.alpha, 0.99)),
            lr=max(base.lr * 0.8, 1e-5),
            lambda_u=base.lambda_u,
            tau=min(0.99, base.tau * 1.05),
        )
    if regime == "mild_drift":
        return HParams(
            alpha=max(0.95, base.alpha * 0.97),
            lr=base.lr * 1.3,
            lambda_u=max(0.1, base.lambda_u * 0.8),
            tau=max(0.5, base.tau * 0.95),
        )
    return HParams(
        alpha=max(0.85, base.alpha * 0.9),
        lr=base.lr * 2.0,
        lambda_u=max(0.05, base.lambda_u * 0.5),
        tau=max(0.3, base.tau * 0.9),
    )


def _blend(prev: float, target: float, weight: float) -> float:
    weight = min(max(weight, 0.0), 1.0)
    return (1 - weight) * prev + weight * target
