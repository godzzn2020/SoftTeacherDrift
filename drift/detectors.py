"""漂移检测封装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import math

from river import drift


def get_drift_flag(detector: drift.base.DriftDetector) -> bool:
    """
    统一从 river 检测器读取是否触发漂移，保持与 offline sweep 脚本一致。
    """
    if hasattr(detector, "drift_detected"):
        return bool(getattr(detector, "drift_detected"))
    if hasattr(detector, "change_detected"):
        return bool(getattr(detector, "change_detected"))
    return False


def _resolve_signal(values: Dict[str, float], key: str) -> float:
    """根据别名获取信号值。"""
    aliases = SIGNAL_ALIASES.get(key, (key,))
    for candidate in aliases:
        if candidate in values:
            value = values[candidate]
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            return value
    return float("nan")


def _ph_detector(**kwargs: float) -> drift.PageHinkley:
    """创建 PageHinkley 实例。"""
    return drift.PageHinkley(**kwargs)


def _error_signal_detector() -> Dict[str, drift.base.DriftDetector]:
    return {
        "error_rate": _ph_detector(delta=0.005, alpha=0.15, threshold=0.2, min_instances=25),
    }


def _entropy_signal_detector() -> Dict[str, drift.base.DriftDetector]:
    return {
        "teacher_entropy": _ph_detector(delta=0.01, alpha=0.3, threshold=0.5, min_instances=20),
    }


def _divergence_signal_detector() -> Dict[str, drift.base.DriftDetector]:
    return {
        "divergence": _ph_detector(delta=0.005, alpha=0.1, threshold=0.05, min_instances=30),
    }


def _merge_detectors(*builders: Callable[[], Dict[str, drift.base.DriftDetector]]) -> Dict[str, drift.base.DriftDetector]:
    detectors: Dict[str, drift.base.DriftDetector] = {}
    for builder in builders:
        detectors.update(builder())
    return detectors


def _build_error_only_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _error_signal_detector()


def _build_entropy_only_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _entropy_signal_detector()


def _build_divergence_only_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _divergence_signal_detector()


def _build_error_entropy_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _merge_detectors(_error_signal_detector, _entropy_signal_detector)


def _build_error_divergence_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _merge_detectors(_error_signal_detector, _divergence_signal_detector)


def _build_entropy_divergence_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _merge_detectors(_entropy_signal_detector, _divergence_signal_detector)


def _build_all_signals_ph_meta() -> Dict[str, drift.base.DriftDetector]:
    return _merge_detectors(
        _error_signal_detector,
        _entropy_signal_detector,
        _divergence_signal_detector,
    )


MONITOR_PRESETS: Dict[str, Callable[[], Dict[str, drift.base.DriftDetector]]] = {
    "error_only_ph_meta": _build_error_only_ph_meta,
    "entropy_only_ph_meta": _build_entropy_only_ph_meta,
    "divergence_only_ph_meta": _build_divergence_only_ph_meta,
    "error_entropy_ph_meta": _build_error_entropy_ph_meta,
    "error_divergence_ph_meta": _build_error_divergence_ph_meta,
    "entropy_divergence_ph_meta": _build_entropy_divergence_ph_meta,
    "all_signals_ph_meta": _build_all_signals_ph_meta,
    # 兼容旧名称
    "error_ph_meta": _build_error_only_ph_meta,
    "divergence_ph_meta": _build_divergence_only_ph_meta,
}

SIGNAL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "error_rate": ("error_rate", "student_error_rate"),
    "divergence": ("divergence", "divergence_js"),
    "teacher_entropy": ("teacher_entropy",),
}


@dataclass
class DriftMonitor:
    """同时监控多个指标的漂移检测器集合。"""

    detectors: Dict[str, drift.base.DriftDetector]
    trigger_mode: str = "or"
    trigger_k: int = 2
    trigger_weights: Optional[Dict[str, float]] = None
    trigger_threshold: float = 0.5
    confirm_window: int = 200
    history: List[int] = field(default_factory=list)
    last_drift_step: int = -1

    def __post_init__(self) -> None:
        self._prev_values_on_drift: Dict[str, float] = {}
        self._prev_values_all: Dict[str, float] = {}
        self.last_vote_count: int = 0
        self.last_vote_score: float = 0.0
        self.last_fused_severity: float = 0.0
        # two-stage state
        self._pending_candidate_step: Optional[int] = None
        self._pending_confirm_deadline_step: Optional[int] = None
        self.last_candidate_flag: bool = False
        self.last_confirm_delay: int = -1
        self.candidate_history: List[int] = []
        self.confirmed_history: List[int] = []
        self.candidate_count_total: int = 0
        self.confirmed_count_total: int = 0

    def update(self, values: Dict[str, float], step: int) -> Tuple[bool, float]:
        """输入最新批次统计，返回漂移标志与严重度。"""
        trigger_mode = (self.trigger_mode or "or").lower()
        if trigger_mode not in {"or", "k_of_n", "weighted", "two_stage"}:
            trigger_mode = "or"
        k = int(self.trigger_k) if self.trigger_k is not None else 2
        k = max(1, k)
        weights = self.trigger_weights or {"error_rate": 0.5, "divergence": 0.3, "teacher_entropy": 0.2}
        threshold = float(self.trigger_threshold) if self.trigger_threshold is not None else 0.5
        confirm_window = max(1, int(self.confirm_window))

        monitor_severity = 0.0
        fused_severity = 0.0
        vote_count = 0
        vote_score = 0.0

        for name, detector in self.detectors.items():
            value = _resolve_signal(values, name)
            if isinstance(value, float) and math.isnan(value):
                continue
            delta_all = abs(value - self._prev_values_all.get(name, value))
            fused_severity = max(fused_severity, delta_all)
            self._prev_values_all[name] = value
            detector.update(value)
            drifted = get_drift_flag(detector)
            if drifted:
                vote_count += 1
                vote_score += float(weights.get(name, 1.0))
                delta_on_drift = abs(value - self._prev_values_on_drift.get(name, value))
                monitor_severity = max(monitor_severity, delta_on_drift)
                self._prev_values_on_drift[name] = value

        candidate_flag = vote_count >= 1
        drift_flag = False
        confirm_delay = -1

        if trigger_mode == "or":
            drift_flag = candidate_flag
        elif trigger_mode == "k_of_n":
            drift_flag = vote_count >= k
        elif trigger_mode == "weighted":
            drift_flag = vote_score >= threshold
        else:  # two_stage: candidate OR -> confirm weighted within window
            # clear expired pending
            if self._pending_confirm_deadline_step is not None and step > self._pending_confirm_deadline_step:
                self._pending_candidate_step = None
                self._pending_confirm_deadline_step = None
            # register candidate
            if candidate_flag and self._pending_candidate_step is None:
                self._pending_candidate_step = step
                self._pending_confirm_deadline_step = step + confirm_window
                self.candidate_history.append(step)
                self.candidate_count_total += 1
            # confirm
            if self._pending_candidate_step is not None and vote_score >= threshold:
                drift_flag = True
                confirm_delay = max(0, step - self._pending_candidate_step)
                self._pending_candidate_step = None
                self._pending_confirm_deadline_step = None
                self.confirmed_history.append(step)
                self.confirmed_count_total += 1

        self.last_vote_count = vote_count
        self.last_vote_score = float(vote_score)
        self.last_fused_severity = float(fused_severity)
        self.last_candidate_flag = bool(candidate_flag)
        self.last_confirm_delay = int(confirm_delay)

        if drift_flag:
            self.history.append(step)
            self.last_drift_step = step
        return drift_flag, monitor_severity


def build_default_monitor(
    preset: str = "error_ph_meta",
    *,
    trigger_mode: str = "or",
    trigger_k: int = 2,
    trigger_weights: Optional[Dict[str, float]] = None,
    trigger_threshold: float = 0.5,
    confirm_window: int = 200,
) -> DriftMonitor:
    """根据预设构建监控器。"""
    if preset == "none":
        return DriftMonitor(
            detectors={},
            trigger_mode=trigger_mode,
            trigger_k=trigger_k,
            trigger_weights=trigger_weights,
            trigger_threshold=trigger_threshold,
            confirm_window=confirm_window,
        )
    factory = MONITOR_PRESETS.get(preset)
    if factory is None:
        raise ValueError(f"未知 monitor preset: {preset}")
    detectors_dict = factory()
    return DriftMonitor(
        detectors=detectors_dict,
        trigger_mode=trigger_mode,
        trigger_k=trigger_k,
        trigger_weights=trigger_weights,
        trigger_threshold=trigger_threshold,
        confirm_window=confirm_window,
    )
