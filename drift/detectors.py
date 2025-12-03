"""漂移检测封装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

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
    history: List[int] = field(default_factory=list)
    last_drift_step: int = -1

    def __post_init__(self) -> None:
        self._prev_values: Dict[str, float] = {}

    def update(self, values: Dict[str, float], step: int) -> Tuple[bool, float]:
        """输入最新批次统计，返回漂移标志与严重度。"""
        drift_flag = False
        severity = 0.0
        for name, detector in self.detectors.items():
            value = _resolve_signal(values, name)
            if isinstance(value, float) and math.isnan(value):
                continue
            detector.update(value)
            if get_drift_flag(detector):
                drift_flag = True
                delta = abs(value - self._prev_values.get(name, value))
                severity = max(severity, delta)
                self._prev_values[name] = value
        if drift_flag:
            self.history.append(step)
            self.last_drift_step = step
        return drift_flag, severity


def build_default_monitor(preset: str = "error_ph_meta") -> DriftMonitor:
    """根据预设构建监控器。"""
    if preset == "none":
        return DriftMonitor(detectors={})
    factory = MONITOR_PRESETS.get(preset)
    if factory is None:
        raise ValueError(f"未知 monitor preset: {preset}")
    detectors_dict = factory()
    return DriftMonitor(detectors=detectors_dict)
