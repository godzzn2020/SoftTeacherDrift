"""漂移检测封装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import math

from river import drift


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
            value = values.get(name)
            if value is None or math.isnan(value):
                continue
            detector.update(value)
            if getattr(detector, "change_detected", False):
                drift_flag = True
                delta = abs(value - self._prev_values.get(name, value))
                severity = max(severity, delta)
                self._prev_values[name] = value
        if drift_flag:
            self.history.append(step)
            self.last_drift_step = step
        return drift_flag, severity


def build_default_monitor() -> DriftMonitor:
    """构建默认监控器，分别监控误差与散度。"""
    detectors = {
        "error_rate": drift.ADWIN(delta=0.002),
        "divergence": drift.PageHinkley(min_instances=30, delta=0.05, threshold=20),
    }
    return DriftMonitor(detectors=detectors)

