"""在线漂移严重度校准工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SeverityStats:
    """用于跟踪特征的在线均值与方差。"""

    mean: float = 0.0
    var: float = 1.0


class SeverityCalibrator:
    """基于三信号（误差/散度/熵）估计漂移严重度。"""

    def __init__(
        self,
        ema_momentum: float = 0.99,
        eps: float = 1e-6,
        severity_low: float = 0.0,
        severity_high: float = 2.0,
        weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
    ) -> None:
        self.m = ema_momentum
        self.eps = eps
        self.severity_low = severity_low
        self.severity_high = max(severity_high, severity_low + 1e-6)
        self.weights = weights
        self.baseline_error: Optional[float] = None
        self.baseline_div: Optional[float] = None
        self.baseline_entropy: Optional[float] = None

        self.mean_error = 0.0
        self.mean_div = 0.0
        self.mean_entropy = 0.0
        self.var_error = 1.0
        self.var_div = 1.0
        self.var_entropy = 1.0
        self.stats_initialized = False

    def update_baselines(self, err: float, div: float, entropy: float) -> None:
        """通过 EMA 更新三个基线。"""
        if self.baseline_error is None:
            self.baseline_error = err
            self.baseline_div = div
            self.baseline_entropy = entropy
            return
        m = self.m
        self.baseline_error = m * self.baseline_error + (1 - m) * err
        self.baseline_div = m * self.baseline_div + (1 - m) * div
        self.baseline_entropy = m * self.baseline_entropy + (1 - m) * entropy

    def _update_stats(self, x_err: float, x_div: float, x_ent: float) -> None:
        """更新正向特征的在线均值与方差。"""
        if not self.stats_initialized:
            self.mean_error = x_err
            self.mean_div = x_div
            self.mean_entropy = x_ent
            self.var_error = 1.0
            self.var_div = 1.0
            self.var_entropy = 1.0
            self.stats_initialized = True
            return
        m = self.m
        self.mean_error = m * self.mean_error + (1 - m) * x_err
        self.mean_div = m * self.mean_div + (1 - m) * x_div
        self.mean_entropy = m * self.mean_entropy + (1 - m) * x_ent
        self.var_error = m * self.var_error + (1 - m) * (x_err - self.mean_error) ** 2
        self.var_div = m * self.var_div + (1 - m) * (x_div - self.mean_div) ** 2
        self.var_entropy = m * self.var_entropy + (1 - m) * (x_ent - self.mean_entropy) ** 2

    def compute_severity(self, err: float, div: float, entropy: float) -> Tuple[float, float]:
        """返回 (severity_raw, severity_norm)。"""
        if self.baseline_error is None:
            return 0.0, 0.0

        delta_error = err - self.baseline_error
        delta_div = div - self.baseline_div
        delta_entropy = entropy - self.baseline_entropy

        x_error_pos = max(0.0, delta_error)
        x_div_pos = max(0.0, delta_div)
        x_entropy_pos = max(0.0, -delta_entropy)

        if x_error_pos <= 0 and x_div_pos <= 0 and x_entropy_pos <= 0:
            return 0.0, 0.0

        self._update_stats(x_error_pos, x_div_pos, x_entropy_pos)
        if not self.stats_initialized:
            return 0.0, 0.0

        std_error = (self.var_error + self.eps) ** 0.5
        std_div = (self.var_div + self.eps) ** 0.5
        std_entropy = (self.var_entropy + self.eps) ** 0.5

        z_error = (x_error_pos - self.mean_error) / std_error
        z_div = (x_div_pos - self.mean_div) / std_div
        z_entropy = (x_entropy_pos - self.mean_entropy) / std_entropy

        w_error, w_div, w_entropy = self.weights
        severity_raw = w_error * z_error + w_div * z_div + w_entropy * z_entropy
        s_norm = (severity_raw - self.severity_low) / (self.severity_high - self.severity_low)
        s_norm = max(0.0, min(1.0, s_norm))
        return severity_raw, s_norm

