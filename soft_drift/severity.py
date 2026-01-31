"""在线漂移严重度校准工具。"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
        entropy_mode: str = "overconfident",
        mode: str = "weighted",
    ) -> None:
        self.m = ema_momentum
        self.eps = eps
        self.severity_low = severity_low
        self.severity_high = max(severity_high, severity_low + 1e-6)
        self.weights = weights
        if entropy_mode not in {"overconfident", "uncertain", "abs"}:
            raise ValueError(f"未知 entropy_mode: {entropy_mode}（可选 overconfident/uncertain/abs）")
        self.entropy_mode = entropy_mode
        if mode not in {"weighted", "max"}:
            raise ValueError(f"未知 severity mode: {mode}（可选 weighted/max）")
        self.mode = mode
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
        self.stats_initialized_error = False
        self.stats_initialized_div = False
        self.stats_initialized_entropy = False

    @staticmethod
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

    def update_baselines(self, err: float, div: float, entropy: float) -> None:
        """通过 EMA 更新三个基线。"""
        err_v = self._clean_value(err)
        div_v = self._clean_value(div)
        ent_v = self._clean_value(entropy)
        if err_v is not None:
            if self.baseline_error is None:
                self.baseline_error = err_v
            else:
                self.baseline_error = self.m * self.baseline_error + (1 - self.m) * err_v
        if div_v is not None:
            if self.baseline_div is None:
                self.baseline_div = div_v
            else:
                self.baseline_div = self.m * self.baseline_div + (1 - self.m) * div_v
        if ent_v is not None:
            if self.baseline_entropy is None:
                self.baseline_entropy = ent_v
            else:
                self.baseline_entropy = self.m * self.baseline_entropy + (1 - self.m) * ent_v

    def _update_stats(self, x_err: Optional[float], x_div: Optional[float], x_ent: Optional[float]) -> None:
        """更新正向特征的在线均值与方差。"""
        m = self.m
        if x_err is not None:
            if not self.stats_initialized_error:
                self.mean_error = x_err
                self.var_error = 1.0
                self.stats_initialized_error = True
            else:
                self.mean_error = m * self.mean_error + (1 - m) * x_err
                self.var_error = m * self.var_error + (1 - m) * (x_err - self.mean_error) ** 2
        if x_div is not None:
            if not self.stats_initialized_div:
                self.mean_div = x_div
                self.var_div = 1.0
                self.stats_initialized_div = True
            else:
                self.mean_div = m * self.mean_div + (1 - m) * x_div
                self.var_div = m * self.var_div + (1 - m) * (x_div - self.mean_div) ** 2
        if x_ent is not None:
            if not self.stats_initialized_entropy:
                self.mean_entropy = x_ent
                self.var_entropy = 1.0
                self.stats_initialized_entropy = True
            else:
                self.mean_entropy = m * self.mean_entropy + (1 - m) * x_ent
                self.var_entropy = m * self.var_entropy + (1 - m) * (x_ent - self.mean_entropy) ** 2
        if self.stats_initialized_error or self.stats_initialized_div or self.stats_initialized_entropy:
            self.stats_initialized = True

    def compute_severity(self, err: float, div: float, entropy: float) -> Tuple[float, float]:
        """返回 (severity_raw, severity_norm)。"""
        err_v = self._clean_value(err)
        div_v = self._clean_value(div)
        ent_v = self._clean_value(entropy)

        delta_error = None
        if err_v is not None and self.baseline_error is not None:
            delta_error = err_v - self.baseline_error
        delta_div = None
        if div_v is not None and self.baseline_div is not None:
            delta_div = div_v - self.baseline_div
        delta_entropy = None
        if ent_v is not None and self.baseline_entropy is not None:
            delta_entropy = ent_v - self.baseline_entropy

        x_error_pos = max(0.0, delta_error) if delta_error is not None else None
        x_div_pos = max(0.0, delta_div) if delta_div is not None else None
        if self.entropy_mode == "overconfident":
            # 教师“更自信”（entropy 下降）视为 drift 风险
            x_entropy_pos = max(0.0, -delta_entropy) if delta_entropy is not None else None
        elif self.entropy_mode == "uncertain":
            # 教师“更不确定”（entropy 上升）也视为 drift 风险
            x_entropy_pos = max(0.0, delta_entropy) if delta_entropy is not None else None
        else:  # abs
            x_entropy_pos = abs(delta_entropy) if delta_entropy is not None else None

        active = [
            x for x in (x_error_pos, x_div_pos, x_entropy_pos) if x is not None and x > 0.0
        ]
        if not active:
            return 0.0, 0.0

        self._update_stats(x_error_pos, x_div_pos, x_entropy_pos)
        if not self.stats_initialized:
            return 0.0, 0.0

        z_values = []
        w_values = []

        if x_error_pos is not None and self.stats_initialized_error:
            std_error = (self.var_error + self.eps) ** 0.5
            z_error = (x_error_pos - self.mean_error) / std_error
            z_values.append(z_error)
            w_values.append(float(self.weights[0]))
        if x_div_pos is not None and self.stats_initialized_div:
            std_div = (self.var_div + self.eps) ** 0.5
            z_div = (x_div_pos - self.mean_div) / std_div
            z_values.append(z_div)
            w_values.append(float(self.weights[1]))
        if x_entropy_pos is not None and self.stats_initialized_entropy:
            std_entropy = (self.var_entropy + self.eps) ** 0.5
            z_entropy = (x_entropy_pos - self.mean_entropy) / std_entropy
            z_values.append(z_entropy)
            w_values.append(float(self.weights[2]))

        if not z_values:
            return 0.0, 0.0

        if self.mode == "max":
            severity_raw = float(max(z_values))
        else:
            weight_sum = sum(w for w in w_values if w > 0.0)
            if weight_sum <= 0.0:
                return 0.0, 0.0
            severity_raw = float(sum(w * z for w, z in zip(w_values, z_values)) / weight_sum)
        s_norm = (severity_raw - self.severity_low) / (self.severity_high - self.severity_low)
        s_norm = max(0.0, min(1.0, s_norm))
        return severity_raw, s_norm
