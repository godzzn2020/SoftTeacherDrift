"""漂移检测封装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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


PH_PARAM_KEYS = ("threshold", "delta", "alpha", "min_instances")

DEFAULT_PH_PARAMS: Dict[str, Dict[str, Any]] = {
    "error_rate": {"delta": 0.005, "alpha": 0.15, "threshold": 0.2, "min_instances": 25},
    "divergence": {"delta": 0.005, "alpha": 0.1, "threshold": 0.05, "min_instances": 30},
    "teacher_entropy": {"delta": 0.01, "alpha": 0.3, "threshold": 0.5, "min_instances": 20},
}


def _normalize_ph_overrides(overrides: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    if not overrides:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for signal, params in overrides.items():
        if signal not in DEFAULT_PH_PARAMS or not isinstance(params, dict):
            continue
        clean: Dict[str, Any] = {}
        for k, v in params.items():
            if k not in PH_PARAM_KEYS or v is None:
                continue
            if k == "min_instances":
                clean[k] = int(float(v))
            else:
                clean[k] = float(v)
        if clean:
            out[signal] = clean
    return out


def _parse_preset_inline_overrides(preset: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    支持通过 monitor_preset 字符串内联覆盖 PH 参数：
      - 语法：<base_preset>@error.threshold=0.1,error.min_instances=10,divergence.threshold=0.02
      - signal 前缀：error/divergence/entropy（分别映射到 error_rate/divergence/teacher_entropy）
      - 参数名：threshold/delta/alpha/min_instances
    """
    if "@" not in (preset or ""):
        return preset, {}
    base, spec = preset.split("@", 1)
    base = base.strip()
    spec = (spec or "").strip()
    if not spec:
        return base, {}
    signal_alias = {
        "error": "error_rate",
        "err": "error_rate",
        "divergence": "divergence",
        "div": "divergence",
        "entropy": "teacher_entropy",
        "ent": "teacher_entropy",
    }
    param_alias = {
        "threshold": "threshold",
        "delta": "delta",
        "alpha": "alpha",
        "min_instances": "min_instances",
        "min": "min_instances",
    }
    overrides: Dict[str, Dict[str, Any]] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        left, raw = token.split("=", 1)
        left = left.strip()
        raw = raw.strip()
        if not left or not raw:
            continue
        if "." in left:
            sig, param = left.split(".", 1)
        elif "_" in left:
            sig, param = left.split("_", 1)
        else:
            continue
        sig_key = signal_alias.get(sig.strip().lower())
        param_key = param_alias.get(param.strip().lower())
        if sig_key is None or param_key is None:
            continue
        overrides.setdefault(sig_key, {})[param_key] = raw
    return base, _normalize_ph_overrides(overrides)


def _build_detectors_with_params(
    preset: str,
    ph_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, drift.base.DriftDetector], Dict[str, Dict[str, Any]]]:
    if preset in {"error_ph_meta", "error_only_ph_meta"}:
        signals = ["error_rate"]
    elif preset == "entropy_only_ph_meta":
        signals = ["teacher_entropy"]
    elif preset in {"divergence_ph_meta", "divergence_only_ph_meta"}:
        signals = ["divergence"]
    elif preset == "error_entropy_ph_meta":
        signals = ["error_rate", "teacher_entropy"]
    elif preset == "error_divergence_ph_meta":
        signals = ["error_rate", "divergence"]
    elif preset == "entropy_divergence_ph_meta":
        signals = ["teacher_entropy", "divergence"]
    elif preset == "all_signals_ph_meta":
        signals = ["error_rate", "teacher_entropy", "divergence"]
    else:
        raise ValueError(f"未知 monitor preset: {preset}")

    overrides = _normalize_ph_overrides(ph_overrides)
    used: Dict[str, Dict[str, Any]] = {}
    detectors_dict: Dict[str, drift.base.DriftDetector] = {}
    for sig in signals:
        params: Dict[str, Any] = dict(DEFAULT_PH_PARAMS.get(sig, {}))
        params.update(overrides.get(sig, {}))
        used[sig] = dict(params)
        detectors_dict[sig] = _ph_detector(**params)  # type: ignore[arg-type]
    return detectors_dict, used


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
    # two_stage 的“确认冷却”，用于抑制过密 confirm（单位：sample_idx；若未提供 sample_idx 则退化为 step）
    confirm_cooldown: int = 0
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
        self._pending_confirm_hits: int = 0
        self.last_candidate_flag: bool = False
        self.last_confirm_delay: int = -1
        self.candidate_history: List[int] = []
        self.confirmed_history: List[int] = []
        self.candidate_count_total: int = 0
        self.confirmed_count_total: int = 0
        self._last_confirmed_pos: Optional[int] = None
        self.last_cooldown_active: bool = False
        self.last_cooldown_remaining: int = 0
        # confirm-rule / gating diagnostics
        self.last_confirm_rule: str = "weighted"
        self.last_confirm_k: int = 2
        self.last_confirm_hits: int = 0
        self.last_divergence_gate_enabled: bool = False
        self.last_divergence_gate_ok: bool = True
        self.last_divergence_gate_steps: int = 0
        self.last_divergence_gate_samples: int = 0
        self.last_divergence_gate_value_thr: float = 0.05
        self._recent_divergence_steps: List[int] = []
        self._recent_divergence_positions: List[int] = []
        self._recent_divergence_values: List[Tuple[int, int, float]] = []
        # adaptive cooldown state（按 sample_idx/step 时间轴）
        self._recent_confirm_positions: List[int] = []
        self._adaptive_current_cooldown: int = 0
        self.last_effective_confirm_cooldown: int = 0
        self.last_confirm_rate_per10k: float = 0.0
        self.last_adaptive_cooldown_enabled: bool = False

    def _resolve_confirm_cooldown(self) -> int:
        cooldown = int(getattr(self, "confirm_cooldown", 0) or 0)
        if cooldown > 0:
            return cooldown
        tw = getattr(self, "trigger_weights", None)
        if isinstance(tw, dict):
            for key in ("__confirm_cooldown", "confirm_cooldown", "cooldown"):
                if key not in tw:
                    continue
                try:
                    return max(0, int(float(tw[key])))  # type: ignore[arg-type]
                except Exception:
                    continue
        return 0

    def _resolve_adaptive_cfg(self) -> Dict[str, Any]:
        """
        从属性或 trigger_weights 解析自适应 cooldown 参数。

        约定（优先级：显式属性 > trigger_weights）：
          - adaptive_cooldown_enabled: 0/1
          - adaptive_window: 最近窗口长度（sample_idx）
          - adaptive_lower_per10k / adaptive_upper_per10k: 触发阈值（confirm_rate_per_10k）
          - adaptive_cooldown_low / adaptive_cooldown_high: 低/高 cooldown（sample_idx）
        """
        cfg: Dict[str, Any] = {
            "enabled": bool(getattr(self, "adaptive_cooldown_enabled", False)),
            "window": int(getattr(self, "adaptive_window", 10000) or 10000),
            "lower_per10k": float(getattr(self, "adaptive_lower_per10k", 10.0) or 10.0),
            "upper_per10k": float(getattr(self, "adaptive_upper_per10k", 25.0) or 25.0),
            "cooldown_low": int(getattr(self, "adaptive_cooldown_low", 200) or 200),
            "cooldown_high": int(getattr(self, "adaptive_cooldown_high", 500) or 500),
        }

        tw = getattr(self, "trigger_weights", None)
        if isinstance(tw, dict):
            def _get(keys: Tuple[str, ...]) -> Optional[float]:
                for k in keys:
                    if k not in tw:
                        continue
                    try:
                        return float(tw[k])  # type: ignore[arg-type]
                    except Exception:
                        return None
                return None

            enabled_v = _get(("adaptive_cooldown", "adaptive_cd", "adaptive"))
            if enabled_v is not None:
                cfg["enabled"] = bool(int(enabled_v))
            window_v = _get(("adaptive_window", "adaptive_win"))
            if window_v is not None:
                cfg["window"] = int(window_v)
            lower_v = _get(("adaptive_lower_per10k", "adaptive_lower"))
            if lower_v is not None:
                cfg["lower_per10k"] = float(lower_v)
            upper_v = _get(("adaptive_upper_per10k", "adaptive_upper"))
            if upper_v is not None:
                cfg["upper_per10k"] = float(upper_v)
            low_v = _get(("adaptive_cooldown_low", "adaptive_low"))
            if low_v is not None:
                cfg["cooldown_low"] = int(low_v)
            high_v = _get(("adaptive_cooldown_high", "adaptive_high"))
            if high_v is not None:
                cfg["cooldown_high"] = int(high_v)

        cfg["window"] = max(1, int(cfg["window"] or 1))
        cfg["cooldown_low"] = max(0, int(cfg["cooldown_low"] or 0))
        cfg["cooldown_high"] = max(0, int(cfg["cooldown_high"] or 0))
        cfg["lower_per10k"] = max(0.0, float(cfg["lower_per10k"] or 0.0))
        cfg["upper_per10k"] = max(float(cfg["lower_per10k"]), float(cfg["upper_per10k"] or 0.0))
        return cfg

    def _resolve_confirm_rule_cfg(self) -> Dict[str, Any]:
        """
        two_stage confirm-side 规则配置（仅用于 trigger_mode=two_stage）：

        通过 trigger_weights 透传（值为数值，便于 ExperimentConfig 记录）：
          - confirm_rule: 0=weighted（默认，vote_score>=threshold 立即确认）
                          1=k_of_n（在 confirm_window 内累计 hit>=k 才确认；hit 定义为 vote_score>=threshold）
                          2=weighted+error_gate（vote_score>=threshold 且 error_rate>=confirm_error_gate_thr 才确认）
          - confirm_k: k_of_n 的 k（默认 2）
          - confirm_error_gate_thr: error_gate 的阈值（仅 confirm_rule=2 生效）
        """
        tw = getattr(self, "trigger_weights", None)
        rule = 0.0
        k = 2.0
        err_gate_thr: Optional[float] = None
        if isinstance(tw, dict):
            try:
                if "confirm_rule" in tw:
                    rule = float(tw["confirm_rule"])  # type: ignore[arg-type]
            except Exception:
                rule = 0.0
            try:
                if "confirm_k" in tw:
                    k = float(tw["confirm_k"])  # type: ignore[arg-type]
            except Exception:
                k = 2.0
            for key in ("confirm_error_gate_thr", "confirm_err_gate_thr", "confirm_err_thr"):
                if key not in tw:
                    continue
                try:
                    err_gate_thr = float(tw[key])  # type: ignore[arg-type]
                    break
                except Exception:
                    continue
        rule_i = int(rule)
        if rule_i not in (0, 1, 2):
            rule_i = 0
        return {"rule": rule_i, "k": max(1, int(k)), "err_gate_thr": err_gate_thr}

    def _resolve_divergence_gate_cfg(self) -> Dict[str, Any]:
        """
        confirm-side divergence gate（仅用于 trigger_mode=two_stage）：

        通过 trigger_weights 透传：
          - divergence_gate: 0/1 是否启用
          - divergence_gate_steps: 在最近多少 step 内出现过 divergence drift 才允许 confirm（用于 L=window）
          - divergence_gate_samples: 在最近多少 sample_idx 内出现过 divergence drift 才允许 confirm（用于 L=500）
        """
        tw = getattr(self, "trigger_weights", None)
        enabled = 0.0
        steps = 0.0
        samples = 0.0
        if isinstance(tw, dict):
            try:
                if "divergence_gate" in tw:
                    enabled = float(tw["divergence_gate"])  # type: ignore[arg-type]
            except Exception:
                enabled = 0.0
            try:
                if "divergence_gate_steps" in tw:
                    steps = float(tw["divergence_gate_steps"])  # type: ignore[arg-type]
            except Exception:
                steps = 0.0
            try:
                if "divergence_gate_samples" in tw:
                    samples = float(tw["divergence_gate_samples"])  # type: ignore[arg-type]
            except Exception:
                samples = 0.0
        # value_thr 默认不启用：若用户显式提供阈值（通过 trigger_weights 透传），才使用“divergence 值门控”；
        # 否则按“divergence drift 触发历史”门控（更符合“divergence 出现过”的语义，也更稳健）。
        value_thr: Optional[float] = None
        if isinstance(tw, dict):
            for key in ("divergence_gate_value_thr", "divergence_gate_value", "div_gate_value"):
                if key not in tw:
                    continue
                try:
                    value_thr = float(tw[key])  # type: ignore[arg-type]
                    break
                except Exception:
                    continue
        return {
            "enabled": bool(int(enabled)),
            "steps": max(0, int(steps)),
            "samples": max(0, int(samples)),
            "value_thr": value_thr,
        }

    def _resolve_effective_cooldown(self, current_pos: int) -> Tuple[int, float, bool]:
        cfg = self._resolve_adaptive_cfg()
        enabled = bool(cfg.get("enabled"))
        if not enabled:
            return self._resolve_confirm_cooldown(), 0.0, False

        window = int(cfg["window"])
        # 维护最近窗口内 confirm 位置
        if self._recent_confirm_positions:
            cutoff = int(current_pos - window)
            # confirm 数量通常很小，这里用线性裁剪即可
            self._recent_confirm_positions = [x for x in self._recent_confirm_positions if int(x) >= cutoff]
        count = len(self._recent_confirm_positions)
        rate_per10k = (count * 10000.0) / float(window) if window > 0 else 0.0

        lower = float(cfg["lower_per10k"])
        upper = float(cfg["upper_per10k"])
        cd_low = int(cfg["cooldown_low"])
        cd_high = int(cfg["cooldown_high"])
        if rate_per10k > upper:
            cd = cd_high
        elif rate_per10k < lower:
            cd = cd_low
        else:
            cd = int(self._adaptive_current_cooldown or cd_low)
        self._adaptive_current_cooldown = int(cd)
        return int(cd), float(rate_per10k), True

    def update(self, values: Dict[str, float], step: int, sample_idx: Optional[int] = None) -> Tuple[bool, float]:
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
        divergence_drifted = False
        divergence_value: Optional[float] = None
        error_rate_value: Optional[float] = None

        for name, detector in self.detectors.items():
            value = _resolve_signal(values, name)
            if isinstance(value, float) and math.isnan(value):
                continue
            if str(name) == "divergence":
                divergence_value = float(value)
            if str(name) == "error_rate":
                error_rate_value = float(value)
            delta_all = abs(value - self._prev_values_all.get(name, value))
            fused_severity = max(fused_severity, delta_all)
            self._prev_values_all[name] = value
            detector.update(value)
            drifted = get_drift_flag(detector)
            if drifted:
                if str(name) == "divergence":
                    divergence_drifted = True
                vote_count += 1
                vote_score += float(weights.get(name, 1.0))
                delta_on_drift = abs(value - self._prev_values_on_drift.get(name, value))
                monitor_severity = max(monitor_severity, delta_on_drift)
                self._prev_values_on_drift[name] = value

        candidate_flag = vote_count >= 1
        drift_flag = False
        confirm_delay = -1
        current_pos = int(sample_idx) if sample_idx is not None else int(step)

        # 维护 divergence 触发历史（用于 confirm-side gating）
        if divergence_drifted:
            self._recent_divergence_steps.append(int(step))
            self._recent_divergence_positions.append(int(current_pos))
        if divergence_value is not None and not math.isnan(float(divergence_value)):
            self._recent_divergence_values.append((int(step), int(current_pos), float(divergence_value)))
        # 线性裁剪，避免列表无限增长（divergence 触发通常较少）
        if self._recent_divergence_steps:
            self._recent_divergence_steps = self._recent_divergence_steps[-512:]
        if self._recent_divergence_positions:
            self._recent_divergence_positions = self._recent_divergence_positions[-512:]
        if self._recent_divergence_values:
            self._recent_divergence_values = self._recent_divergence_values[-2048:]

        cooldown, rate_per10k, adaptive_enabled = self._resolve_effective_cooldown(current_pos)
        cooldown_active = False
        cooldown_remaining = 0
        if cooldown > 0 and self._last_confirmed_pos is not None:
            gap = int(current_pos - int(self._last_confirmed_pos))
            if gap < int(cooldown):
                cooldown_active = True
                cooldown_remaining = int(cooldown - gap)

        if cooldown_active:
            # cooldown 期间不允许新 confirm，且清空 pending，避免“过期后补确认”造成不必要的晚检。
            self._pending_candidate_step = None
            self._pending_confirm_deadline_step = None
            self._pending_confirm_hits = 0
        else:
            if trigger_mode == "or":
                drift_flag = candidate_flag
            elif trigger_mode == "k_of_n":
                drift_flag = vote_count >= k
            elif trigger_mode == "weighted":
                drift_flag = vote_score >= threshold
            else:  # two_stage: candidate OR -> confirm weighted within window
                confirm_cfg = self._resolve_confirm_rule_cfg()
                confirm_rule = int(confirm_cfg["rule"])
                confirm_k = int(confirm_cfg["k"])
                err_gate_thr = confirm_cfg.get("err_gate_thr")
                gate_cfg = self._resolve_divergence_gate_cfg()
                gate_enabled = bool(gate_cfg["enabled"])
                gate_steps = int(gate_cfg["steps"])
                gate_samples = int(gate_cfg["samples"])
                gate_value_thr = gate_cfg.get("value_thr")
                gate_ok = True

                # clear expired pending
                if self._pending_confirm_deadline_step is not None and step > self._pending_confirm_deadline_step:
                    self._pending_candidate_step = None
                    self._pending_confirm_deadline_step = None
                    self._pending_confirm_hits = 0
                # register candidate
                if candidate_flag and self._pending_candidate_step is None:
                    self._pending_candidate_step = step
                    self._pending_confirm_deadline_step = step + confirm_window
                    self._pending_confirm_hits = 0
                    self.candidate_history.append(step)
                    self.candidate_count_total += 1
                # confirm (rule-dependent)
                confirm_hit = bool(vote_score >= threshold)
                # k_of_n confirm：hit 定义为 vote_score>=threshold（与 weighted 的阈值语义一致）
                k_of_n_hit = bool(confirm_hit)
                error_gate_ok = True
                if confirm_rule == 2:
                    if err_gate_thr is None:
                        error_gate_ok = False
                    elif error_rate_value is None or math.isnan(float(error_rate_value)):
                        error_gate_ok = False
                    else:
                        error_gate_ok = bool(float(error_rate_value) >= float(err_gate_thr))
                if self._pending_candidate_step is not None:
                    hit = confirm_hit if confirm_rule == 0 else k_of_n_hit
                    if hit:
                        self._pending_confirm_hits += 1

                    if gate_enabled:
                        # divergence gate：
                        # - 默认：按 divergence drift 触发历史门控（最近 L 内出现过 divergence drift 才允许 confirm）
                        # - 若显式提供 gate_value_thr：改为按“divergence 值在窗口内出现过（>=thr）”门控，并在失败时退化到 drift 历史。
                        if gate_samples > 0:
                            cutoff_pos = int(current_pos - gate_samples)
                            if gate_value_thr is not None:
                                gate_ok = any((pos >= cutoff_pos and val >= float(gate_value_thr)) for _, pos, val in self._recent_divergence_values)
                                if not gate_ok:
                                    gate_ok = any(int(x) >= cutoff_pos for x in self._recent_divergence_positions)
                            else:
                                gate_ok = any(int(x) >= cutoff_pos for x in self._recent_divergence_positions)
                        else:
                            steps_window = gate_steps if gate_steps > 0 else int(confirm_window)
                            cutoff_step = int(step - steps_window)
                            if gate_value_thr is not None:
                                gate_ok = any((st >= cutoff_step and val >= float(gate_value_thr)) for st, _, val in self._recent_divergence_values)
                                if not gate_ok:
                                    gate_ok = any(int(x) >= cutoff_step for x in self._recent_divergence_steps)
                            else:
                                gate_ok = any(int(x) >= cutoff_step for x in self._recent_divergence_steps)

                    should_confirm = False
                    if confirm_rule == 0:
                        should_confirm = bool(confirm_hit)
                    else:
                        should_confirm = bool(self._pending_confirm_hits >= confirm_k) if confirm_rule == 1 else bool(confirm_hit and error_gate_ok)

                    if should_confirm and gate_ok:
                        drift_flag = True
                        confirm_delay = max(0, step - self._pending_candidate_step)
                        self._pending_candidate_step = None
                        self._pending_confirm_deadline_step = None
                        self._pending_confirm_hits = 0
                        self.confirmed_history.append(step)
                        self.confirmed_count_total += 1

                # diagnostics
                if confirm_rule == 0:
                    self.last_confirm_rule = "weighted"
                elif confirm_rule == 1:
                    self.last_confirm_rule = "k_of_n"
                else:
                    self.last_confirm_rule = "weighted_error_gate"
                self.last_confirm_k = int(confirm_k)
                self.last_confirm_hits = int(self._pending_confirm_hits)
                self.last_divergence_gate_enabled = bool(gate_enabled)
                self.last_divergence_gate_ok = bool(gate_ok)
                self.last_divergence_gate_steps = int(gate_steps)
                self.last_divergence_gate_samples = int(gate_samples)
                self.last_divergence_gate_value_thr = float(gate_value_thr) if gate_value_thr is not None else float("nan")

        self.last_vote_count = vote_count
        self.last_vote_score = float(vote_score)
        self.last_fused_severity = float(fused_severity)
        self.last_candidate_flag = bool(candidate_flag)
        self.last_confirm_delay = int(confirm_delay)
        self.last_cooldown_active = bool(cooldown_active)
        self.last_cooldown_remaining = int(cooldown_remaining)
        self.last_effective_confirm_cooldown = int(cooldown)
        self.last_confirm_rate_per10k = float(rate_per10k)
        self.last_adaptive_cooldown_enabled = bool(adaptive_enabled)

        if drift_flag:
            self.history.append(step)
            self.last_drift_step = step
            self._last_confirmed_pos = int(current_pos)
            self._recent_confirm_positions.append(int(current_pos))
        return drift_flag, monitor_severity


def build_default_monitor(
    preset: str = "error_ph_meta",
    *,
    trigger_mode: str = "or",
    trigger_k: int = 2,
    trigger_weights: Optional[Dict[str, float]] = None,
    trigger_threshold: float = 0.5,
    confirm_window: int = 200,
    ph_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
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
    base_preset, inline_overrides = _parse_preset_inline_overrides(preset)
    merged_overrides = _normalize_ph_overrides(ph_overrides)
    for sig, params in inline_overrides.items():
        merged_overrides.setdefault(sig, {}).update(params)
    detectors_dict, used_params = _build_detectors_with_params(base_preset, merged_overrides)
    monitor = DriftMonitor(
        detectors=detectors_dict,
        trigger_mode=trigger_mode,
        trigger_k=trigger_k,
        trigger_weights=trigger_weights,
        trigger_threshold=trigger_threshold,
        confirm_window=confirm_window,
    )
    setattr(monitor, "preset_base", base_preset)
    setattr(monitor, "ph_overrides", merged_overrides)
    setattr(monitor, "ph_params", used_params)
    return monitor
