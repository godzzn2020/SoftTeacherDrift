"""漂移检测封装。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import math

import numpy as np
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

_CANDIDATE_SIGNAL_ALIASES: Dict[str, str] = {
    "error": "error_rate",
    "err": "error_rate",
    "error_rate": "error_rate",
    "student_error_rate": "error_rate",
    "div": "divergence",
    "divergence": "divergence",
    "divergence_js": "divergence",
    "entropy": "teacher_entropy",
    "teacher_entropy": "teacher_entropy",
    "ent": "teacher_entropy",
}


def _normalize_candidate_signals(value: Any) -> Optional[set[str]]:
    if value is None:
        return None
    tokens: List[str]
    if isinstance(value, (list, tuple, set)):
        tokens = [str(v).strip() for v in value if str(v).strip()]
    else:
        tokens = [t.strip() for t in str(value).split(",") if t.strip()]
    if not tokens:
        return None
    out: set[str] = set()
    for token in tokens:
        key = _CANDIDATE_SIGNAL_ALIASES.get(token.lower())
        if key:
            out.add(key)
    return out or None


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
    # two_stage 的 confirm 规则（默认 weighted；perm_test 为新增的置换检验 confirm）
    confirm_rule: str = "weighted"
    # permutation-test confirm 的默认参数（可通过 trigger_weights 的 __perm_* 覆盖）
    perm_pre_n: int = 200
    perm_post_n: int = 50
    perm_n_perm: int = 200
    perm_alpha: float = 0.01
    perm_stat: str = "fused_score"  # fused_score / delta_fused_score / vote_score
    perm_min_effect: float = 0.0
    perm_rng_seed: int = 0
    perm_delta_k: int = 50
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

        # permutation-test confirm（two_stage confirm_rule=perm_test）
        self.last_perm_pvalue: float = float("nan")
        # 口径：one_sided_pos -> signed(post.mean-pre.mean)；two_sided/abs_one_sided -> abs(post.mean-pre.mean)
        self.last_perm_effect: float = float("nan")
        self.perm_test_count_total: int = 0
        self.perm_accept_count_total: int = 0
        self.perm_reject_count_total: int = 0
        self._pending_confirm_rule_name: Optional[str] = None
        self._perm_pre_seq: List[float] = []
        self._perm_post_seq: List[float] = []
        self._perm_fused_score_hist = deque(maxlen=4096)
        self._perm_stat_hist = deque(maxlen=4096)
        self._perm_rng: Optional[np.random.Generator] = None
        self._perm_rng_seed_current: Optional[int] = None
        self._perm_prev_pos: Optional[int] = None
        self._perm_pvalues: List[float] = []
        self._perm_effects: List[float] = []

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

    def _clear_pending_state(self) -> None:
        self._pending_candidate_step = None
        self._pending_confirm_deadline_step = None
        self._pending_confirm_hits = 0
        self._pending_confirm_rule_name = None
        self._perm_pre_seq = []
        self._perm_post_seq = []

    def _resolve_confirm_rule_name(self) -> str:
        # two_stage confirm 规则选择：
        # - 优先：trigger_weights["__confirm_rule"]（字符串，如 "perm_test"）
        # - 次选：显式设置的 self.confirm_rule（仅当不为默认 "weighted" 时生效，避免覆盖旧的数值 confirm_rule）
        # - 兜底：兼容旧实现的数值 confirm_rule（0/1/2）
        tw = getattr(self, "trigger_weights", None)
        if isinstance(tw, dict):
            for key in ("__confirm_rule", "__confirm_rule_name", "confirm_rule_name", "confirm_rule_str"):
                if key not in tw:
                    continue
                v = tw.get(key)
                name = str(v).strip().lower() if v is not None else ""
                if name:
                    return name
        attr_rule = str(getattr(self, "confirm_rule", "") or "").strip().lower()
        if attr_rule and attr_rule != "weighted":
            return attr_rule
        cfg = self._resolve_confirm_rule_cfg()
        rule_i = int(cfg.get("rule", 0))
        if rule_i == 1:
            return "k_of_n"
        if rule_i == 2:
            return "weighted_error_gate"
        return "weighted"

    def _resolve_perm_test_cfg(self) -> Dict[str, Any]:
        tw = getattr(self, "trigger_weights", None)

        def _int_from(keys: Tuple[str, ...], default: int) -> int:
            if isinstance(tw, dict):
                for k in keys:
                    if k not in tw:
                        continue
                    try:
                        return int(float(tw[k]))  # type: ignore[arg-type]
                    except Exception:
                        continue
            return int(default)

        def _float_from(keys: Tuple[str, ...], default: float) -> float:
            if isinstance(tw, dict):
                for k in keys:
                    if k not in tw:
                        continue
                    try:
                        return float(tw[k])  # type: ignore[arg-type]
                    except Exception:
                        continue
            return float(default)

        def _str_from(keys: Tuple[str, ...], default: str) -> str:
            if isinstance(tw, dict):
                for k in keys:
                    if k not in tw:
                        continue
                    v = tw.get(k)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s:
                        return s
            return str(default)

        pre_n = _int_from(("__perm_pre_n", "perm_pre_n"), int(getattr(self, "perm_pre_n", 200) or 200))
        post_n = _int_from(("__perm_post_n", "perm_post_n"), int(getattr(self, "perm_post_n", 50) or 50))
        n_perm = _int_from(("__perm_n_perm", "perm_n_perm"), int(getattr(self, "perm_n_perm", 200) or 200))
        alpha = _float_from(("__perm_alpha", "perm_alpha"), float(getattr(self, "perm_alpha", 0.01) or 0.01))
        stat = _str_from(("__perm_stat", "perm_stat"), str(getattr(self, "perm_stat", "fused_score") or "fused_score")).lower()
        side = _str_from(("__perm_side", "perm_side"), str(getattr(self, "perm_side", "one_sided_pos") or "one_sided_pos")).lower()
        min_eff = _float_from(("__perm_min_effect", "perm_min_effect"), float(getattr(self, "perm_min_effect", 0.0) or 0.0))
        rng_seed = _int_from(("__perm_rng_seed", "perm_rng_seed"), int(getattr(self, "perm_rng_seed", 0) or 0))
        delta_k = _int_from(("__perm_delta_k", "perm_delta_k"), int(getattr(self, "perm_delta_k", 50) or 50))

        pre_n = max(1, int(pre_n))
        post_n = max(1, int(post_n))
        n_perm = max(1, int(n_perm))
        alpha = float(alpha)
        if not (0.0 < alpha <= 1.0):
            alpha = 0.01
        if stat not in {"fused_score", "delta_fused_score", "vote_score"}:
            stat = "fused_score"
        if side not in {"one_sided_pos", "two_sided", "abs_one_sided"}:
            side = "one_sided_pos"
        min_eff = float(min_eff)
        delta_k = max(1, int(delta_k))
        return {
            "pre_n": pre_n,
            "post_n": post_n,
            "n_perm": n_perm,
            "alpha": alpha,
            "stat": stat,
            "side": side,
            "min_effect": min_eff,
            "rng_seed": int(rng_seed),
            "delta_k": delta_k,
        }

    def _ensure_perm_rng(self, seed: int) -> np.random.Generator:
        if self._perm_rng is None or self._perm_rng_seed_current is None or int(self._perm_rng_seed_current) != int(seed):
            self._perm_rng = np.random.default_rng(int(seed))
            self._perm_rng_seed_current = int(seed)
        return self._perm_rng

    def _perm_test_one_sided(self, pre_seq: List[float], post_seq: List[float], *, n_perm: int, seed: int) -> Tuple[float, float]:
        pre = np.asarray(list(pre_seq), dtype=np.float64)
        post = np.asarray(list(post_seq), dtype=np.float64)
        if pre.size <= 0 or post.size <= 0:
            return float("nan"), float("nan")
        obs = float(post.mean() - pre.mean())
        if not (obs > 0.0):
            return 1.0, float(obs)
        allv = np.concatenate([pre, post], axis=0)
        n_pre = int(pre.size)
        rng = np.random.default_rng(int(seed))
        ge = 0
        for _ in range(int(n_perm)):
            idx = rng.permutation(allv.size)
            pre_p = allv[idx[:n_pre]]
            post_p = allv[idx[n_pre:]]
            perm_obs = float(post_p.mean() - pre_p.mean())
            if perm_obs >= obs:
                ge += 1
        p = (1.0 + float(ge)) / (1.0 + float(int(n_perm)))
        return float(p), float(obs)

    def _perm_test(
        self,
        pre_seq: List[float],
        post_seq: List[float],
        *,
        n_perm: int,
        seed: int,
        side: str,
    ) -> Tuple[float, float]:
        pre = np.asarray(list(pre_seq), dtype=np.float64)
        post = np.asarray(list(post_seq), dtype=np.float64)
        if pre.size <= 0 or post.size <= 0:
            return float("nan"), float("nan")

        obs = float(post.mean() - pre.mean())
        side = str(side or "one_sided_pos").lower()
        if side not in {"one_sided_pos", "two_sided", "abs_one_sided"}:
            side = "one_sided_pos"

        if side == "one_sided_pos":
            return self._perm_test_one_sided(pre_seq, post_seq, n_perm=n_perm, seed=seed)

        allv = np.concatenate([pre, post], axis=0)
        n_pre = int(pre.size)
        rng = np.random.default_rng(int(seed))

        if side == "abs_one_sided":
            threshold = float(abs(obs))
            if not (threshold > 0.0):
                return 1.0, float(threshold)
            ge = 0
            for _ in range(int(n_perm)):
                idx = rng.permutation(allv.size)
                pre_p = allv[idx[:n_pre]]
                post_p = allv[idx[n_pre:]]
                perm_eff = float(abs(post_p.mean() - pre_p.mean()))
                if perm_eff >= threshold:
                    ge += 1
            p = (1.0 + float(ge)) / (1.0 + float(int(n_perm)))
            # effect 口径：two_sided/abs_one_sided 统一输出 abs(obs)
            return float(p), float(threshold)

        # two_sided
        threshold = float(abs(obs))
        if not (threshold > 0.0):
            return 1.0, float(threshold)
        ge = 0
        for _ in range(int(n_perm)):
            idx = rng.permutation(allv.size)
            pre_p = allv[idx[:n_pre]]
            post_p = allv[idx[n_pre:]]
            perm_eff = float(abs(post_p.mean() - pre_p.mean()))
            if perm_eff >= threshold:
                ge += 1
        p = (1.0 + float(ge)) / (1.0 + float(int(n_perm)))
        # effect 口径：two_sided/abs_one_sided 统一输出 abs(obs)
        return float(p), float(threshold)

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
        candidate_filter = None
        if isinstance(self.trigger_weights, dict):
            candidate_filter = _normalize_candidate_signals(
                self.trigger_weights.get("__candidate_signals")
                or self.trigger_weights.get("candidate_signals")
            )

        # --- perm_test: continuous stat ---
        # fused_score（perm_test 使用的连续统计量）：Σ_i w_i * ph_evidence_i
        # 其中 ph_evidence_i 来自 PageHinkley 内部累积量（max(_sum_increase,_sum_decrease)），非 0/1 drift_flag；
        # delta_fused_score：fused_score_t - median(fused_score_{t-k...t-1})（不包含当前 t）
        base_w = {"error_rate": 0.5, "divergence": 0.3, "teacher_entropy": 0.2}
        fused_score_t = 0.0
        # 说明：weights 与 detector 的 signal key 对齐；fused_score 仅对当前 preset 的 detectors 做融合。
        fused_keys = list(self.detectors.keys()) if self.detectors else ["error_rate", "divergence", "teacher_entropy"]
        for key in fused_keys:
            v = _resolve_signal(values, key)
            if isinstance(v, float) and math.isnan(v):
                continue
            w = float(weights.get(key, base_w.get(key, 1.0))) if isinstance(weights, dict) else float(base_w.get(key, 1.0))
            fused_score_t += w * float(v)

        # 默认 stat：fused_score（在 two_stage+perm_test 且 stat=delta_fused_score 时会在分支内改写）
        stat_t = float(fused_score_t)

        monitor_severity = 0.0
        fused_severity = 0.0
        vote_count = 0
        vote_score = 0.0
        divergence_drifted = False
        divergence_value: Optional[float] = None
        error_rate_value: Optional[float] = None

        ph_score = 0.0

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
            # perm_test: 使用 PH 内部累积统计量作为连续证据（避免直接用 0/1 drift_flag）
            try:
                inc = float(detector.__dict__.get("_sum_increase", 0.0) or 0.0)
                dec = float(detector.__dict__.get("_sum_decrease", 0.0) or 0.0)
                ev = max(inc, dec)
            except Exception:
                ev = 0.0
            try:
                ph_score += float(weights.get(name, 1.0)) * float(ev)
            except Exception:
                ph_score += float(ev)
            drifted = get_drift_flag(detector)
            if drifted:
                if str(name) == "divergence":
                    divergence_drifted = True
                use_for_candidate = True
                if candidate_filter is not None:
                    use_for_candidate = str(name) in candidate_filter
                if use_for_candidate:
                    vote_count += 1
                    vote_score += float(weights.get(name, 1.0))
                delta_on_drift = abs(value - self._prev_values_on_drift.get(name, value))
                monitor_severity = max(monitor_severity, delta_on_drift)
                self._prev_values_on_drift[name] = value

        fused_score_t = float(ph_score)
        stat_t = float(fused_score_t)

        candidate_flag = vote_count >= 1
        drift_flag = False
        confirm_delay = -1
        current_pos = int(sample_idx) if sample_idx is not None else int(step)

        # perm_test 的 pre/post_n 以 sample_idx 计数：用当前 batch 的样本数展开为样本级序列（每个样本复用本 batch 的 stat_t）
        prev_pos_for_perm = getattr(self, "_perm_prev_pos", None)
        batch_n = 1
        if prev_pos_for_perm is not None:
            try:
                batch_n = max(1, int(current_pos - int(prev_pos_for_perm)))
            except Exception:
                batch_n = 1
        self._perm_prev_pos = int(current_pos)

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
            self._clear_pending_state()
        else:
            if trigger_mode == "or":
                drift_flag = candidate_flag
            elif trigger_mode == "k_of_n":
                drift_flag = vote_count >= k
            elif trigger_mode == "weighted":
                drift_flag = vote_score >= threshold
            else:  # two_stage: candidate OR -> confirm weighted within window
                confirm_rule_name = self._resolve_confirm_rule_name()
                pending_rule_name = str(getattr(self, "_pending_confirm_rule_name", None) or confirm_rule_name or "weighted").lower()
                perm_enabled = pending_rule_name == "perm_test"
                perm_cfg = self._resolve_perm_test_cfg() if perm_enabled else None
                stat_t = float(fused_score_t)
                if perm_enabled and perm_cfg is not None:
                    perm_stat = str(perm_cfg.get("stat") or "fused_score").lower()
                    if perm_stat == "vote_score":
                        stat_t = float(vote_score)
                    elif perm_stat == "delta_fused_score":
                        delta_k = int(perm_cfg.get("delta_k") or 50)
                        prev_fused = list(self._perm_fused_score_hist)
                        tail = prev_fused[-max(1, min(delta_k, len(prev_fused))):] if prev_fused else []
                        baseline = float(np.median(np.asarray(tail, dtype=np.float64))) if tail else 0.0
                        stat_t = float(fused_score_t - baseline)

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
                    self._pending_confirm_rule_name = str(confirm_rule_name or "weighted").lower()
                    if str(self._pending_confirm_rule_name) == "perm_test":
                        perm_cfg_now = self._resolve_perm_test_cfg()
                        pre_n = int(perm_cfg_now.get("pre_n") or 200)
                        post_n = int(perm_cfg_now.get("post_n") or 50)
                        pre_hist = list(self._perm_stat_hist)
                        # 预窗口去污染：candidate 往往触发在“已接近/刚跨过阈值”的阶段，最近一小段可能已被 drift 污染。
                        # 默认跳过最近 post_n 个样本（若不足则退化为直接取最后 pre_n）。
                        if pre_hist and len(pre_hist) >= (pre_n + post_n):
                            self._perm_pre_seq = pre_hist[-(pre_n + post_n) : -post_n]
                        else:
                            self._perm_pre_seq = pre_hist[-pre_n:] if pre_hist else []
                        self._perm_post_seq = []
                    self.candidate_history.append(step)
                    self.candidate_count_total += 1
                # confirm (rule-dependent)
                confirm_hit = bool(vote_score >= threshold)
                perm_ok = True
                if perm_enabled and self._pending_candidate_step is not None and perm_cfg is not None:
                    # pending 期间收集 post_seq（从 candidate_step 起，包含当前 step）
                    self._perm_post_seq.extend([float(stat_t)] * int(batch_n))
                    if len(self._perm_post_seq) > 4096:
                        self._perm_post_seq = self._perm_post_seq[-4096:]
                    pre_n = int(perm_cfg.get("pre_n") or 200)
                    post_n = int(perm_cfg.get("post_n") or 50)
                    if len(self._perm_pre_seq) >= pre_n and len(self._perm_post_seq) >= post_n:
                        # 使用最近 post_n 的滑动窗口；达到 post_n 后每新增 1 个样本重算一次
                        pre_seq = list(self._perm_pre_seq[-pre_n:])
                        post_seq = list(self._perm_post_seq[-post_n:])
                        n_perm = int(perm_cfg.get("n_perm") or 200)
                        rng_seed = int(perm_cfg.get("rng_seed") or 0)
                        side = str(perm_cfg.get("side") or "one_sided_pos")
                        p, obs = self._perm_test(pre_seq, post_seq, n_perm=n_perm, seed=rng_seed, side=side)
                        self.last_perm_pvalue = float(p)
                        self.last_perm_effect = float(obs)
                        self.perm_test_count_total += 1
                        if len(self._perm_pvalues) < 10000 and not math.isnan(float(p)):
                            self._perm_pvalues.append(float(p))
                        if len(self._perm_effects) < 10000 and not math.isnan(float(obs)):
                            self._perm_effects.append(float(obs))
                        alpha = float(perm_cfg.get("alpha") or 0.01)
                        min_eff = float(perm_cfg.get("min_effect") or 0.0)
                        eff_for_min = float(obs) if str(side).lower() == "one_sided_pos" else float(abs(obs))
                        perm_ok = bool(float(p) <= float(alpha) and float(eff_for_min) >= float(min_eff))
                        if perm_ok:
                            self.perm_accept_count_total += 1
                        else:
                            self.perm_reject_count_total += 1
                    else:
                        perm_ok = False
                if perm_enabled:
                    confirm_hit = bool(perm_ok)
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
                if perm_enabled:
                    self.last_confirm_rule = "perm_test"
                elif confirm_rule == 0:
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

        # 更新 ring buffer：必须在 candidate 注册之后（避免 pre_seq 误包含当前 step）
        self._perm_fused_score_hist.extend([float(fused_score_t)] * int(batch_n))
        self._perm_stat_hist.extend([float(stat_t)] * int(batch_n))

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


def _quickcheck_perm_test_sides() -> None:
    m = DriftMonitor(detectors={})
    pre = [0.0] * 50
    post_small = [0.2] * 50
    post_big = [1.0] * 50
    n_perm = 200
    seed = 0

    p_small, e_small = m._perm_test(pre, post_small, n_perm=n_perm, seed=seed, side="one_sided_pos")
    p_big, e_big = m._perm_test(pre, post_big, n_perm=n_perm, seed=seed, side="one_sided_pos")
    assert 0.0 <= p_small <= 1.0 and 0.0 <= p_big <= 1.0
    assert e_small > 0 and e_big > 0
    assert p_big <= p_small, (p_small, p_big)
    assert abs(e_small - 0.2) < 1e-12 and abs(e_big - 1.0) < 1e-12

    p2, e2 = m._perm_test(pre, post_big, n_perm=n_perm, seed=seed, side="two_sided")
    pa, ea = m._perm_test(pre, post_big, n_perm=n_perm, seed=seed, side="abs_one_sided")
    assert 0.0 <= p2 <= 1.0 and 0.0 <= pa <= 1.0
    assert abs(e2 - 1.0) < 1e-12 and abs(ea - 1.0) < 1e-12
    assert abs(p2 - pa) < 1e-12, (p2, pa)

    # one_sided_pos：负效应直接返回 p=1（与历史行为一致）；two_sided/abs_one_sided：effect 取 abs(obs)
    p_neg, e_neg = m._perm_test(pre, [-1.0] * 50, n_perm=n_perm, seed=seed, side="one_sided_pos")
    p_neg2, e_neg2 = m._perm_test(pre, [-1.0] * 50, n_perm=n_perm, seed=seed, side="two_sided")
    assert abs(p_neg - 1.0) < 1e-12 and e_neg < 0
    assert 0.0 <= p_neg2 <= 1.0 and abs(e_neg2 - 1.0) < 1e-12

    print("[quickcheck] perm_side effect semantics OK")


if __name__ == "__main__":
    _quickcheck_perm_test_sides()
