"""运行单次师生 EMA 半监督实验。"""

from __future__ import annotations

import argparse
from typing import List, Optional

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PARENT = ROOT_DIR.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import torch

from experiments.first_stage_experiments import (
    ExperimentConfig,
    run_experiment as run_ts_experiment,
)
from soft_drift.utils.run_paths import create_experiment_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-Student EMA 单次实验")
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=[
            "sea",
            "sine",
            "stagger",
            "hyperplane",
            "uspds_csv",
            "insects_river",
            "insects_real",
            "sea_saved",
            "hyperplane_saved",
            "synth_saved",
        ],
        help="数据集类型",
    )
    parser.add_argument("--dataset_name", required=True, help="数据集名称/变体")
    parser.add_argument("--csv_path", type=str, help="CSV 数据集路径（仅 uspds_csv）")
    parser.add_argument("--label_col", type=str, help="CSV 标签列名，缺省为最后一列")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="ts_drift_adapt",
        help="模型变体标识（baseline_student/mean_teacher/ts_drift_adapt 等）",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--initial_alpha", type=float, default=0.99)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="none",
        choices=[
            "none",
            "error_ph_meta",
            "error_only_ph_meta",
            "entropy_only_ph_meta",
            "divergence_ph_meta",
            "divergence_only_ph_meta",
            "error_entropy_ph_meta",
            "error_divergence_ph_meta",
            "entropy_divergence_ph_meta",
            "all_signals_ph_meta",
        ],
        help="选择漂移检测器预设，none 表示不启用",
    )
    parser.add_argument(
        "--signal_set",
        type=str,
        default=None,
        choices=["error", "proxy", "all"],
        help="信号组合（可选）：error=仅监督，proxy=entropy+divergence，all=三者；为空则沿用 monitor_preset",
    )
    parser.add_argument(
        "--trigger_mode",
        type=str,
        default="or",
        choices=["or", "k_of_n", "weighted", "two_stage"],
        help="多 detector 融合触发策略（默认 or）",
    )
    parser.add_argument(
        "--trigger_k",
        type=int,
        default=2,
        help="trigger_mode=k_of_n 时的 k（至少 k 个 detector 同时触发才报警）",
    )
    parser.add_argument(
        "--trigger_threshold",
        type=float,
        default=0.5,
        help="trigger_mode=weighted 时的阈值（vote_score >= threshold 触发）",
    )
    parser.add_argument(
        "--confirm_window",
        type=int,
        default=200,
        help="trigger_mode=two_stage 时的 confirm_window（候选触发后在该窗口内确认）",
    )
    parser.add_argument(
        "--trigger_weights",
        type=str,
        default="",
        help="trigger_mode=weighted 时的权重（形如 error_rate=0.5,divergence=0.3,teacher_entropy=0.2；空表示使用默认）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--severity_scheduler_scale",
        type=float,
        default=1.0,
        help="全局 severity-aware 调度缩放，0 关闭，1 默认，>1 更激进",
    )
    parser.add_argument(
        "--use_severity_v2",
        action="store_true",
        help="启用 Severity-Aware v2（漂移后 severity 以 carry+decay 形式持续影响调度）",
    )
    parser.add_argument(
        "--severity_gate",
        type=str,
        default="none",
        choices=["none", "confirmed_only", "confirmed_streak"],
        help="severity v2 的 gating：confirmed_only 表示仅在票分 >= trigger_threshold 的 drift 才更新 carry/freeze；confirmed_streak 表示需要连续满足该条件 >= severity_gate_min_streak",
    )
    parser.add_argument(
        "--severity_gate_min_streak",
        type=int,
        default=1,
        help="severity_gate=confirmed_streak 时的最小连续确认步数（>=1；1 等价于 confirmed_only）",
    )
    parser.add_argument(
        "--entropy_mode",
        type=str,
        default="overconfident",
        choices=["overconfident", "uncertain", "abs"],
        help="SeverityCalibrator 的 entropy 正向增量定义",
    )
    parser.add_argument(
        "--severity_decay",
        type=float,
        default=0.95,
        help="Severity-Aware v2 的 carry 衰减系数（0~1，越大影响越持久）",
    )
    parser.add_argument(
        "--freeze_baseline_steps",
        type=int,
        default=0,
        help="检测到漂移后冻结 SeverityCalibrator baseline 的步数（0 关闭）",
    )
    parser.add_argument(
        "--ema_decay_mode",
        type=str,
        default="none",
        choices=["none", "fixed", "severity_continuous", "severity_event", "severity_event_reverse"],
        help="EMA 衰减率调度模式（none/fixed/severity_continuous/severity_event/severity_event_reverse）",
    )
    parser.add_argument("--ema_gamma_min", type=float, default=0.95, help="EMA 衰减率下界")
    parser.add_argument("--ema_gamma_max", type=float, default=0.999, help="EMA 衰减率上界")
    parser.add_argument("--ema_gamma_fixed", type=float, default=None, help="EMA 固定衰减率（fixed 模式）")
    parser.add_argument(
        "--ema_severity_mode",
        type=str,
        default="max",
        choices=["max", "weighted"],
        help="severity 聚合方式（max/weighted）",
    )
    parser.add_argument(
        "--ema_severity_weights",
        type=str,
        default="",
        help="severity 权重（weighted 模式；格式 error=0.0,divergence=0.5,teacher_entropy=0.5）",
    )
    parser.add_argument("--ema_severity_smoothing", type=float, default=0.9, help="severity EMA 平滑系数")
    parser.add_argument("--ema_severity_threshold", type=float, default=0.6, help="事件触发阈值")
    parser.add_argument("--ema_severity_threshold_off", type=float, default=None, help="事件退出阈值（可选）")
    parser.add_argument("--ema_cooldown_steps", type=int, default=200, help="事件触发后保持低γ的步数")
    parser.add_argument("--ema_use_candidate", action="store_true", help="事件触发时是否使用 candidate_flag")
    parser.add_argument("--ema_use_drift_flag", action="store_true", help="事件触发时是否使用 drift_flag")
    parser.add_argument(
        "--loss_scheduler_mode",
        type=str,
        default="none",
        choices=["none", "fixed", "severity_continuous", "severity_event"],
        help="lambda/tau 调度模式（none/fixed/severity_continuous/severity_event）",
    )
    parser.add_argument("--loss_lambda_min", type=float, default=None)
    parser.add_argument("--loss_lambda_max", type=float, default=None)
    parser.add_argument("--loss_tau_min", type=float, default=None)
    parser.add_argument("--loss_tau_max", type=float, default=None)
    parser.add_argument("--loss_lambda_fixed", type=float, default=None)
    parser.add_argument("--loss_tau_fixed", type=float, default=None)
    parser.add_argument(
        "--loss_severity_mode",
        type=str,
        default="max",
        choices=["max", "weighted"],
        help="loss severity 聚合方式（max/weighted）",
    )
    parser.add_argument(
        "--loss_severity_weights",
        type=str,
        default="",
        help="loss severity 权重（weighted 模式；格式 error=0.0,divergence=0.5,teacher_entropy=0.5）",
    )
    parser.add_argument("--loss_severity_momentum", type=float, default=0.99)
    parser.add_argument("--loss_severity_smoothing", type=float, default=0.9)
    parser.add_argument("--loss_severity_low", type=float, default=0.0)
    parser.add_argument("--loss_severity_high", type=float, default=2.0)
    parser.add_argument("--loss_severity_eps", type=float, default=1e-6)
    parser.add_argument("--loss_severity_use_error", action="store_true")
    parser.add_argument("--loss_event_on", type=float, default=0.6)
    parser.add_argument("--loss_event_off", type=float, default=0.4)
    parser.add_argument("--loss_cooldown_steps", type=int, default=200)
    parser.add_argument("--loss_use_candidate", action="store_true")
    parser.add_argument("--loss_use_drift_flag", action="store_true")
    parser.add_argument("--loss_apply_lambda", type=int, default=1, choices=[0, 1])
    parser.add_argument("--loss_apply_tau", type=int, default=1, choices=[0, 1])
    parser.add_argument("--loss_safety_enabled", action="store_true")
    parser.add_argument("--loss_safety_use_candidate", action="store_true")
    parser.add_argument("--loss_safety_use_severity", action="store_true")
    parser.add_argument("--loss_safety_severity_threshold", type=float, default=0.6)
    parser.add_argument("--loss_safety_cooldown_steps", type=int, default=200)
    parser.add_argument("--loss_safety_tau", type=float, default=None)
    parser.add_argument("--loss_safety_lambda", type=float, default=None)
    parser.add_argument("--results_root", type=str, default="results", help="结果输出根目录")
    parser.add_argument("--logs_root", type=str, default="logs", help="日志根目录（用于自动路径时）")
    parser.add_argument("--experiment_name", type=str, default="run_experiment", help="实验名称前缀")
    parser.add_argument("--run_name", type=str, default=None, help="自定义 run 名称")
    parser.add_argument("--run_id", type=str, default=None, help="覆盖自动生成的 run_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log_path
    exp_run = None
    if not log_path:
        exp_run = create_experiment_run(
            experiment_name=args.experiment_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            run_name=args.run_name,
            run_id=args.run_id,
        )
        dataset_run = exp_run.prepare_dataset_run(args.dataset_name, args.model_variant, args.seed)
        log_path = str(dataset_run.log_csv_path())
        print(f"[run] {exp_run.describe()} log -> {log_path}")
    config = ExperimentConfig(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
        label_col=args.label_col,
        model_variant=args.model_variant,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        labeled_ratio=args.labeled_ratio,
        initial_alpha=args.initial_alpha,
        initial_lr=args.initial_lr,
        lambda_u=args.lambda_u,
        tau=args.tau,
        seed=args.seed,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        log_path=log_path,
        monitor_preset=args.monitor_preset,
        signal_set=args.signal_set,
        severity_scheduler_scale=args.severity_scheduler_scale,
        trigger_mode=args.trigger_mode,
        trigger_k=int(args.trigger_k),
        trigger_threshold=float(args.trigger_threshold),
        trigger_weights=parse_trigger_weights(args.trigger_weights),
        confirm_window=int(args.confirm_window),
        use_severity_v2=bool(args.use_severity_v2),
        severity_gate=str(args.severity_gate),
        severity_gate_min_streak=int(getattr(args, "severity_gate_min_streak", 1) or 1),
        entropy_mode=str(args.entropy_mode),
        severity_decay=float(args.severity_decay),
        freeze_baseline_steps=int(args.freeze_baseline_steps),
        ema_decay_mode=str(args.ema_decay_mode),
        ema_gamma_min=float(args.ema_gamma_min),
        ema_gamma_max=float(args.ema_gamma_max),
        ema_gamma_fixed=float(args.ema_gamma_fixed) if args.ema_gamma_fixed is not None else None,
        ema_severity_mode=str(args.ema_severity_mode),
        ema_severity_weights=parse_ema_weights(args.ema_severity_weights),
        ema_severity_smoothing=float(args.ema_severity_smoothing),
        ema_severity_threshold=float(args.ema_severity_threshold),
        ema_severity_threshold_off=float(args.ema_severity_threshold_off) if args.ema_severity_threshold_off is not None else None,
        ema_cooldown_steps=int(args.ema_cooldown_steps),
        ema_use_candidate=bool(args.ema_use_candidate),
        ema_use_drift_flag=bool(args.ema_use_drift_flag),
        loss_scheduler_mode=str(args.loss_scheduler_mode),
        loss_lambda_min=float(args.loss_lambda_min) if args.loss_lambda_min is not None else None,
        loss_lambda_max=float(args.loss_lambda_max) if args.loss_lambda_max is not None else None,
        loss_tau_min=float(args.loss_tau_min) if args.loss_tau_min is not None else None,
        loss_tau_max=float(args.loss_tau_max) if args.loss_tau_max is not None else None,
        loss_lambda_fixed=float(args.loss_lambda_fixed) if args.loss_lambda_fixed is not None else None,
        loss_tau_fixed=float(args.loss_tau_fixed) if args.loss_tau_fixed is not None else None,
        loss_severity_mode=str(args.loss_severity_mode),
        loss_severity_weights=parse_loss_weights(args.loss_severity_weights),
        loss_severity_momentum=float(args.loss_severity_momentum),
        loss_severity_smoothing=float(args.loss_severity_smoothing),
        loss_severity_low=float(args.loss_severity_low),
        loss_severity_high=float(args.loss_severity_high),
        loss_severity_eps=float(args.loss_severity_eps),
        loss_severity_use_error=bool(args.loss_severity_use_error),
        loss_event_on=float(args.loss_event_on),
        loss_event_off=float(args.loss_event_off),
        loss_cooldown_steps=int(args.loss_cooldown_steps),
        loss_use_candidate=bool(args.loss_use_candidate),
        loss_use_drift_flag=bool(args.loss_use_drift_flag),
        loss_apply_lambda=bool(int(args.loss_apply_lambda)),
        loss_apply_tau=bool(int(args.loss_apply_tau)),
        loss_safety_enabled=bool(args.loss_safety_enabled),
        loss_safety_use_candidate=bool(args.loss_safety_use_candidate),
        loss_safety_use_severity=bool(args.loss_safety_use_severity),
        loss_safety_severity_threshold=float(args.loss_safety_severity_threshold),
        loss_safety_cooldown_steps=int(args.loss_safety_cooldown_steps),
        loss_safety_tau=float(args.loss_safety_tau) if args.loss_safety_tau is not None else None,
        loss_safety_lambda=float(args.loss_safety_lambda) if args.loss_safety_lambda is not None else None,
    )
    logs = run_ts_experiment(config=config, device=args.device)
    if logs.empty:
        print("未生成有效日志。")
        return
    final_acc = logs["metric_accuracy"].iloc[-1]
    print(f"[{args.dataset_name}][{args.model_variant}] 最终准确率: {final_acc:.4f}")
    if args.log_path:
        print(f"训练日志已保存至 {args.log_path}")


def parse_hidden_dims(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else [128, 64]


def parse_trigger_weights(spec: str) -> Optional[dict]:
    if not spec:
        return None
    if spec.strip().lower() in {"none", "null"}:
        return None
    weights = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--trigger_weights 格式错误：{token}（期望 key=value）")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"--trigger_weights 格式错误：{token}（空 key）")
        try:
            weights[key] = float(value)
        except Exception:
            weights[key] = value
    return weights or None


def parse_ema_weights(spec: str) -> Optional[tuple[float, float, float]]:
    if not spec:
        return None
    if spec.strip().lower() in {"none", "null"}:
        return None
    weights = {"error_rate": None, "divergence": None, "teacher_entropy": None}
    if "=" not in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--ema_severity_weights 需要 3 个数或 key=value")
        return tuple(float(x) for x in parts)  # type: ignore[return-value]
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--ema_severity_weights 格式错误：{token}（期望 key=value）")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"error", "err"}:
            key = "error_rate"
        if key in {"div", "divergence"}:
            key = "divergence"
        if key in {"entropy", "teacher_entropy"}:
            key = "teacher_entropy"
        if key not in weights:
            raise ValueError(f"--ema_severity_weights 未知 key：{key}")
        weights[key] = float(value)
    return (
        float(weights["error_rate"] or 0.0),
        float(weights["divergence"] or 0.0),
        float(weights["teacher_entropy"] or 0.0),
    )


def parse_loss_weights(spec: str) -> Optional[tuple[float, float, float]]:
    if not spec:
        return None
    if spec.strip().lower() in {"none", "null"}:
        return None
    weights = {"error_rate": None, "divergence": None, "teacher_entropy": None}
    if "=" not in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--loss_severity_weights 需要 3 个数或 key=value")
        return tuple(float(x) for x in parts)  # type: ignore[return-value]
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--loss_severity_weights 格式错误：{token}（期望 key=value）")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"error", "err"}:
            key = "error_rate"
        if key in {"div", "divergence"}:
            key = "divergence"
        if key in {"entropy", "teacher_entropy"}:
            key = "teacher_entropy"
        if key not in weights:
            raise ValueError(f"--loss_severity_weights 未知 key：{key}")
        weights[key] = float(value)
    return (
        float(weights["error_rate"] or 0.0),
        float(weights["divergence"] or 0.0),
        float(weights["teacher_entropy"] or 0.0),
    )


if __name__ == "__main__":
    main()
