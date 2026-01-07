"""批量运行真实数据集的 ts_drift_adapt/ts_drift_adapt_severity 实验。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run

@dataclass
class RealDatasetConfig:
    dataset_type: str
    dataset_name: str
    csv_path: str
    label_col: Optional[str]
    batch_size: int
    n_steps: int
    labeled_ratio: float
    initial_alpha: float
    initial_lr: float
    lambda_u: float
    tau: float


REAL_DATASETS: Dict[str, RealDatasetConfig] = {
    "Electricity": RealDatasetConfig(
        dataset_type="uspds_csv",
        dataset_name="Electricity",
        csv_path="datasets/real/Electricity.csv",
        label_col=None,
        batch_size=256,
        n_steps=1000,
        labeled_ratio=0.05,
        initial_alpha=0.995,
        initial_lr=5e-4,
        lambda_u=0.7,
        tau=0.9,
    ),
    "NOAA": RealDatasetConfig(
        dataset_type="uspds_csv",
        dataset_name="NOAA",
        csv_path="datasets/real/NOAA.csv",
        label_col=None,
        batch_size=256,
        n_steps=1200,
        labeled_ratio=0.05,
        initial_alpha=0.995,
        initial_lr=5e-4,
        lambda_u=0.7,
        tau=0.9,
    ),
    "INSECTS_abrupt_balanced": RealDatasetConfig(
        dataset_type="insects_real",
        dataset_name="INSECTS_abrupt_balanced",
        csv_path="datasets/real/INSECTS_abrupt_balanced.csv",
        label_col=None,
        batch_size=256,
        n_steps=1200,
        labeled_ratio=0.05,
        initial_alpha=0.97,
        initial_lr=1e-3,
        lambda_u=0.5,
        tau=0.8,
    ),
    "Airlines": RealDatasetConfig(
        dataset_type="uspds_csv",
        dataset_name="Airlines",
        csv_path="datasets/real/Airlines.csv",
        label_col=None,
        batch_size=256,
        n_steps=1000,
        labeled_ratio=0.05,
        initial_alpha=0.995,
        initial_lr=5e-4,
        lambda_u=0.7,
        tau=0.9,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在真实数据集上运行 ts_drift_adapt 多 seed 实验")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Electricity,NOAA,INSECTS_abrupt_balanced,Airlines",
        help="逗号分隔的数据集名称（必须是 REAL_DATASETS 中的 key）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="需要运行的随机种子",
    )
    parser.add_argument(
        "--model_variants",
        type=str,
        default="ts_drift_adapt",
        help="逗号分隔的模型变体列表（例如 ts_drift_adapt,ts_drift_adapt_severity）",
    )
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="error_divergence_ph_meta",
        choices=["none", "error_ph_meta", "divergence_ph_meta", "error_divergence_ph_meta"],
        help="run_experiment 的漂移检测预设",
    )
    parser.add_argument(
        "--trigger_mode",
        type=str,
        default="or",
        choices=["or", "k_of_n", "weighted", "two_stage"],
        help="多 detector 融合触发策略（默认 or）",
    )
    parser.add_argument("--trigger_k", type=int, default=2, help="trigger_mode=k_of_n 时的 k")
    parser.add_argument(
        "--trigger_threshold",
        type=float,
        default=0.5,
        help="trigger_mode=weighted 时的阈值",
    )
    parser.add_argument("--confirm_window", type=int, default=200, help="trigger_mode=two_stage 的 confirm_window")
    parser.add_argument(
        "--trigger_weights",
        type=str,
        default="",
        help="trigger_mode=weighted 时的权重（key=value 逗号分隔，空表示默认）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="运行设备（传给 run_experiment）")
    parser.add_argument("--logs_root", type=str, default="logs", help="日志输出根目录")
    parser.add_argument("--results_root", type=str, default="results", help="结果输出根目录")
    parser.add_argument("--run_name", type=str, default=None, help="附加到 run_id 的别名")
    parser.add_argument("--run_id", type=str, default=None, help="覆盖自动生成的 run_id")
    parser.add_argument(
        "--severity_scheduler_scale",
        type=float,
        default=1.0,
        help="severity-aware 调度缩放（仅 *_severity 变体生效）",
    )
    parser.add_argument("--use_severity_v2", action="store_true", help="启用 Severity-Aware v2")
    parser.add_argument(
        "--severity_gate",
        type=str,
        default="none",
        choices=["none", "confirmed_only"],
        help="severity v2 gating（confirmed_only 仅高置信 drift 更新 carry/freeze）",
    )
    parser.add_argument(
        "--entropy_mode",
        type=str,
        default="overconfident",
        choices=["overconfident", "uncertain", "abs"],
        help="SeverityCalibrator 的 entropy 正向增量定义",
    )
    parser.add_argument("--severity_decay", type=float, default=0.95, help="Severity-Aware v2 的 carry 衰减系数")
    parser.add_argument("--freeze_baseline_steps", type=int, default=0, help="漂移后冻结 baseline 的步数")
    return parser.parse_args()


def parse_model_variants(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def build_command(
    cfg: RealDatasetConfig,
    model_variant: str,
    seed: int,
    log_path: Path,
    monitor_preset: str,
    device: str,
    severity_scheduler_scale: float,
    experiment_run: ExperimentRun,
    experiment_run_params: Dict[str, Any],
) -> List[str]:
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--dataset_type",
        cfg.dataset_type,
        "--dataset_name",
        cfg.dataset_name,
        "--model_variant",
        model_variant,
        "--seed",
        str(seed),
        "--monitor_preset",
        monitor_preset,
        "--trigger_mode",
        str(experiment_run_params.get("trigger_mode", "or")),
        "--trigger_k",
        str(experiment_run_params.get("trigger_k", 2)),
        "--trigger_threshold",
        str(experiment_run_params.get("trigger_threshold", 0.5)),
        "--confirm_window",
        str(experiment_run_params.get("confirm_window", 200)),
        "--device",
        device,
        "--log_path",
        str(log_path),
        "--results_root",
        str(experiment_run.results_root),
        "--logs_root",
        str(experiment_run.logs_root),
        "--experiment_name",
        experiment_run.experiment_name,
        "--run_id",
        experiment_run.run_id,
        "--batch_size",
        str(cfg.batch_size),
        "--labeled_ratio",
        str(cfg.labeled_ratio),
        "--n_steps",
        str(cfg.n_steps),
        "--initial_alpha",
        str(cfg.initial_alpha),
        "--initial_lr",
        str(cfg.initial_lr),
        "--lambda_u",
        str(cfg.lambda_u),
        "--tau",
        str(cfg.tau),
        "--csv_path",
        cfg.csv_path,
    ]
    if cfg.label_col:
        cmd.extend(["--label_col", cfg.label_col])
    cmd.extend(["--severity_scheduler_scale", str(severity_scheduler_scale)])
    if experiment_run_params.get("trigger_weights"):
        cmd.extend(["--trigger_weights", str(experiment_run_params["trigger_weights"])])
    if experiment_run_params.get("use_severity_v2"):
        cmd.append("--use_severity_v2")
        cmd.extend(["--severity_gate", str(experiment_run_params.get("severity_gate", "none"))])
        cmd.extend(["--entropy_mode", str(experiment_run_params.get("entropy_mode", "overconfident"))])
        cmd.extend(["--severity_decay", str(experiment_run_params.get("severity_decay", 0.95))])
        cmd.extend(["--freeze_baseline_steps", str(experiment_run_params.get("freeze_baseline_steps", 0))])
    return cmd


def run_dataset(
    name: str,
    cfg: RealDatasetConfig,
    model_variants: List[str],
    seeds: List[int],
    monitor_preset: str,
    device: str,
    severity_scheduler_scale: float,
    experiment_run: ExperimentRun,
    *,
    trigger_mode: str = "or",
    trigger_k: int = 2,
    trigger_threshold: float = 0.5,
    trigger_weights: str = "",
    use_severity_v2: bool = False,
    entropy_mode: str = "overconfident",
    severity_decay: float = 0.95,
    freeze_baseline_steps: int = 0,
    confirm_window: int = 200,
    severity_gate: str = "none",
) -> None:
    for variant in model_variants:
        for seed in seeds:
            run_paths = experiment_run.prepare_dataset_run(cfg.dataset_name, variant, seed)
            log_path = run_paths.log_csv_path()
            exp_params = {
                "trigger_mode": trigger_mode,
                "trigger_k": trigger_k,
                "trigger_threshold": trigger_threshold,
                "trigger_weights": trigger_weights,
                "confirm_window": confirm_window,
                "use_severity_v2": use_severity_v2,
                "severity_gate": severity_gate,
                "entropy_mode": entropy_mode,
                "severity_decay": severity_decay,
                "freeze_baseline_steps": freeze_baseline_steps,
            }
            cmd = build_command(
                cfg,
                variant,
                seed,
                log_path,
                monitor_preset,
                device,
                severity_scheduler_scale,
                experiment_run,
                exp_params,
            )
            print(f"[run] dataset={cfg.dataset_name} model={variant} seed={seed} log={log_path}")
            subprocess.run(cmd, check=True)
            run_paths.update_legacy_pointer()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    experiment_run = create_experiment_run(
        experiment_name="run_real_adaptive",
        results_root=args.results_root,
        logs_root=args.logs_root,
        run_name=args.run_name,
        run_id=args.run_id,
    )
    print(f"[run] run_real_adaptive run_id={experiment_run.run_id}")
    model_variants = parse_model_variants(args.model_variants)
    trigger_mode = args.trigger_mode
    trigger_k = int(args.trigger_k)
    trigger_threshold = float(args.trigger_threshold)
    trigger_weights = args.trigger_weights
    confirm_window = int(args.confirm_window)
    use_severity_v2 = bool(args.use_severity_v2)
    severity_gate = str(args.severity_gate)
    entropy_mode = str(args.entropy_mode)
    severity_decay = float(args.severity_decay)
    freeze_baseline_steps = int(args.freeze_baseline_steps)
    for name in datasets:
        if name not in REAL_DATASETS:
            print(f"[warn] dataset '{name}' 未在 REAL_DATASETS 中定义，跳过")
            continue
        cfg = REAL_DATASETS[name]
        run_dataset(
            name,
            cfg,
            model_variants,
            args.seeds,
            args.monitor_preset,
            args.device,
            args.severity_scheduler_scale,
            experiment_run,
            trigger_mode=trigger_mode,
            trigger_k=trigger_k,
            trigger_threshold=trigger_threshold,
            trigger_weights=trigger_weights,
            confirm_window=confirm_window,
            use_severity_v2=use_severity_v2,
            severity_gate=severity_gate,
            entropy_mode=entropy_mode,
            severity_decay=severity_decay,
            freeze_baseline_steps=freeze_baseline_steps,
        )
    print("[done] all requested runs finished.")


if __name__ == "__main__":
    main()
