"""Step-1 实验脚本，实现多数据集批量运行与统一日志。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
from river import metrics

from data import streams
from drift import detectors
from models.teacher_student import TeacherStudentModel
from scheduler.hparam_scheduler import HParams, SchedulerState
from training.loop import FeatureVectorizer, LabelEncoder, TrainingConfig, run_training_loop


def _use_severity_scheduler(model_variant: str) -> bool:
    return model_variant.endswith("_severity")


@dataclass
class ExperimentConfig:
    """单次实验的高层配置。"""

    dataset_type: str
    dataset_name: str
    csv_path: Optional[str] = None
    label_col: Optional[str] = None
    model_variant: str = "ts_drift_adapt"
    n_steps: int = 500
    batch_size: int = 64
    labeled_ratio: float = 0.1
    initial_alpha: float = 0.99
    initial_lr: float = 1e-3
    lambda_u: float = 1.0
    tau: float = 0.9
    seed: int = 42
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    stream_kwargs: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[str] = None
    monitor_preset: str = "none"


def run_experiment(config: ExperimentConfig, device: str = "cpu") -> pd.DataFrame:
    """
    构建数据流、模型与监控组件，运行在线训练，并返回日志 DataFrame。
    同时依据 config.log_path 写入 CSV（若提供）。
    """

    stream_info = streams.build_stream(
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        csv_path=config.csv_path,
        label_col=config.label_col,
        seed=config.seed,
        **config.stream_kwargs,
    )
    if stream_info.n_classes <= 0:
        raise ValueError("需要提供有效的类别数以构建模型。")
    vectorizer = FeatureVectorizer(
        input_dim=stream_info.n_features,
        feature_order=stream_info.feature_names,
    )
    label_encoder = LabelEncoder(classes=stream_info.classes)
    model = TeacherStudentModel(
        input_dim=stream_info.n_features,
        num_classes=stream_info.n_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        device=torch.device(device),
    )
    optimizer = torch.optim.Adam(model.student.parameters(), lr=config.initial_lr)
    monitor = detectors.build_default_monitor(preset=config.monitor_preset)
    initial_hparams = HParams(
        alpha=config.initial_alpha,
        lr=config.initial_lr,
        lambda_u=config.lambda_u,
        tau=config.tau,
    )
    scheduler_state = SchedulerState(base_hparams=initial_hparams)
    metric = metrics.Accuracy()
    batch_iter = streams.batch_stream(
        stream=stream_info,
        batch_size=config.batch_size,
        labeled_ratio=config.labeled_ratio,
        seed=config.seed,
    )
    training_config = TrainingConfig(
        n_steps=config.n_steps,
        device=device,
        log_path=config.log_path,
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        model_variant=config.model_variant,
        seed=config.seed,
        use_severity_scheduler=_use_severity_scheduler(config.model_variant),
    )
    logs_df = run_training_loop(
        batch_iter=batch_iter,
        model=model,
        optimizer=optimizer,
        drift_monitor=monitor,
        scheduler_state=scheduler_state,
        metric=metric,
        initial_hparams=initial_hparams,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        config=training_config,
    )
    return logs_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1 师生 EMA 实验批处理")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        type=str,
        default="sea_abrupt4,hyperplane_slow,electricity,airlines,insects_abrupt_balanced",
        help="逗号分隔的数据集名称列表，或 all",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ts_drift_adapt",
        help="逗号分隔的模型变体列表，或 all",
    )
    parser.add_argument(
        "--monitor_preset",
        type=str,
        default="none",
        choices=["none", "error_ph_meta", "divergence_ph_meta", "error_divergence_ph_meta"],
        help="漂移检测器预设（none 表示不启用）",
    )
    args = parser.parse_args()

    dataset_filters = _parse_list(args.datasets)
    model_filters = _parse_list(args.models)

    base_configs = _default_experiment_configs(args.seed)
    selected_datasets = [
        cfg for cfg in base_configs if not dataset_filters or cfg.dataset_name in dataset_filters
    ]
    if not selected_datasets:
        raise ValueError("未匹配到任何数据集配置")

    available_models = ["baseline_student", "mean_teacher", "ts_drift_adapt", "ts_drift_adapt_severity"]
    model_list = model_filters or available_models

    for base_cfg in selected_datasets:
        for model_variant in model_list:
            log_path = _default_log_path(base_cfg.dataset_name, model_variant, args.seed)
            cfg = replace(
                base_cfg,
                model_variant=model_variant,
                seed=args.seed,
                log_path=log_path,
                monitor_preset=args.monitor_preset,
            )
            logs = run_experiment(cfg, device=args.device)
            final_acc = logs["metric_accuracy"].iloc[-1] if not logs.empty else float("nan")
            print(
                f"[{cfg.dataset_name}][{cfg.model_variant}][seed={cfg.seed}] final accuracy = {final_acc:.4f}"
            )


def _default_log_path(dataset_name: str, model_variant: str, seed: int) -> str:
    log_dir = os.path.join("logs", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{dataset_name}__{model_variant}__seed{seed}.csv"
    return os.path.join(log_dir, filename)


def _parse_list(spec: str) -> Optional[List[str]]:
    if not spec or spec.lower() == "all":
        return None
    return [item.strip() for item in spec.split(",") if item.strip()]


def _default_experiment_configs(seed: int) -> List[ExperimentConfig]:
    """构造 Stage-1 默认实验集合。"""
    return [
        ExperimentConfig(
            dataset_type="sea",
            dataset_name="sea_abrupt4",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.99,
            initial_lr=1e-3,
            lambda_u=1.0,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="sine",
            dataset_name="sine_abrupt4",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.99,
            initial_lr=1e-3,
            lambda_u=1.0,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="stagger",
            dataset_name="stagger_abrupt3",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.99,
            initial_lr=1e-3,
            lambda_u=1.0,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="hyperplane",
            dataset_name="hyperplane_slow",
            n_steps=800,
            batch_size=64,
            labeled_ratio=0.1,
            initial_alpha=0.98,
            initial_lr=5e-4,
            lambda_u=0.8,
            tau=0.85,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="uspds_csv",
            dataset_name="electricity",
            csv_path="data/uspds/electricity.csv",
            label_col=None,
            n_steps=1000,
            batch_size=256,
            labeled_ratio=0.05,
            initial_alpha=0.995,
            initial_lr=5e-4,
            lambda_u=0.7,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="uspds_csv",
            dataset_name="airlines",
            csv_path="data/uspds/airlines.csv",
            label_col=None,
            n_steps=1000,
            batch_size=256,
            labeled_ratio=0.05,
            initial_alpha=0.995,
            initial_lr=5e-4,
            lambda_u=0.7,
            tau=0.9,
            seed=seed,
        ),
        ExperimentConfig(
            dataset_type="insects_real",
            dataset_name="INSECTS_abrupt_balanced",
            csv_path="datasets/real/INSECTS_abrupt_balanced.csv",
            n_steps=1200,
            batch_size=256,
            labeled_ratio=0.05,
            initial_alpha=0.97,
            initial_lr=1e-3,
            lambda_u=0.5,
            tau=0.8,
            seed=seed,
        ),
    ]


if __name__ == "__main__":
    main()
