"""运行单次师生 EMA 半监督实验。"""

from __future__ import annotations

import argparse
from typing import List

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
        choices=["none", "error_ph_meta", "divergence_ph_meta", "error_divergence_ph_meta"],
        help="选择漂移检测器预设，none 表示不启用",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        log_path=args.log_path,
        monitor_preset=args.monitor_preset,
        severity_scheduler_scale=args.severity_scheduler_scale,
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


if __name__ == "__main__":
    main()
