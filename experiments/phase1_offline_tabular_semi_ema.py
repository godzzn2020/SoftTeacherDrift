"""Phase 1：离线表格半监督（Teacher-Student EMA）实验脚本。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.offline_real_datasets import OfflineDatasetConfig, OfflineDatasetSplits, load_offline_real_dataset
from models.tabular_mlp_baseline import TabularMLPConfig
from soft_drift.utils.run_paths import generate_run_id
from training.tabular_semi_ema import TabularSemiEMATrainingConfig, run_tabular_semi_ema_training

MODEL_VARIANT = "tabular_mlp_semi_ema"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase1 离线半监督 Tabular MLP + EMA Teacher")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Airlines,Electricity,NOAA,INSECTS_abrupt_balanced",
        help="逗号分隔的数据集列表",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="逗号或空格分隔的随机种子列表",
    )
    parser.add_argument(
        "--labeled_ratios",
        type=str,
        default="0.05,0.1",
        help="逗号分隔的 labeled_ratio 列表",
    )
    parser.add_argument("--output_dir", type=str, default="results/phase1_offline_semisup")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--no_batchnorm", action="store_true")
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--rampup_epochs", type=int, default=5)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    return parser.parse_args()


def parse_list(spec: str, cast) -> List:
    tokens = [item.strip() for item in spec.replace(";", ",").split(",") if item.strip()]
    return [cast(tok) for tok in tokens]


def load_dataset_splits(dataset: str, seed: int, cache: Dict[Tuple[str, int], OfflineDatasetSplits]) -> OfflineDatasetSplits:
    key = (dataset, seed)
    if key not in cache:
        cfg = OfflineDatasetConfig(name=dataset, random_state=seed)
        cache[key] = load_offline_real_dataset(cfg)
    return cache[key]


def append_run_records(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    columns = [
        "dataset_name",
        "model_variant",
        "seed",
        "run_id",
        "labeled_ratio",
        "train_samples",
        "train_labeled_samples",
        "val_samples",
        "test_samples",
        "best_epoch",
        "best_val_acc_teacher",
        "test_acc_teacher",
        "best_val_acc_student",
        "test_acc_student",
        "max_epochs",
        "batch_size",
        "optimizer",
        "lr",
        "weight_decay",
        "hidden_dim",
        "num_layers",
        "dropout",
        "activation",
        "use_batchnorm",
        "ema_momentum",
        "lambda_u",
        "rampup_epochs",
        "confidence_threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize_run_level(run_csv: Path, out_csv: Path) -> None:
    if not run_csv.exists():
        return
    df = pd.read_csv(run_csv)
    if df.empty:
        return
    grouped = df.groupby(["dataset_name", "model_variant", "labeled_ratio"], dropna=False)
    rows = []
    for (dataset, variant, ratio), group in grouped:
        rows.append(
            {
                "dataset_name": dataset,
                "model_variant": variant,
                "labeled_ratio": ratio,
                "runs": int(len(group)),
                "best_val_acc_teacher_mean": float(group["best_val_acc_teacher"].mean()),
                "best_val_acc_teacher_std": float(group["best_val_acc_teacher"].std(ddof=0)),
                "test_acc_teacher_mean": float(group["test_acc_teacher"].mean()),
                "test_acc_teacher_std": float(group["test_acc_teacher"].std(ddof=0)),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    datasets = parse_list(args.datasets, str)
    seeds = parse_list(args.seeds, int)
    ratios = parse_list(args.labeled_ratios, float)
    if not datasets:
        raise ValueError("至少需要一个数据集")
    if not seeds:
        raise ValueError("至少需要一个随机种子")
    if not ratios:
        raise ValueError("至少需要一个 labeled_ratio")
    output_dir = Path(args.output_dir)
    run_csv = output_dir / "run_level_metrics.csv"
    summary_csv = output_dir / "summary_metrics_by_dataset_variant.csv"
    split_cache: Dict[Tuple[str, int], OfflineDatasetSplits] = {}

    all_records: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            splits = load_dataset_splits(dataset, seed, split_cache)
            for ratio in ratios:
                run_id = generate_run_id("phase1_mlp_semi_ema")
                model_cfg = TabularMLPConfig(
                    input_dim=splits.input_dim,
                    num_classes=splits.num_classes,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_batchnorm=not args.no_batchnorm,
                    activation=args.activation,
                )
                train_cfg = TabularSemiEMATrainingConfig(
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    optimizer=args.optimizer,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    labeled_ratio=ratio,
                    ema_momentum=args.ema_momentum,
                    lambda_u=args.lambda_u,
                    rampup_epochs=args.rampup_epochs,
                    confidence_threshold=args.confidence_threshold,
                    device=args.device,
                    num_workers=args.num_workers,
                )
                metrics = run_tabular_semi_ema_training(
                    dataset_name=dataset,
                    splits=splits,
                    labeled_ratio=ratio,
                    seed=seed,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                )
                record = {
                    "dataset_name": dataset,
                    "model_variant": MODEL_VARIANT,
                    "seed": seed,
                    "run_id": run_id,
                    "labeled_ratio": ratio,
                    "max_epochs": args.max_epochs,
                    "batch_size": args.batch_size,
                    "optimizer": args.optimizer,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "activation": args.activation,
                    "use_batchnorm": not args.no_batchnorm,
                    "ema_momentum": args.ema_momentum,
                    "lambda_u": args.lambda_u,
                    "rampup_epochs": args.rampup_epochs,
                    "confidence_threshold": args.confidence_threshold,
                }
                record.update(metrics)
                all_records.append(record)
    append_run_records(run_csv, all_records)
    summarize_run_level(run_csv, summary_csv)
    print(f"[done] run-level metrics -> {run_csv}")
    print(f"[done] summary metrics -> {summary_csv}")


if __name__ == "__main__":
    main()
