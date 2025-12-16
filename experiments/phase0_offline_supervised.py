"""Phase 0：真实数据集离线监督 Tabular MLP 基线训练脚本。

注意：本脚本不会在导入时触发训练，仅在 CLI 调用（`python experiments/phase0_offline_supervised.py ...`）
且位于 `if __name__ == "__main__":` 中才会真正运行，以避免误触发大规模实验。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.offline_real_datasets import OfflineDatasetConfig, load_offline_real_dataset
from models.tabular_mlp_baseline import TabularMLPConfig, build_tabular_mlp
from soft_drift.utils.run_paths import create_experiment_run

EXPERIMENT_NAME = "phase0_offline_supervised"
MODEL_VARIANT = "tabular_mlp_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase0 离线监督 Tabular MLP 基线")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Airlines,Electricity,NOAA,INSECTS_abrupt_balanced",
        help="逗号分隔的真实数据集列表",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="逗号或空格分隔的随机种子列表",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        help="学习率调度器类型",
    )
    parser.add_argument("--lr_step_size", type=int, default=15, help="StepLR 的 step_size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR 的 gamma")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu"],
        help="隐藏层激活函数",
    )
    parser.add_argument(
        "--no_batchnorm",
        action="store_true",
        help="禁用 BatchNorm（默认启用）",
    )
    parser.add_argument("--labeled_ratio", type=float, default=1.0, help="训练集标注比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个数据集仅取前 N 条样本（用于快速 sanity check）",
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--run_tag", type=str, default=None, help="附加到 run_id 的标识")
    parser.add_argument("--run_id", type=str, default=None, help="覆盖自动生成的 run_id（谨慎使用）")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker 数（默认 0 避免多进程开销）",
    )
    return parser.parse_args()


def parse_str_list(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_int_list(spec: str) -> List[int]:
    tokens = spec.replace(",", " ").split()
    return [int(tok) for tok in tokens if tok.strip()]


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_dataloader(
    features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, num_workers: int
) -> Optional[DataLoader]:
    if len(features) == 0:
        return None
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for xb, yb in loader:
        xb = xb.to(device=device, non_blocking=True).float()
        yb = yb.to(device=device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        batch_size = xb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_samples += batch_size
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    if loader is None:
        return math.nan, math.nan
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for xb, yb in loader:
        xb = xb.to(device=device, non_blocking=True).float()
        yb = yb.to(device=device, non_blocking=True).long()
        logits = model(xb)
        loss = criterion(logits, yb)
        batch_size = xb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_samples += batch_size
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def snapshot_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def select_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.lr_step_size),
            gamma=args.lr_gamma,
        )
    raise ValueError(f"未知 lr_scheduler: {args.lr_scheduler}")


def format_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    if not datasets:
        raise ValueError("需要至少一个数据集")
    if not seeds:
        raise ValueError("需要至少一个随机种子")
    experiment_run = create_experiment_run(
        experiment_name=EXPERIMENT_NAME,
        results_root=args.results_root,
        logs_root=args.logs_root,
        run_name=args.run_tag,
        run_id=args.run_id,
    )
    print(f"[run] {EXPERIMENT_NAME} run_id={experiment_run.run_id}")
    run_records: List[Dict[str, object]] = []

    for dataset_name in datasets:
        for seed in seeds:
            dataset_run = experiment_run.prepare_dataset_run(dataset_name, MODEL_VARIANT, seed)
            result_dir = dataset_run.results_dir
            ensure_dir(result_dir / "metrics.csv")
            log_path = result_dir / "train.log"
            log_file = log_path.open("w", encoding="utf-8")

            def log(msg: str) -> None:
                line = f"[{format_timestamp()}][{dataset_name}][seed={seed}] {msg}"
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

            torch.manual_seed(seed)
            np.random.seed(seed)
            log("loading dataset splits...")
            ds_cfg = OfflineDatasetConfig(
                name=dataset_name,
                labeled_ratio=args.labeled_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=seed,
                max_samples=args.max_samples,
            )
            splits = load_offline_real_dataset(ds_cfg)
            X_train_full_size = len(splits.X_train)
            total_train_size = X_train_full_size
            if splits.X_train_unlabeled is not None:
                total_train_size += len(splits.X_train_unlabeled)
            log(
                f"dataset ready: train={total_train_size} (labeled={len(splits.X_train)}), "
                f"val={len(splits.X_val)}, test={len(splits.X_test)}, input_dim={splits.input_dim}, classes={splits.num_classes}"
            )

            model_cfg = TabularMLPConfig(
                input_dim=splits.input_dim,
                num_classes=splits.num_classes,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_batchnorm=not args.no_batchnorm,
                activation=args.activation,
            )
            model = build_tabular_mlp(model_cfg).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = select_scheduler(args, optimizer)

            train_loader = create_dataloader(
                splits.X_train,
                splits.y_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            val_loader = create_dataloader(
                splits.X_val,
                splits.y_val,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            test_loader = create_dataloader(
                splits.X_test,
                splits.y_test,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            if train_loader is None:
                log("警告：训练样本为空，跳过该组合")
                log_file.close()
                continue

            metrics_rows: List[Dict[str, float]] = []
            best_val_acc = -1.0
            best_epoch = 0
            best_state: Optional[Dict[str, torch.Tensor]] = None

            for epoch in range(1, args.max_epochs + 1):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                current_lr = optimizer.param_groups[0]["lr"]
                if scheduler is not None:
                    scheduler.step()
                metrics_rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": current_lr,
                    }
                )
                log(
                    f"epoch={epoch}/{args.max_epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.4e}"
                )
                if not math.isnan(val_acc) and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = snapshot_state_dict(model)

            if best_state is not None:
                model.load_state_dict(best_state)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            log(f"best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")

            metrics_df = pd.DataFrame(metrics_rows)
            metrics_path = result_dir / "metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            summary = {
                "dataset_name": dataset_name,
                "seed": seed,
                "labeled_ratio": args.labeled_ratio,
                "train_samples": total_train_size,
                "train_labeled_samples": len(splits.X_train),
                "val_samples": len(splits.X_val),
                "test_samples": len(splits.X_test),
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "max_epochs": args.max_epochs,
                "batch_size": args.batch_size,
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "activation": args.activation,
                "use_batchnorm": not args.no_batchnorm,
            }
            summary_path = result_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            run_records.append(summary)
            log_file.close()

    if not run_records:
        print("[warn] 未生成任何有效结果")
        return
    summary_dir = experiment_run.summary_dir()
    run_level_csv = summary_dir / "run_level_metrics.csv"
    summary_df = pd.DataFrame(run_records)
    summary_df.to_csv(run_level_csv, index=False)

    md_path = summary_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Phase0 Offline Supervised Summary — run_id={experiment_run.run_id}\n\n")
        f.write("| dataset | seed | labeled_ratio | best_val_acc | test_acc | best_epoch |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in run_records:
            f.write(
                f"| {row['dataset_name']} | {row['seed']} | {row['labeled_ratio']:.2f} | "
                f"{row['best_val_acc']:.4f} | {row['test_acc']:.4f} | {row['best_epoch']} |\n"
            )
    print(f"[done] summary saved to {run_level_csv} and {md_path}")


if __name__ == "__main__":
    main()
