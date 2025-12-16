"""离线表格半监督（Student-Teacher EMA）训练实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.offline_real_datasets import OfflineDatasetSplits
from models.tabular_mlp_baseline import TabularMLPConfig, build_tabular_mlp


@dataclass
class TabularSemiEMATrainingConfig:
    """半监督训练配置。"""

    max_epochs: int = 50
    batch_size: int = 1024
    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    labeled_ratio: float = 0.05
    ema_momentum: float = 0.99
    lambda_u: float = 1.0
    rampup_epochs: int = 5
    confidence_threshold: float = 0.8
    device: str = "cuda"
    num_workers: int = 0


class SemiSupervisedDataset(Dataset):
    """含有 labeled mask 的训练数据集。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray) -> None:
        self.features = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(y).long()
        self.labeled_mask = torch.from_numpy(labeled_mask.astype(np.int64))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.labels[idx],
            self.labeled_mask[idx],
        )


class FullyLabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.features = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _create_labeled_mask(n_samples: int, labeled_ratio: float, seed: int) -> np.ndarray:
    if not 0 < labeled_ratio <= 1.0:
        raise ValueError("labeled_ratio 需在 (0, 1]")
    rng = np.random.default_rng(seed)
    n_labeled = max(1, int(round(n_samples * labeled_ratio)))
    n_labeled = min(n_labeled, n_samples)
    mask = np.zeros(n_samples, dtype=bool)
    indices = rng.permutation(n_samples)[:n_labeled]
    mask[indices] = True
    return mask


def _build_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"不支持的 optimizer: {name}")


@torch.no_grad()
def _update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


@torch.no_grad()
def _evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(xb.size(0))
    return correct / total if total > 0 else float("nan")


def run_tabular_semi_ema_training(
    dataset_name: str,
    splits: OfflineDatasetSplits,
    labeled_ratio: float,
    seed: int,
    model_cfg: TabularMLPConfig,
    train_cfg: TabularSemiEMATrainingConfig,
) -> Dict[str, float]:
    """运行一次离线半监督 EMA 训练，并返回关键指标。"""

    cfg = copy.deepcopy(train_cfg)
    cfg.labeled_ratio = labeled_ratio
    device = torch.device(cfg.device)
    student = build_tabular_mlp(model_cfg).to(device)
    teacher = build_tabular_mlp(model_cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    optimizer = _build_optimizer(cfg.optimizer, student.parameters(), cfg.lr, cfg.weight_decay)

    labeled_mask = _create_labeled_mask(len(splits.X_train), labeled_ratio, seed)
    train_dataset = SemiSupervisedDataset(splits.X_train, splits.y_train, labeled_mask)
    val_dataset = FullyLabeledDataset(splits.X_val, splits.y_val)
    test_dataset = FullyLabeledDataset(splits.X_test, splits.y_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    best_teacher_val = -float("inf")
    best_epoch = 0
    best_teacher_state: Optional[Dict[str, torch.Tensor]] = None
    best_student_state: Optional[Dict[str, torch.Tensor]] = None
    best_student_val = 0.0

    print(
        f"[phase1][{dataset_name}][seed={seed}][ratio={labeled_ratio:.2f}] "
        f"train={len(train_dataset)} labeled={labeled_mask.sum()} device={device}"
    )

    for epoch in range(cfg.max_epochs):
        student.train()
        teacher.eval()
        if cfg.rampup_epochs <= 0:
            unsup_weight = cfg.lambda_u
        else:
            progress = min(1.0, float(epoch + 1) / float(cfg.rampup_epochs))
            unsup_weight = cfg.lambda_u * progress
        total_loss = 0.0
        total_steps = 0
        for xb, yb, mask in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device).bool()
            logits_student = student(xb)
            sup_loss = torch.tensor(0.0, device=device)
            if mask.any():
                sup_loss = nn.functional.cross_entropy(logits_student[mask], yb[mask])
            unsup_loss = torch.tensor(0.0, device=device)
            if (~mask).any():
                with torch.no_grad():
                    logits_teacher = teacher(xb)
                    probs_teacher = torch.softmax(logits_teacher, dim=1)
                    max_prob, pseudo_labels = probs_teacher.max(dim=1)
                confident_mask = (~mask) & (max_prob >= cfg.confidence_threshold)
                if confident_mask.any():
                    unsup_loss = nn.functional.cross_entropy(
                        logits_student[confident_mask], pseudo_labels[confident_mask]
                    )
            loss = sup_loss + unsup_weight * unsup_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _update_ema(student, teacher, cfg.ema_momentum)
            total_loss += float(loss.item())
            total_steps += 1
        avg_loss = total_loss / max(1, total_steps)
        student_val_acc = _evaluate_accuracy(student, val_loader, device)
        teacher_val_acc = _evaluate_accuracy(teacher, val_loader, device)
        print(
            f"[phase1][{dataset_name}] epoch={epoch+1}/{cfg.max_epochs} "
            f"loss={avg_loss:.4f} sup_ratio={cfg.labeled_ratio:.2f} "
            f"val_acc_student={student_val_acc:.4f} val_acc_teacher={teacher_val_acc:.4f}"
        )
        if teacher_val_acc > best_teacher_val:
            best_teacher_val = teacher_val_acc
            best_student_val = student_val_acc
            best_epoch = epoch + 1
            best_teacher_state = copy.deepcopy(teacher.state_dict())
            best_student_state = copy.deepcopy(student.state_dict())

    assert best_teacher_state is not None and best_student_state is not None, "训练未产生有效的 best state"
    teacher.load_state_dict(best_teacher_state)
    student.load_state_dict(best_student_state)
    student_test_acc = _evaluate_accuracy(student, test_loader, device)
    teacher_test_acc = _evaluate_accuracy(teacher, test_loader, device)

    return {
        "train_samples": len(splits.X_train),
        "train_labeled_samples": int(labeled_mask.sum()),
        "val_samples": len(splits.X_val),
        "test_samples": len(splits.X_test),
        "best_epoch": best_epoch,
        "best_val_acc_teacher": best_teacher_val,
        "best_val_acc_student": best_student_val,
        "test_acc_teacher": teacher_test_acc,
        "test_acc_student": student_test_acc,
    }
