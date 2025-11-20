"""在线训练循环实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import os
import time

import numpy as np
import pandas as pd
import torch
from river import metrics
from torch import Tensor

from drift.detectors import DriftMonitor
from drift.signals import compute_signals
from models.teacher_student import LossOutputs, TeacherStudentModel
from scheduler.hparam_scheduler import HParams, SchedulerState, update_hparams


@dataclass
class TrainingConfig:
    """训练循环配置。"""

    n_steps: int
    device: str = "cpu"
    log_path: Optional[str] = None
    dataset_type: str = "unknown"
    dataset_name: str = "unknown"
    model_variant: str = "ts_drift_adapt"
    seed: int = 0


class FeatureVectorizer:
    """将 dict 或数组样本转为统一向量。"""

    def __init__(self, input_dim: int, feature_order: Optional[Sequence[str]] = None) -> None:
        self.input_dim = input_dim
        self.feature_order: Optional[List[str]] = list(feature_order) if feature_order else None

    def transform(self, sample: Any) -> np.ndarray:
        if isinstance(sample, np.ndarray):
            return sample.astype(np.float32)
        if isinstance(sample, dict):
            if self.feature_order is None:
                self.feature_order = list(sorted(sample.keys()))
            vec = np.array([float(sample.get(k, 0.0)) for k in self.feature_order], dtype=np.float32)
            if len(vec) != self.input_dim:
                raise ValueError("特征维度与模型输入不一致")
            return vec
        if isinstance(sample, (list, tuple)):
            return np.asarray(sample, dtype=np.float32)
        raise TypeError(f"不支持的样本类型: {type(sample)}")

    def transform_many(self, samples: Sequence[Any]) -> Optional[np.ndarray]:
        if not samples:
            return None
        return np.stack([self.transform(s) for s in samples], axis=0)


class LabelEncoder:
    """标签编码器。"""

    def __init__(self, classes: Optional[Sequence[Any]] = None) -> None:
        self.class_to_idx: Dict[Any, int] = {}
        self.idx_to_class: List[Any] = []
        if classes:
            for c in classes:
                self._add_class(c)

    def _add_class(self, label: Any) -> int:
        if label not in self.class_to_idx:
            idx = len(self.idx_to_class)
            self.class_to_idx[label] = idx
            self.idx_to_class.append(label)
        return self.class_to_idx[label]

    def encode(self, label: Any) -> int:
        return self._add_class(label)

    def encode_many(self, labels: Sequence[Any]) -> np.ndarray:
        return np.asarray([self.encode(lbl) for lbl in labels], dtype=np.int64)

    def decode(self, idx: int) -> Any:
        return self.idx_to_class[idx]

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_class)


def run_training_loop(
    batch_iter: Iterator[Tuple[List[Any], List[Any], List[Any], List[Any]]],
    model: TeacherStudentModel,
    optimizer: torch.optim.Optimizer,
    drift_monitor: DriftMonitor,
    scheduler_state: SchedulerState,
    metric: metrics.base.Metric,
    initial_hparams: HParams,
    vectorizer: FeatureVectorizer,
    label_encoder: LabelEncoder,
    config: TrainingConfig,
) -> pd.DataFrame:
    """运行在线训练，返回逐批日志 DataFrame。"""
    device = torch.device(config.device)
    model.to(device)
    current_hparams = initial_hparams
    kappa_metric = metrics.CohenKappa()
    logs: List[Dict[str, Any]] = []
    seen_samples = 0
    sample_idx = -1
    for step, batch in enumerate(batch_iter, start=1):
        if step > config.n_steps:
            break
        scheduler_state.step = step
        (
            x_labeled_raw,
            y_labeled_raw,
            x_unlabeled_raw,
            y_unlabeled_raw,
        ) = batch
        batch_sample_count = len(x_labeled_raw) + len(x_unlabeled_raw)
        seen_samples += batch_sample_count
        sample_idx += batch_sample_count
        x_labeled_np = vectorizer.transform_many(x_labeled_raw)
        x_unlabeled_np = vectorizer.transform_many(x_unlabeled_raw)
        y_labeled_np = label_encoder.encode_many(y_labeled_raw) if y_labeled_raw else None
        y_unlabeled_np = label_encoder.encode_many(y_unlabeled_raw) if y_unlabeled_raw else None

        x_labeled = _to_tensor(x_labeled_np, device)
        y_labeled = _to_label_tensor(y_labeled_np, device)
        x_unlabeled = _to_tensor(x_unlabeled_np, device)

        optimizer.zero_grad(set_to_none=True)
        losses = model.compute_losses(
            x_labeled=x_labeled,
            y_labeled=y_labeled,
            x_unlabeled=x_unlabeled,
            hparams=current_hparams,
        )
        losses.total.backward()
        optimizer.step()
        model.update_teacher(current_hparams.alpha)

        stats = _collect_statistics(losses, y_labeled_np, y_unlabeled_np)
        drift_flag, severity = drift_monitor.update(stats["signals"], step)
        current_hparams, regime = update_hparams(
            scheduler_state, current_hparams, drift_flag, severity
        )
        _set_optimizer_lr(optimizer, current_hparams.lr)

        acc_value, kappa_value = _update_metrics(
            metric,
            kappa_metric,
            stats["student_probs_labeled"],
            y_labeled_np,
            stats["student_probs_unlabeled"],
            y_unlabeled_np,
        )
        logs.append(
            {
                "step": step,
                "seen_samples": seen_samples,
                "sample_idx": sample_idx,
                "dataset_name": config.dataset_name,
                "dataset_type": config.dataset_type,
                "model_variant": config.model_variant,
                "seed": config.seed,
                "metric_accuracy": acc_value,
                "metric_kappa": kappa_value,
                "student_error_rate": stats["signals"]["error_rate"],
                "teacher_entropy": stats["signals"]["teacher_entropy"],
                "divergence_js": stats["signals"]["divergence"],
                "drift_flag": int(drift_flag),
                "drift_severity": severity,
                "regime": regime,
                "alpha": current_hparams.alpha,
                "lr": current_hparams.lr,
                "lambda_u": current_hparams.lambda_u,
                "tau": current_hparams.tau,
                "timestamp": time.time(),
                "supervised_loss": float(losses.supervised.detach().cpu().item()),
                "unsupervised_loss": float(losses.unsupervised.detach().cpu().item()),
            }
        )
    df = pd.DataFrame(logs)
    if config.log_path:
        log_dir = os.path.dirname(config.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        df.to_csv(config.log_path, index=False)
    return df


def _to_tensor(arr: Optional[np.ndarray], device: torch.device) -> Optional[Tensor]:
    if arr is None:
        return None
    return torch.from_numpy(arr).to(device=device)


def _to_label_tensor(arr: Optional[np.ndarray], device: torch.device) -> Optional[Tensor]:
    if arr is None:
        return None
    return torch.from_numpy(arr.astype(np.int64)).to(device=device)


def _collect_statistics(
    losses: LossOutputs,
    y_labeled: Optional[np.ndarray],
    y_unlabeled: Optional[np.ndarray],
) -> Dict[str, Any]:
    """提取漂移信号与概率。"""
    student_probs_labeled = None
    if "student_logits_labeled" in losses.details and y_labeled is not None:
        probs = torch.softmax(losses.details["student_logits_labeled"], dim=1)
        student_probs_labeled = probs.detach().cpu().numpy()
    teacher_probs = None
    student_probs_unlabeled = None
    if "teacher_probs" in losses.details:
        teacher_probs = losses.details["teacher_probs"].detach().cpu().numpy()
    if "student_probs_unlabeled" in losses.details:
        student_probs_unlabeled = losses.details["student_probs_unlabeled"].detach().cpu().numpy()
    signals = compute_signals(
        student_probs_labeled=student_probs_labeled,
        y_labeled=y_labeled,
        teacher_probs_unlabeled=teacher_probs,
        student_probs_unlabeled=student_probs_unlabeled,
    )
    return {
        "signals": signals,
        "student_probs_labeled": student_probs_labeled,
        "student_probs_unlabeled": student_probs_unlabeled,
    }


def _update_metrics(
    metric: metrics.base.Metric,
    kappa_metric: metrics.base.Metric,
    student_probs_labeled: Optional[np.ndarray],
    y_labeled: Optional[np.ndarray],
    student_probs_unlabeled: Optional[np.ndarray],
    y_unlabeled: Optional[np.ndarray],
) -> Tuple[float, float]:
    """逐样本更新准确率与 Kappa。"""
    if student_probs_labeled is not None and y_labeled is not None:
        preds = student_probs_labeled.argmax(axis=1)
        for y_true, y_pred in zip(y_labeled, preds):
            metric.update(y_true=y_true, y_pred=y_pred)
            kappa_metric.update(y_true=y_true, y_pred=y_pred)
    if student_probs_unlabeled is not None and y_unlabeled is not None:
        preds_u = student_probs_unlabeled.argmax(axis=1)
        for y_true, y_pred in zip(y_unlabeled, preds_u):
            metric.update(y_true=y_true, y_pred=y_pred)
            kappa_metric.update(y_true=y_true, y_pred=y_pred)
    return metric.get(), kappa_metric.get()


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
