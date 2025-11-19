"""批次统计信号。"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def batch_error_rate(student_probs: Optional[np.ndarray], y_true: Optional[np.ndarray]) -> float:
    """学生在有标签样本上的错误率。"""
    if student_probs is None or y_true is None or len(y_true) == 0:
        return float("nan")
    preds = student_probs.argmax(axis=1)
    return float(np.mean(preds != y_true))


def batch_teacher_entropy(teacher_probs: Optional[np.ndarray], eps: float = 1e-8) -> float:
    """教师预测熵。"""
    if teacher_probs is None or len(teacher_probs) == 0:
        return float("nan")
    probs = np.clip(teacher_probs, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return float(np.mean(entropy))


def batch_divergence(
    teacher_probs: Optional[np.ndarray],
    student_probs: Optional[np.ndarray],
    eps: float = 1e-8,
) -> float:
    """教师与学生分布间的 JS 散度。"""
    if (
        teacher_probs is None
        or student_probs is None
        or len(teacher_probs) == 0
        or len(student_probs) == 0
    ):
        return float("nan")
    t = np.clip(teacher_probs, eps, 1.0)
    s = np.clip(student_probs, eps, 1.0)
    m = 0.5 * (t + s)
    js = 0.5 * (np.sum(t * (np.log(t) - np.log(m)), axis=1)) + 0.5 * (
        np.sum(s * (np.log(s) - np.log(m)), axis=1)
    )
    return float(np.mean(js))


def compute_signals(
    student_probs_labeled: Optional[np.ndarray],
    y_labeled: Optional[np.ndarray],
    teacher_probs_unlabeled: Optional[np.ndarray],
    student_probs_unlabeled: Optional[np.ndarray],
) -> Dict[str, float]:
    """组合输出字典。"""
    return {
        "error_rate": batch_error_rate(student_probs_labeled, y_labeled),
        "teacher_entropy": batch_teacher_entropy(teacher_probs_unlabeled),
        "divergence": batch_divergence(teacher_probs_unlabeled, student_probs_unlabeled),
    }

