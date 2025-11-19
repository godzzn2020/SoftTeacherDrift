"""基于 river 的数据流工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from river import stream as rv_stream
from river.datasets import synth


Array = np.ndarray


def _dict_to_array(x: Dict[str, float]) -> Array:
    """将 river 样本 dict 转为 numpy 数组。"""
    return np.asarray(list(x.values()), dtype=np.float32)


def make_sea_stream(
    n_concepts: int,
    concept_length: int,
    seed: int,
) -> Iterator[Tuple[Array, int]]:
    """生成包含概念漂移的 SEA 数据流。"""
    if n_concepts <= 0 or concept_length <= 0:
        raise ValueError("n_concepts 与 concept_length 需为正整数")
    rng = np.random.default_rng(seed)
    for concept_idx in range(n_concepts):
        dataset = synth.SEA(
            variant=concept_idx % 4,
            seed=int(rng.integers(0, 1_000_000)),
            noise=0.0,
        )
        for x, y in dataset.take(concept_length):
            yield _dict_to_array(x), int(bool(y))


def make_hyperplane_stream(
    n_samples: int,
    drift_speed: float,
    seed: int,
) -> Iterator[Tuple[Array, int]]:
    """生成 Hyperplane 渐变漂移流。"""
    if n_samples <= 0:
        raise ValueError("n_samples 需为正整数")
    dataset = synth.Hyperplane(
        seed=seed,
        n_features=5,
        n_drift_features=5,
        mag_change=drift_speed,
    )
    for x, y in dataset.take(n_samples):
        yield _dict_to_array(x), int(y)


@dataclass
class UspdsStream(Iterable[Tuple[Dict[str, float], Any]]):
    """封装 USPDS 式 CSV 流，同时暴露特征/类别信息。"""

    rows: List[Tuple[Dict[str, float], Any]]
    feature_names: List[str]
    classes: List[Any]

    def __iter__(self) -> Iterator[Tuple[Dict[str, float], Any]]:
        for row in self.rows:
            yield row


def load_uspds_stream(csv_path: str, label_col: str) -> UspdsStream:
    """加载本地 CSV 并返回 river 兼容的字典流。"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label 列 {label_col} 不存在")
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols]
    y = df[label_col]
    rows = [(dict(x), label) for x, label in rv_stream.iter_pandas(X=X, y=y)]
    classes = list(pd.unique(y))
    return UspdsStream(rows=rows, feature_names=feature_cols, classes=classes)


def batch_stream(
    stream: Iterable[Tuple[Any, Any]],
    batch_size: int,
    labeled_ratio: float,
    seed: int = 0,
) -> Iterator[Tuple[List[Any], List[Any], List[Any], List[Any]]]:
    """将样本流按批次划分并拆分有/无标签子集。"""
    if not 0 < labeled_ratio <= 1:
        raise ValueError("labeled_ratio 需在 (0, 1]")
    rng = np.random.default_rng(seed)
    batch_x: List[Any] = []
    batch_y: List[Any] = []
    for x, y in stream:
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) < batch_size:
            continue
        yield _split_batch(batch_x, batch_y, labeled_ratio, rng)
        batch_x, batch_y = [], []
    if batch_x:
        yield _split_batch(batch_x, batch_y, labeled_ratio, rng)


def _split_batch(
    xs: Sequence[Any],
    ys: Sequence[Any],
    labeled_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    size = len(xs)
    n_labeled = max(1, int(size * labeled_ratio))
    n_labeled = min(n_labeled, size)
    indices = np.arange(size)
    rng.shuffle(indices)
    labeled_idx = indices[:n_labeled]
    unlabeled_idx = indices[n_labeled:]
    X_labeled = [xs[i] for i in labeled_idx]
    y_labeled = [ys[i] for i in labeled_idx]
    X_unlabeled = [xs[i] for i in unlabeled_idx]
    y_unlabeled = [ys[i] for i in unlabeled_idx]
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled
