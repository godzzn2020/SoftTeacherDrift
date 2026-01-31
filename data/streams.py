"""统一的数据流构建与批次拆分工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import argparse
import json

import numpy as np
import pandas as pd
from river import stream as rv_stream
from river.datasets import synth, Insects

from datasets.preprocessing import load_tabular_csv_dataset


SEA_CONFIGS = {
    "sea_abrupt4": {"concept_ids": [0, 1, 2, 3, 0], "concept_length": 10000},
    "sea_reoccurring": {"concept_ids": [0, 1, 0, 2, 0, 3], "concept_length": 1000},
}

HYPERPLANE_CONFIGS = {
    "hyperplane_slow": {
        "n_features": 6,
        "n_drift_features": 6,
        "mag_change": 0.001,
        "sigma": 0.005,
        "noise_percentage": 0.01,
        "n_samples": 12_000,
    },
    "hyperplane_fast": {
        "n_features": 8,
        "n_drift_features": 6,
        "mag_change": 0.02,
        "sigma": 0.02,
        "noise_percentage": 0.02,
        "n_samples": 12_000,
    },
}

SINE_DEFAULT = {
    "sine_abrupt4": {
        "classification_functions": [0, 1, 2, 3, 0],
        "segment_length": 10000,
        "balance_classes": False,
        "has_noise": False,
    }
}

STAGGER_DEFAULT = {
    "stagger_abrupt3": {
        "classification_functions": [0, 1, 2, 0],
        "segment_length": 20000,
        "balance_classes": False,
    }
}

COVSHIFT_LINEAR_DEFAULTS = {
    "n_features": 8,
    "segment_length": 10000,
    "w": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "b": 0.0,
}

DEFAULT_ABRUPT_SYNTH_DATASETS = {
    "sea_abrupt4": {
        "dataset_type": "sea",
        "n_samples": 50000,
        "drift_cfg": {"concept_ids": [0, 1, 2, 3, 0], "concept_length": 10000, "drift_type": "abrupt"},
    },
    "sine_abrupt4": {
        "dataset_type": "sine",
        "n_samples": 50000,
        "drift_cfg": {
            "classification_functions": [0, 1, 2, 3, 0],
            "segment_length": 10000,
            "drift_type": "abrupt",
        },
    },
    "stagger_abrupt3": {
        "dataset_type": "stagger",
        "n_samples": 60000,
        "drift_cfg": {
            "classification_functions": [0, 1, 2, 0],
            "segment_length": 20000,
            "drift_type": "abrupt",
        },
    },
}


@dataclass
class StreamInfo(Iterable[Tuple[Dict[str, float], Any]]):
    """包含可重复迭代的数据流与相关元信息。"""

    iterator_fn: Callable[[], Iterator[Tuple[Dict[str, float], Any]]]
    n_features: int
    n_classes: int
    dataset_type: str
    dataset_name: str
    feature_names: Optional[List[str]] = None
    classes: Optional[List[Any]] = None

    def __iter__(self) -> Iterator[Tuple[Dict[str, float], Any]]:
        return self.iterator_fn()


def build_stream(
    dataset_type: str,
    dataset_name: str,
    csv_path: Optional[str] = None,
    label_col: Optional[str] = None,
    seed: int = 42,
    **kwargs: Any,
) -> StreamInfo:
    """根据类型构建统一的 river 风格数据流。"""
    dataset_type = dataset_type.lower()
    if dataset_type == "sea":
        return _build_sea_stream(dataset_name, seed=seed, **kwargs)
    if dataset_type == "hyperplane":
        return _build_hyperplane_stream(dataset_name, seed=seed, **kwargs)
    if dataset_type == "sine":
        return _build_sine_stream(dataset_name, seed=seed, **kwargs)
    if dataset_type == "stagger":
        return _build_stagger_stream(dataset_name, seed=seed, **kwargs)
    if dataset_type == "covshift_linear":
        return _build_covshift_linear_stream(dataset_name, seed=seed, **kwargs)
    if dataset_type == "uspds_csv":
        if csv_path is None:
            raise ValueError("uspds_csv 数据集需要提供 csv_path")
        return _build_uspds_csv_stream(dataset_name, csv_path, label_col)
    if dataset_type == "insects_river":
        return _build_insects_stream(dataset_name, seed=seed)
    if dataset_type == "insects_real":
        csv = csv_path or f"datasets/real/{dataset_name}.csv"
        return _build_uspds_csv_stream(dataset_name, csv, label_col)
    if dataset_type in {"sea_saved", "hyperplane_saved", "synth_saved"}:
        stream_info, _ = load_saved_synth_stream(
            dataset_name=dataset_name,
            seed=seed,
            data_root=kwargs.get("data_root", "data/synthetic"),
        )
        return stream_info
    raise ValueError(f"未知的数据集类型: {dataset_type}")


def _build_sea_stream(dataset_name: str, seed: int, **kwargs: Any) -> StreamInfo:
    config = SEA_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(f"未知的 SEA 配置: {dataset_name}")
    concept_length = int(kwargs.get("concept_length", config["concept_length"]))
    plan: Sequence[Tuple[int, int]] = kwargs.get("segments", None)
    if plan is None:
        plan = [(variant, concept_length) for variant in config["concept_ids"]]
    rng = np.random.default_rng(seed)

    def iterator() -> Iterator[Tuple[Dict[str, float], int]]:
        for variant, length in plan:
            dataset = synth.SEA(
                variant=variant,
                seed=int(rng.integers(0, 1_000_000)),
                noise=0.0,
            )
            for x, y in dataset.take(length):
                yield _format_synthetic_dict(x), int(bool(y))

    n_features = 3
    feature_names = [f"f{i}" for i in range(n_features)]
    return StreamInfo(
        iterator_fn=iterator,
        n_features=n_features,
        n_classes=2,
        dataset_type="sea",
        dataset_name=dataset_name,
        feature_names=feature_names,
        classes=[0, 1],
    )


def _build_hyperplane_stream(dataset_name: str, seed: int, **kwargs: Any) -> StreamInfo:
    config = HYPERPLANE_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(f"未知的 Hyperplane 配置: {dataset_name}")
    cfg = {**config, **kwargs}
    n_samples = int(cfg.get("n_samples", config["n_samples"]))
    n_features = int(cfg.get("n_features", config["n_features"]))
    dataset = synth.Hyperplane(
        seed=seed,
        n_features=n_features,
        n_drift_features=int(cfg.get("n_drift_features", config["n_drift_features"])),
        mag_change=float(cfg.get("mag_change", config["mag_change"])),
        sigma=float(cfg.get("sigma", config["sigma"])),
        noise_percentage=float(cfg.get("noise_percentage", config["noise_percentage"])),
    )

    def iterator() -> Iterator[Tuple[Dict[str, float], int]]:
        for x, y in dataset.take(n_samples):
            yield _format_synthetic_dict(x), int(y)

    feature_names = [f"f{i}" for i in range(n_features)]
    return StreamInfo(
        iterator_fn=iterator,
        n_features=n_features,
        n_classes=2,
        dataset_type="hyperplane",
        dataset_name=dataset_name,
        feature_names=feature_names,
        classes=[0, 1],
    )


def _build_sine_stream(dataset_name: str, seed: int, **kwargs: Any) -> StreamInfo:
    cfg = SINE_DEFAULT.get(dataset_name)
    if cfg is None:
        raise ValueError(f"未知的 Sine 配置: {dataset_name}")
    functions = kwargs.get("classification_functions", cfg["classification_functions"])
    segment_length = int(kwargs.get("segment_length", cfg["segment_length"]))
    def iterator() -> Iterator[Tuple[Dict[str, float], int]]:
        total = 0
        concept_id = 0
        while True:
            if total >= kwargs.get("n_samples", segment_length * len(functions)):
                break
            length = min(segment_length, kwargs.get("n_samples", segment_length * len(functions)) - total)
            dataset = synth.Sine(
                classification_function=functions[concept_id % len(functions)],
                seed=seed + concept_id * 31,
                balance_classes=bool(kwargs.get("balance_classes", cfg["balance_classes"])),
                has_noise=bool(kwargs.get("has_noise", cfg["has_noise"])),
            )
            for x, y in dataset.take(length):
                yield _format_synthetic_dict(x), int(bool(y))
            concept_id += 1
            total += length

    feature_names = ["f0", "f1"]
    return StreamInfo(
        iterator_fn=iterator,
        n_features=len(feature_names),
        n_classes=2,
        dataset_type="sine",
        dataset_name=dataset_name,
        feature_names=feature_names,
        classes=[0, 1],
    )


def _build_stagger_stream(dataset_name: str, seed: int, **kwargs: Any) -> StreamInfo:
    cfg = STAGGER_DEFAULT.get(dataset_name)
    if cfg is None:
        raise ValueError(f"未知的 STAGGER 配置: {dataset_name}")
    functions = kwargs.get("classification_functions", cfg["classification_functions"])
    segment_length = int(kwargs.get("segment_length", cfg["segment_length"]))

    def iterator() -> Iterator[Tuple[Dict[str, float], Any]]:
        total = 0
        concept_id = 0
        n_samples = kwargs.get("n_samples", segment_length * len(functions))
        while total < n_samples:
            length = min(segment_length, n_samples - total)
            dataset = synth.STAGGER(
                classification_function=functions[concept_id % len(functions)],
                seed=seed + concept_id * 73,
                balance_classes=bool(kwargs.get("balance_classes", cfg["balance_classes"])),
            )
            for x, y in dataset.take(length):
                yield _format_synthetic_dict(x), int(bool(y))
            concept_id += 1
            total += length

    feature_names = ["f0", "f1", "f2"]
    return StreamInfo(
        iterator_fn=iterator,
        n_features=len(feature_names),
        n_classes=2,
        dataset_type="stagger",
        dataset_name=dataset_name,
        feature_names=feature_names,
        classes=[0, 1],
    )


def _covshift_linear_segments(dataset_name: str, d: int) -> List[Dict[str, np.ndarray]]:
    if d < 2:
        raise ValueError("covshift_linear 需要至少 2 维特征")
    if dataset_name == "covshift_mean3":
        mu0 = np.zeros(d, dtype=float)
        mu1 = np.zeros(d, dtype=float)
        mu1[0] = 1.5
        mu2 = np.zeros(d, dtype=float)
        mu2[0] = -1.5
        cov = np.eye(d, dtype=float)
        return [
            {"mu": mu0, "cov": cov},
            {"mu": mu1, "cov": cov},
            {"mu": mu2, "cov": cov},
        ]
    if dataset_name == "covshift_scale3":
        mu = np.zeros(d, dtype=float)
        cov0 = np.eye(d, dtype=float)
        cov1 = np.eye(d, dtype=float)
        cov1[0, 0] = 4.0
        cov2 = np.eye(d, dtype=float)
        cov2[0, 0] = 0.25
        return [
            {"mu": mu, "cov": cov0},
            {"mu": mu, "cov": cov1},
            {"mu": mu, "cov": cov2},
        ]
    if dataset_name == "covshift_corr3":
        mu = np.zeros(d, dtype=float)
        cov0 = np.eye(d, dtype=float)
        cov1 = np.eye(d, dtype=float)
        cov1[0, 1] = 0.8
        cov1[1, 0] = 0.8
        cov2 = np.eye(d, dtype=float)
        cov2[0, 1] = -0.8
        cov2[1, 0] = -0.8
        return [
            {"mu": mu, "cov": cov0},
            {"mu": mu, "cov": cov1},
            {"mu": mu, "cov": cov2},
        ]
    raise ValueError(f"未知的 covshift_linear 配置: {dataset_name}")


def _build_covshift_linear_stream(dataset_name: str, seed: int, **kwargs: Any) -> StreamInfo:
    d = int(kwargs.get("n_features", COVSHIFT_LINEAR_DEFAULTS["n_features"]))
    seg_len = int(kwargs.get("segment_length", COVSHIFT_LINEAR_DEFAULTS["segment_length"]))
    w = np.asarray(kwargs.get("w", COVSHIFT_LINEAR_DEFAULTS["w"]), dtype=float)
    b = float(kwargs.get("b", COVSHIFT_LINEAR_DEFAULTS["b"]))
    if w.shape[0] != d:
        raise ValueError(f"covshift_linear w 维度不匹配: w={w.shape[0]} d={d}")
    segments = _covshift_linear_segments(dataset_name, d)
    rng = np.random.default_rng(seed)

    def iterator() -> Iterator[Tuple[Dict[str, float], int]]:
        for seg in segments:
            xs = rng.multivariate_normal(mean=seg["mu"], cov=seg["cov"], size=seg_len)
            logits = xs @ w + b
            ys = (logits > 0).astype(np.int64)
            for x, y in zip(xs, ys):
                yield {f"f{i}": float(x[i]) for i in range(d)}, int(y)

    feature_names = [f"f{i}" for i in range(d)]
    return StreamInfo(
        iterator_fn=iterator,
        n_features=d,
        n_classes=2,
        dataset_type="covshift_linear",
        dataset_name=dataset_name,
        feature_names=feature_names,
        classes=[0, 1],
    )


def _build_uspds_csv_stream(
    dataset_name: str,
    csv_path: str,
    label_col: Optional[str],
) -> StreamInfo:
    tabular = load_tabular_csv_dataset(csv_path=csv_path, label_col=label_col)

    def iterator() -> Iterator[Tuple[Dict[str, float], Any]]:
        return rv_stream.iter_pandas(X=tabular.features, y=tabular.labels)

    return StreamInfo(
        iterator_fn=iterator,
        n_features=len(tabular.feature_names),
        n_classes=len(tabular.classes),
        dataset_type="uspds_csv",
        dataset_name=dataset_name,
        feature_names=tabular.feature_names,
        classes=tabular.classes,
    )


def _build_insects_stream(dataset_name: str, seed: int) -> StreamInfo:
    variant = dataset_name.replace("insects_", "")
    dataset = Insects(variant=variant)

    def iterator() -> Iterator[Tuple[Dict[str, float], Any]]:
        for x, y in dataset:
            yield _ensure_float_dict(x), y

    n_classes = 24 if variant == "out-of-control" else 6
    return StreamInfo(
        iterator_fn=iterator,
        n_features=dataset.n_features,
        n_classes=n_classes,
        dataset_type="insects_river",
        dataset_name=dataset_name,
        feature_names=None,
        classes=None,
    )


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


def _format_synthetic_dict(x: Dict[Any, float]) -> Dict[str, float]:
    keys = sorted(x.keys())
    return {f"f{idx}": float(x[key]) for idx, key in enumerate(keys)}


def _ensure_float_dict(x: Dict[Any, Any]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in x.items()}


def generate_and_save_synth_stream(
    dataset_type: str,
    dataset_name: str,
    n_samples: int,
    seed: int,
    out_root: str = "data/synthetic",
    **drift_cfg: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    生成一个带概念漂移的合成流（SEA 或 Hyperplane），并把数据和漂移真值落盘。
    """
    dataset_type = dataset_type.lower()
    if dataset_type not in {"sea", "hyperplane", "sine", "stagger"}:
        raise ValueError("generate_and_save_synth_stream 仅支持 SEA/Hyperplane/Sine/STAGGER")

    segments, generator_meta = _build_synth_segments(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        n_samples=n_samples,
        seed=seed,
        drift_cfg=drift_cfg,
    )

    rows: List[Dict[str, Any]] = []
    drift_points: List[int] = []
    concept_segments: List[Dict[str, int]] = []
    feature_names: Optional[List[str]] = None
    t = 0
    for seg in segments:
        seg_start = t
        local_iter = _iter_segment(dataset_type, seg)
        for local_idx, (x, y) in enumerate(local_iter):
            feat = _format_synthetic_dict(x)
            if feature_names is None:
                feature_names = list(feat.keys())
            is_drift = 1 if seg["concept_id"] > 0 and local_idx == 0 else 0
            drift_id = len(drift_points) if is_drift else -1
            if is_drift:
                drift_points.append(t)
            row = {
                "t": t,
                "concept_id": seg["concept_id"],
                "is_drift": is_drift,
                "drift_id": drift_id,
                "y": int(y),
            }
            row.update(feat)
            rows.append(row)
            t += 1
        concept_segments.append(
            {"concept_id": seg["concept_id"], "start": seg_start, "end": t - 1}
        )
    df = pd.DataFrame(rows)
    feature_names = feature_names or [
        c
        for c in df.columns
        if c not in {"t", "concept_id", "is_drift", "drift_id", "y"}
    ]
    drifts_meta = [
        {
            "id": idx,
            "start": int(pos),
            "end": int(pos),
            "type": drift_cfg.get("drift_type", "abrupt"),
        }
        for idx, pos in enumerate(drift_points)
    ]
    meta = {
        "dataset_type": dataset_type,
        "dataset_name": dataset_name,
        "seed": seed,
        "n_samples": int(len(df)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "label_name": "y",
        "drift_type": drift_cfg.get("drift_type", "abrupt"),
        "drifts": drifts_meta,
        "concept_segments": concept_segments,
        "generator": generator_meta,
        "classes": [0, 1],
        "n_classes": 2,
    }
    out_dir = Path(out_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{dataset_name}__seed{seed}_data.parquet"
    df.to_parquet(data_path, index=False)
    meta_path = out_dir / f"{dataset_name}__seed{seed}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return df, meta


def load_saved_synth_stream(
    dataset_name: str,
    seed: int,
    data_root: str = "data/synthetic",
) -> Tuple[StreamInfo, Dict[str, Any]]:
    """从 parquet + meta.json 恢复一个 river 风格的 (x_dict, y) 流和 meta。"""
    base_dir = Path(data_root) / dataset_name
    data_path = base_dir / f"{dataset_name}__seed{seed}_data.parquet"
    meta_path = base_dir / f"{dataset_name}__seed{seed}_meta.json"
    if not data_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"未找到保存的流文件: {data_path}")
    df = pd.read_parquet(data_path)
    meta = json.loads(meta_path.read_text())
    feature_names = meta.get("feature_names") or [
        c for c in df.columns if c not in {"t", "concept_id", "is_drift", "drift_id", "y"}
    ]

    def iterator() -> Iterator[Tuple[Dict[str, float], Any]]:
        for row in df.itertuples(index=False):
            x = {fn: float(getattr(row, fn)) for fn in feature_names}
            yield x, getattr(row, "y")

    stream_info = StreamInfo(
        iterator_fn=iterator,
        n_features=len(feature_names),
        n_classes=int(meta.get("n_classes", 2)),
        dataset_type=meta.get("dataset_type", "synth_saved"),
        dataset_name=dataset_name,
        feature_names=feature_names,
        classes=meta.get("classes", [0, 1]),
    )
    return stream_info, meta


def _build_synth_segments(
    dataset_type: str,
    dataset_name: str,
    n_samples: int,
    seed: int,
    drift_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if dataset_type == "sea":
        config = SEA_CONFIGS.get(dataset_name)
        if config is None:
            raise ValueError(f"未知的 SEA 配置: {dataset_name}")
        variants = drift_cfg.get("concept_ids", config["concept_ids"])
        concept_length = int(drift_cfg.get("concept_length", config["concept_length"]))
        rng = np.random.default_rng(seed)
        segments: List[Dict[str, Any]] = []
        total = 0
        concept_idx = 0
        while total < n_samples:
            length = min(concept_length, n_samples - total)
            variant = variants[concept_idx % len(variants)]
            segments.append(
                {
                    "concept_id": concept_idx,
                    "length": length,
                    "dataset_params": {
                        "variant": variant,
                        "seed": int(rng.integers(0, 1_000_000)),
                        "noise": float(drift_cfg.get("noise", 0.0)),
                    },
                }
            )
            total += length
            concept_idx += 1
        generator_meta = {
            "library": "river",
            "class": "SEA",
            "params": {
                "concept_ids": variants,
                "concept_length": concept_length,
                "noise": float(drift_cfg.get("noise", 0.0)),
            },
        }
        return segments, generator_meta
    if dataset_type == "hyperplane":
        config = HYPERPLANE_CONFIGS.get(dataset_name)
        if config is None:
            raise ValueError(f"未知的 Hyperplane 配置: {dataset_name}")
        n_segments = int(drift_cfg.get("n_segments", 4))
        segment_length = int(drift_cfg.get("segment_length", max(1, n_samples // n_segments)))
        segments = []
        total = 0
        for concept_id in range(n_segments):
            remaining = n_samples - total
            if remaining <= 0:
                break
            if concept_id < n_segments - 1:
                length = min(segment_length, remaining)
            else:
                length = remaining
            params = {
                "seed": seed + concept_id * 177,
                "n_features": int(drift_cfg.get("n_features", config["n_features"])),
                "n_drift_features": int(
                    drift_cfg.get("n_drift_features", config["n_drift_features"])
                ),
                "mag_change": float(
                    drift_cfg.get(
                        "mag_change",
                        config["mag_change"] * (1 + 0.5 * concept_id),
                    )
                ),
                "sigma": float(drift_cfg.get("sigma", config["sigma"])),
                "noise_percentage": float(
                    drift_cfg.get("noise_percentage", config["noise_percentage"])
                ),
            }
            segments.append(
                {
                    "concept_id": concept_id,
                    "length": length,
                    "dataset_params": params,
                }
            )
            total += length
        generator_meta = {
            "library": "river",
            "class": "Hyperplane",
            "params": {
                "n_segments": len(segments),
                "segment_length": segment_length,
                "seed": seed,
            },
        }
        return segments, generator_meta
    if dataset_type == "sine":
        cfg = SINE_DEFAULT.get(dataset_name)
        classification_functions = drift_cfg.get(
            "classification_functions", cfg["classification_functions"] if cfg else [0, 1]
        )
        segment_length = int(
            drift_cfg.get("segment_length", cfg["segment_length"] if cfg else 1000)
        )
        segments = []
        total = 0
        concept_id = 0
        while total < n_samples:
            length = min(segment_length, n_samples - total)
            params = {
                "classification_function": classification_functions[concept_id % len(classification_functions)],
                "seed": seed + concept_id * 17,
                "balance_classes": bool(
                    drift_cfg.get("balance_classes", cfg["balance_classes"] if cfg else False)
                ),
                "has_noise": bool(drift_cfg.get("has_noise", cfg["has_noise"] if cfg else False)),
            }
            segments.append(
                {
                    "concept_id": concept_id,
                    "length": length,
                    "dataset_params": params,
                }
            )
            total += length
            concept_id += 1
        generator_meta = {
            "library": "river",
            "class": "Sine",
            "params": {
                "classification_functions": classification_functions,
                "segment_length": segment_length,
                "seed": seed,
            },
        }
        return segments, generator_meta
    if dataset_type == "stagger":
        cfg = STAGGER_DEFAULT.get(dataset_name)
        classification_functions = drift_cfg.get(
            "classification_functions", cfg["classification_functions"] if cfg else [0, 1, 2]
        )
        segment_length = int(
            drift_cfg.get("segment_length", cfg["segment_length"] if cfg else 1000)
        )
        segments = []
        total = 0
        concept_id = 0
        while total < n_samples:
            length = min(segment_length, n_samples - total)
            params = {
                "classification_function": classification_functions[concept_id % len(classification_functions)],
                "seed": seed + concept_id * 23,
                "balance_classes": bool(
                    drift_cfg.get("balance_classes", cfg["balance_classes"] if cfg else False)
                ),
            }
            segments.append(
                {
                    "concept_id": concept_id,
                    "length": length,
                    "dataset_params": params,
                }
            )
            concept_id += 1
            total += length
        generator_meta = {
            "library": "river",
            "class": "STAGGER",
            "params": {
                "classification_functions": classification_functions,
                "segment_length": segment_length,
                "seed": seed,
            },
        }
        return segments, generator_meta
    raise ValueError(f"未知的合成流类型: {dataset_type}")


def _iter_segment(dataset_type: str, seg: Dict[str, Any]) -> Iterator[Tuple[Dict[str, float], Any]]:
    if dataset_type == "sea":
        dataset = synth.SEA(**seg["dataset_params"])
    elif dataset_type == "hyperplane":
        dataset = synth.Hyperplane(**seg["dataset_params"])
    elif dataset_type == "sine":
        dataset = synth.Sine(**seg["dataset_params"])
    elif dataset_type == "stagger":
        dataset = synth.STAGGER(**seg["dataset_params"])
    else:
        raise ValueError(f"Unsupported dataset type for iter segment: {dataset_type}")
    return dataset.take(seg["length"])


def generate_default_abrupt_synth_datasets(
    seeds: List[int] | Tuple[int, ...] = (1,),
    out_root: str = "data/synthetic",
) -> None:
    """为默认的突变漂移合成流生成 parquet + meta。"""
    for dataset_name, cfg in DEFAULT_ABRUPT_SYNTH_DATASETS.items():
        for seed in seeds:
            out_dir = Path(out_root) / dataset_name
            data_path = out_dir / f"{dataset_name}__seed{seed}_data.parquet"
            if data_path.exists():
                continue
            generate_and_save_synth_stream(
                dataset_type=cfg["dataset_type"],
                dataset_name=dataset_name,
                n_samples=cfg["n_samples"],
                seed=seed,
                out_root=out_root,
                **cfg["drift_cfg"],
            )


def _quickcheck_covshift(seed: int, n_per_seg: int, datasets: Sequence[str]) -> None:
    print(f"[covshift_quickcheck] seed={seed} n_per_seg={n_per_seg}")
    for idx, name in enumerate(datasets):
        dataset_seed = int(seed + idx * 9973)
        rng = np.random.default_rng(dataset_seed)
        d = int(COVSHIFT_LINEAR_DEFAULTS["n_features"])
        w = np.asarray(COVSHIFT_LINEAR_DEFAULTS["w"], dtype=float)
        b = float(COVSHIFT_LINEAR_DEFAULTS["b"])
        segments = _covshift_linear_segments(name, d)
        print(f"[dataset] {name} seed={dataset_seed}")
        for seg_idx, seg in enumerate(segments):
            xs = rng.multivariate_normal(mean=seg["mu"], cov=seg["cov"], size=n_per_seg)
            logits = xs @ w + b
            ys = (logits > 0).astype(np.int64)
            f0 = xs[:, 0]
            f1 = xs[:, 1]
            corr = float(np.corrcoef(f0, f1)[0, 1])
            print(
                "  seg{idx}: f0_mean={mean:.3f} f0_var={var:.3f} "
                "corr_f0f1={corr:.3f} label_rate={rate:.3f}".format(
                    idx=seg_idx,
                    mean=float(f0.mean()),
                    var=float(f0.var()),
                    corr=corr,
                    rate=float(ys.mean()),
                )
            )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="data.streams 工具 CLI")
    parser.add_argument(
        "--cmd",
        choices=["generate_default_abrupt", "covshift_quickcheck"],
        required=True,
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[1], help="生成使用的随机种子列表")
    parser.add_argument("--out_root", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42, help="quickcheck 使用的随机种子")
    parser.add_argument("--n_per_seg", type=int, default=1000, help="每段采样数（quickcheck）")
    parser.add_argument(
        "--datasets",
        type=str,
        default="covshift_mean3,covshift_scale3,covshift_corr3",
        help="quickcheck 数据集列表（逗号分隔）",
    )
    args = parser.parse_args()
    if args.cmd == "generate_default_abrupt":
        generate_default_abrupt_synth_datasets(seeds=args.seeds, out_root=args.out_root)
        return
    if args.cmd == "covshift_quickcheck":
        datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
        if not datasets:
            raise ValueError("--datasets 不能为空")
        _quickcheck_covshift(int(args.seed), int(args.n_per_seg), datasets)
        return


if __name__ == "__main__":
    _cli()
