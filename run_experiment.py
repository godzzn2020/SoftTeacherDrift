"""运行师生 EMA 半监督流式实验。"""

from __future__ import annotations

import argparse
import itertools
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from river import metrics

from data import streams
from drift import detectors
from models.teacher_student import TeacherStudentModel
from scheduler.hparam_scheduler import HParams, SchedulerState
from training.loop import (
    FeatureVectorizer,
    LabelEncoder,
    TrainingConfig,
    run_training_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-Student EMA 半监督流式实验")
    parser.add_argument("--dataset_type", choices=["sea", "hyperplane", "uspds"], required=True)
    parser.add_argument("--dataset_path", type=str, help="USPDS CSV 路径")
    parser.add_argument("--label_col", type=str, help="USPDS 标签列名")
    parser.add_argument("--n_concepts", type=int, default=4, help="SEA 概念数量")
    parser.add_argument("--concept_length", type=int, default=500, help="每个概念的样本数")
    parser.add_argument("--n_samples", type=int, default=5000, help="Hyperplane 样本数")
    parser.add_argument("--drift_speed", type=float, default=0.01, help="Hyperplane 漂移速度")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--initial_alpha", type=float, default=0.99)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream_iterable, classes = build_stream(args)
    first_sample, stream_with_first = peek_stream(stream_iterable)
    input_dim = infer_dim(first_sample[0])
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    label_encoder = LabelEncoder(classes=classes)
    num_classes = label_encoder.num_classes
    vectorizer = FeatureVectorizer(input_dim=input_dim)

    model = TeacherStudentModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        device=torch.device(args.device),
    )
    optimizer = torch.optim.Adam(model.student.parameters(), lr=args.initial_lr)
    monitor = detectors.build_default_monitor()
    initial_hparams = HParams(
        alpha=args.initial_alpha,
        lr=args.initial_lr,
        lambda_u=args.lambda_u,
        tau=args.tau,
    )
    scheduler_state = SchedulerState(base_hparams=initial_hparams)
    metric = metrics.Accuracy()

    batched = streams.batch_stream(
        stream=stream_with_first,
        batch_size=args.batch_size,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
    )
    config = TrainingConfig(n_steps=args.n_steps, device=args.device, log_path=args.log_path)
    result = run_training_loop(
        batch_iter=batched,
        model=model,
        optimizer=optimizer,
        drift_monitor=monitor,
        scheduler_state=scheduler_state,
        metric=metric,
        initial_hparams=initial_hparams,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        config=config,
    )
    print(f"最终准确率: {result['metric']:.4f}")
    if args.log_path:
        print(f"训练日志已保存至 {args.log_path}")


def build_stream(args: argparse.Namespace) -> Tuple[Iterable[Tuple[Any, Any]], Sequence[Any]]:
    """根据参数构建数据流及类别集合。"""
    if args.dataset_type == "sea":
        iterable = streams.make_sea_stream(
            n_concepts=args.n_concepts,
            concept_length=args.concept_length,
            seed=args.seed,
        )
        classes: Sequence[Any] = [0, 1]
        return iterable, classes
    if args.dataset_type == "hyperplane":
        iterable = streams.make_hyperplane_stream(
            n_samples=args.n_samples,
            drift_speed=args.drift_speed,
            seed=args.seed,
        )
        classes = [0, 1]
        return iterable, classes
    if args.dataset_type == "uspds":
        if not args.dataset_path or not args.label_col:
            raise ValueError("使用 USPDS 数据集需要提供 --dataset_path 与 --label_col")
        uspds_stream = streams.load_uspds_stream(args.dataset_path, args.label_col)
        return uspds_stream, uspds_stream.classes
    raise ValueError(f"未知数据类型 {args.dataset_type}")


def peek_stream(
    iterable: Iterable[Tuple[Any, Any]],
) -> Tuple[Tuple[Any, Any], Iterable[Tuple[Any, Any]]]:
    """提取首个样本并返回带回放的新迭代器。"""
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("数据流为空") from exc
    new_iter = itertools.chain([first], iterator)
    return first, new_iter


def infer_dim(sample: Any) -> int:
    """从样本推断特征维数。"""
    x = sample
    if isinstance(x, np.ndarray):
        return int(x.shape[0])
    if isinstance(x, dict):
        return len(x)
    if isinstance(x, (list, tuple)):
        return len(x)
    raise TypeError(f"无法解析样本类型: {type(x)}")


def parse_hidden_dims(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else [128, 64]


if __name__ == "__main__":
    main()

