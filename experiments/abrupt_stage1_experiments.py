"""突变漂移第一阶段批量实验脚本。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from typing import Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import streams as data_streams

DEFAULT_SYNTH_DATASETS = ["sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"]
DEFAULT_REAL_DATASETS = ["INSECTS_abrupt_balanced"]
DEFAULT_MODELS = ["baseline_student", "mean_teacher", "ts_drift_adapt"]
DEFAULT_SEEDS = [1, 2, 3]


SYNTH_DATASET_TYPE = {
    "sea_abrupt4": "synth_saved",
    "sine_abrupt4": "synth_saved",
    "stagger_abrupt3": "synth_saved",
}

REAL_DATASET_PARAMS = {
    "INSECTS_abrupt_balanced": {
        "dataset_type": "insects_real",
        "dataset_name": "INSECTS_abrupt_balanced",
        "csv_path": "datasets/real/INSECTS_abrupt_balanced.csv",
        "label_col": None,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="突变漂移 Stage1 实验批量运行")
    parser.add_argument("--device", default="cuda", help="传递给 run_experiment.py 的设备")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_SYNTH_DATASETS + DEFAULT_REAL_DATASETS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--log_root", default="logs")
    parser.add_argument("--synth_root", default="data/synthetic")
    parser.add_argument("--run_script", default="run_experiment.py")
    parser.add_argument("--overwrite", action="store_true", help="若日志存在则重新运行")
    parser.add_argument("--n_steps", type=int, default=1000, help="传递给 run_experiment.py 的 n_steps")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--initial_alpha", type=float, default=0.99)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.9)
    return parser.parse_args()


def ensure_synth_datasets(seeds: Iterable[int], out_root: str) -> None:
    data_streams.generate_default_abrupt_synth_datasets(list(seeds), out_root=out_root)


def build_command(
    args: argparse.Namespace,
    dataset_name: str,
    model_variant: str,
    seed: int,
) -> List[str]:
    log_path = Path(args.log_root) / dataset_name / f"{dataset_name}__{model_variant}__seed{seed}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_spec = {}
    if dataset_name in SYNTH_DATASET_TYPE:
        dataset_spec = {
            "dataset_type": SYNTH_DATASET_TYPE[dataset_name],
            "dataset_name": dataset_name,
        }
    elif dataset_name in REAL_DATASET_PARAMS:
        dataset_spec = REAL_DATASET_PARAMS[dataset_name]
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    cmd = [
        "python",
        args.run_script,
        "--dataset_type",
        dataset_spec["dataset_type"],
        "--dataset_name",
        dataset_spec["dataset_name"],
        "--model_variant",
        model_variant,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--n_steps",
        str(args.n_steps),
        "--batch_size",
        str(args.batch_size),
        "--labeled_ratio",
        str(args.labeled_ratio),
        "--initial_lr",
        str(args.initial_lr),
        "--initial_alpha",
        str(args.initial_alpha),
        "--lambda_u",
        str(args.lambda_u),
        "--tau",
        str(args.tau),
        "--log_path",
        str(log_path),
    ]
    if "csv_path" in dataset_spec and dataset_spec["csv_path"]:
        cmd.extend(["--csv_path", dataset_spec["csv_path"]])
    if dataset_spec.get("label_col"):
        cmd.extend(["--label_col", dataset_spec["label_col"]])
    return cmd


def main() -> None:
    args = parse_args()
    ensure_synth_datasets(args.seeds, args.synth_root)
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                log_path = Path(args.log_root) / dataset / f"{dataset}__{model}__seed{seed}.csv"
                if log_path.exists() and not args.overwrite:
                    continue
                cmd = build_command(args, dataset, model, seed)
                print(f"[RUN] {' '.join(cmd)}")
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
