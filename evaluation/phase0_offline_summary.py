"""汇总 Phase0 离线监督训练结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from soft_drift.utils.run_paths import create_experiment_run

EXPERIMENT_NAME = "phase0_offline_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase0 离线监督结果汇总")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="results/phase0_offline_supervised",
        help="Phase0 训练结果根目录",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="逗号分隔的数据集过滤（缺省表示全部）",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="附加到汇总 run_id 的别名",
    )
    parser.add_argument("--run_id", type=str, default=None, help="覆盖汇总脚本的 run_id")
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="汇总输出根目录",
    )
    parser.add_argument(
        "--logs_root",
        type=str,
        default="logs",
        help="为保持接口一致而需要的 logs 根目录",
    )
    return parser.parse_args()


def parse_str_list(spec: Optional[str]) -> Optional[List[str]]:
    if spec is None:
        return None
    tokens = [item.strip() for item in spec.split(",") if item.strip()]
    return tokens or None


def scan_phase0_runs(root_dir: Path, dataset_filter: Optional[List[str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    datasets = [p for p in root_dir.iterdir() if p.is_dir()]
    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        if dataset_name == "summary":
            continue
        if dataset_filter and dataset_name not in dataset_filter:
            continue
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_variant = model_dir.name
            for seed_dir in model_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue
                try:
                    seed = int(seed_dir.name.replace("seed", ""))
                except ValueError:
                    seed = -1
                for run_dir in seed_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    summary_path = run_dir / "summary.json"
                    if not summary_path.exists():
                        continue
                    with summary_path.open("r", encoding="utf-8") as f:
                        summary = json.load(f)
                    record = {
                        "dataset_name": dataset_name,
                        "model_variant": model_variant,
                        "seed": summary.get("seed", seed),
                        "run_id": run_dir.name,
                    }
                    record.update(summary)
                    records.append(record)
    return records


def summarize_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = df.groupby(["dataset_name", "model_variant"], dropna=False)
    for (dataset, model_variant), grp in grouped:
        row: Dict[str, object] = {
            "dataset_name": dataset,
            "model_variant": model_variant,
            "runs": len(grp),
        }
        for col in ["best_val_acc", "test_acc", "labeled_ratio"]:
            if col in grp.columns:
                row[f"{col}_mean"] = float(grp[col].mean())
                row[f"{col}_std"] = float(grp[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dataset_filter = parse_str_list(args.datasets)
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir {root_dir} 不存在")
    records = scan_phase0_runs(root_dir, dataset_filter)
    if not records:
        print("[warn] 未找到任何 summary.json")
        return
    df = pd.DataFrame(records)
    experiment_run = create_experiment_run(
        experiment_name=EXPERIMENT_NAME,
        results_root=args.results_root,
        logs_root=args.logs_root,
        run_name=args.run_name,
        run_id=args.run_id,
    )
    summary_dir = experiment_run.summary_dir()
    run_level_path = summary_dir / "run_level_metrics.csv"
    df.to_csv(run_level_path, index=False)
    dataset_summary = summarize_by_dataset(df)
    dataset_summary_path = summary_dir / "summary_by_dataset.csv"
    dataset_summary.to_csv(dataset_summary_path, index=False)
    print(f"[done] run-level metrics -> {run_level_path}")
    print(f"[done] dataset summary -> {dataset_summary_path}")


if __name__ == "__main__":
    main()
