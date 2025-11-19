"""根据日志与真值评估合成流与真实流的漂移指标。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.drift_metrics import compute_detection_metrics, compute_lpd


def eval_synthetic(
    dataset_name: str,
    seed: int,
    model_variant: str,
    logs_root: str = "logs",
    synth_root: str = "data/synthetic",
) -> Tuple[Dict[str, float], float]:
    meta_path = Path(synth_root) / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text())
    gt_drifts = [int(d["start"]) for d in meta.get("drifts", [])]
    T = int(meta["n_samples"])

    log_path = Path(logs_root) / dataset_name / f"{dataset_name}__{model_variant}__seed{seed}.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"缺少日志文件: {log_path}")
    df = pd.read_csv(log_path)
    detections = df.loc[df["drift_flag"] == 1, "seen_samples"].astype(int).tolist()
    det_metrics = compute_detection_metrics(gt_drifts, detections, T)
    acc_d = float(df["metric_accuracy"].iloc[-1])
    return det_metrics, acc_d


def eval_real(
    dataset_name: str,
    seed: int,
    model_variant: str,
    baseline_variant: str,
    logs_root: str = "logs",
) -> Dict[str, Any]:
    log_path_d = Path(logs_root) / dataset_name / f"{dataset_name}__{model_variant}__seed{seed}.csv"
    log_path_0 = Path(logs_root) / dataset_name / f"{dataset_name}__{baseline_variant}__seed{seed}.csv"
    if not log_path_d.exists() or not log_path_0.exists():
        raise FileNotFoundError("缺少对比日志文件，请确认实验已完成。")
    df_d = pd.read_csv(log_path_d)
    df_0 = pd.read_csv(log_path_0)
    acc_d = float(df_d["metric_accuracy"].iloc[-1])
    acc0 = float(df_0["metric_accuracy"].iloc[-1])
    n_drifts = int(df_d["drift_flag"].sum())
    lpd = compute_lpd(acc_d, acc0, n_drifts)
    return {"acc_d": acc_d, "acc0": acc0, "n_drifts": n_drifts, "lpd": lpd}


def main() -> None:
    parser = argparse.ArgumentParser(description="评估漂移检测性能")
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_variant", required=True)
    parser.add_argument("--baseline_variant", default="baseline_student")
    parser.add_argument("--logs_root", default="logs")
    parser.add_argument("--synth_root", default="data/synthetic")
    args = parser.parse_args()

    if args.mode == "synthetic":
        det_metrics, acc_d = eval_synthetic(
            dataset_name=args.dataset_name,
            seed=args.seed,
            model_variant=args.model_variant,
            logs_root=args.logs_root,
            synth_root=args.synth_root,
        )
        print(
            f"[synthetic][{args.dataset_name}][{args.model_variant}][seed={args.seed}] "
            f"metrics={det_metrics} acc_d={acc_d:.4f}"
        )
    else:
        res = eval_real(
            dataset_name=args.dataset_name,
            seed=args.seed,
            model_variant=args.model_variant,
            baseline_variant=args.baseline_variant,
            logs_root=args.logs_root,
        )
        print(
            f"[real][{args.dataset_name}][{args.model_variant}][seed={args.seed}] "
            f"acc_d={res['acc_d']:.4f} acc0={res['acc0']:.4f} "
            f"n_drifts={res['n_drifts']} lpd={res['lpd']:.6f}"
        )


if __name__ == "__main__":
    main()
