"""离线漂移检测器网格搜索。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import math
import sys

import pandas as pd
from river import drift

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.drift_metrics import compute_detection_metrics


SYNTH_DATASETS = {"sea_abrupt4", "sine_abrupt4", "stagger_abrupt3"}
REAL_DATASETS = {"INSECTS_abrupt_balanced"}


@dataclass
class DetectorConfig:
    """定义单个检测器组合。"""

    signal_name: str
    detector_type: str
    params: Dict[str, Any]
    value_scale: float = 1.0


def get_drift_flag(detector: drift.base.DriftDetector) -> bool:
    """
    统一从 river 检测器中读取漂移标志。

    - 依据当前 river 发行版（在本地验证）推荐的 `drift_detected` 属性；
    - 为兼容旧版本，若无该属性则回退到 `change_detected`。
    """
    if hasattr(detector, "drift_detected"):
        return bool(getattr(detector, "drift_detected"))
    if hasattr(detector, "change_detected"):
        return bool(getattr(detector, "change_detected"))
    return False


def load_synth_meta(dataset_name: str, seed: int, root: str = "data/synthetic") -> Dict[str, Any]:
    """
    读取合成流 meta.json。
    """
    meta_path = Path(root) / dataset_name / f"{dataset_name}__seed{seed}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到合成流 meta：{meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_insects_meta(meta_path: str = "datasets/real/INSECTS_abrupt_balanced.json") -> Dict[str, Any]:
    """读取 INSECTS meta。"""
    path = Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 INSECTS meta：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_log(
    dataset_name: str,
    model_variant: str,
    seed: int,
    logs_root: str = "logs",
) -> pd.DataFrame:
    """读取训练日志 CSV。"""
    log_path = Path(logs_root) / dataset_name / f"{dataset_name}__{model_variant}__seed{seed}.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"未找到日志：{log_path}")
    return pd.read_csv(log_path)


def build_default_grid() -> List[DetectorConfig]:
    """构造默认 detector 组合。"""
    grid: List[DetectorConfig] = []
    for delta in [0.1, 0.05, 0.01, 0.005, 0.001]:
        grid.append(
            DetectorConfig(
                signal_name="student_error_rate",
                detector_type="adwin",
                params={"delta": delta},
            )
        )
    for threshold in [0.05, 0.1, 0.2, 0.3]:
        grid.append(
            DetectorConfig(
                signal_name="student_error_rate",
                detector_type="ph",
                params={"delta": 0.005, "alpha": 0.15, "threshold": threshold, "min_instances": 25},
            )
        )
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        grid.append(
            DetectorConfig(
                signal_name="teacher_entropy",
                detector_type="ph",
                params={"delta": 0.01, "alpha": 0.5, "threshold": threshold, "min_instances": 20},
            )
        )
    for delta in [0.1, 0.05]:
        grid.append(
            DetectorConfig(
                signal_name="teacher_entropy",
                detector_type="adwin",
                params={"delta": delta},
            )
        )
    for scale in [10.0, 25.0, 50.0, 100.0]:
        for threshold in [0.05, 0.1, 0.2, 0.5, 1.0]:
            grid.append(
                DetectorConfig(
                    signal_name="divergence_js",
                    detector_type="ph",
                params={"delta": 0.005, "alpha": 0.1, "threshold": threshold, "min_instances": 30},
                    value_scale=scale,
                )
            )
    return grid


def run_offline_detector(
    values: Sequence[float],
    sample_idxs: Sequence[int],
    cfg: DetectorConfig,
) -> List[int]:
    """在日志信号上复现检测器输出。"""
    if len(values) != len(sample_idxs):
        raise ValueError("values 与 sample_idxs 长度不一致")
    detector: drift.base.DriftDetector
    if cfg.detector_type == "adwin":
        detector = drift.ADWIN(**cfg.params)
    elif cfg.detector_type == "ph":
        detector = drift.PageHinkley(**cfg.params)
    else:
        raise ValueError(f"未知的 detector_type: {cfg.detector_type}")
    detections: List[int] = []
    for value, idx in zip(values, sample_idxs):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        v_scaled = cfg.value_scale * float(value)
        detector.update(v_scaled)
        if get_drift_flag(detector):
            detections.append(int(idx))
    return detections


def evaluate_one_config(
    dataset_name: str,
    model_variant: str,
    seed: int,
    cfg: DetectorConfig,
    logs_root: str = "logs",
    synth_root: str = "data/synthetic",
    insects_meta_path: str = "datasets/real/INSECTS_abrupt_balanced.json",
) -> Dict[str, Any]:
    """运行离线 detector 并返回指标。"""
    df = load_log(dataset_name, model_variant, seed, logs_root)
    signal = df[cfg.signal_name].astype(float).tolist()
    sample_idxs = df["sample_idx"].astype(int).tolist()
    if dataset_name in SYNTH_DATASETS:
        meta = load_synth_meta(dataset_name, seed, synth_root)
        gt_drifts = [int(d["start"]) for d in meta.get("drifts", [])]
        T = int(meta.get("n_samples", sample_idxs[-1] + 1))
    elif dataset_name in REAL_DATASETS:
        meta = load_insects_meta(insects_meta_path)
        gt_drifts = [int(pos) for pos in meta.get("positions", [])]
        T = int(sample_idxs[-1]) + 1 if sample_idxs else 0
    else:
        raise ValueError(f"未知数据集：{dataset_name}")
    detections = run_offline_detector(signal, sample_idxs, cfg)
    metrics = compute_detection_metrics(gt_drifts, detections, T)
    return {
        "dataset_name": dataset_name,
        "model_variant": model_variant,
        "seed": seed,
        "signal_name": cfg.signal_name,
        "detector_type": cfg.detector_type,
        "params": json.dumps(cfg.params, sort_keys=True),
        "value_scale": cfg.value_scale,
        "n_gt_drifts": len(gt_drifts),
        "n_detected": len(detections),
        "MDR": metrics["MDR"],
        "MTD": metrics["MTD"],
        "MTFA": metrics["MTFA"],
        "MTR": metrics["MTR"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 detector 网格搜索")
    parser.add_argument(
        "--datasets",
        default="sea_abrupt4,sine_abrupt4,stagger_abrupt3,INSECTS_abrupt_balanced",
        help="以逗号分隔的数据集列表",
    )
    parser.add_argument("--model_variant", default="baseline_student")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="需要评估的随机种子",
    )
    parser.add_argument("--logs_root", default="logs")
    parser.add_argument("--synth_root", default="data/synthetic")
    parser.add_argument("--insects_meta", default="datasets/real/INSECTS_abrupt_balanced.json")
    parser.add_argument("--out_csv", default="results/offline_detector_grid.csv")
    parser.add_argument("--out_md_dir", default="results/offline_md")
    parser.add_argument("--top_k", type=int, default=10, help="每个数据集输出的配置数量")
    parser.add_argument(
        "--debug_sanity",
        action="store_true",
        help="运行简单的漂移检测 sanity check（不执行网格搜索）",
    )
    return parser.parse_args()


def ensure_list(value: str) -> List[str]:
    """将逗号分隔字符串转为列表。"""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def save_markdown_tables(df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    """按数据集写入 Markdown 摘要。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, group in df.groupby("dataset_name"):
        successful = group[group["MDR"] < 1.0].copy()
        target = successful if not successful.empty else group.copy()
        target = target.sort_values(
            by=["MDR", "MTD", "MTR"],
            ascending=[True, True, False],
            na_position="last",
        ).head(top_k)
        lines = [
            f"# {dataset} Top Configs",
            "",
            "| signal | detector | params | scale | MDR | MTD | MTFA | MTR | n_detected |",
            "|--------|----------|--------|-------|-----|-----|------|-----|------------|",
        ]
        for _, row in target.iterrows():
            lines.append(
                "| {signal} | {det} | `{params}` | {scale:.2f} | {mdr:.3f} | {mtd:.1f} | {mtfa:.1f} | {mtr:.3f} | {n_det} |".format(
                    signal=row["signal_name"],
                    det=row["detector_type"],
                    params=row["params"],
                    scale=row["value_scale"],
                    mdr=row["MDR"] if not math.isnan(row["MDR"]) else float("nan"),
                    mtd=row["MTD"] if not math.isnan(row["MTD"]) else float("nan"),
                    mtfa=row["MTFA"] if not math.isnan(row["MTFA"]) else float("nan"),
                    mtr=row["MTR"] if not math.isnan(row["MTR"]) else float("nan"),
                    n_det=int(row["n_detected"]),
                )
            )
        (out_dir / f"{dataset}_top_configs.md").write_text("\n".join(lines), encoding="utf-8")


def print_best_summary(df: pd.DataFrame) -> None:
    """在控制台打印每个数据集的最优配置。"""
    for dataset, group in df.groupby("dataset_name"):
        sorted_group = group.sort_values(
            by=["MDR", "MTD", "MTR"],
            ascending=[True, True, False],
            na_position="last",
        )
        if sorted_group.empty:
            continue
        best = sorted_group.iloc[0]
        print(
            f"[{dataset}] best config: signal={best['signal_name']}, detector={best['detector_type']}, "
            f"params={best['params']}, scale={best['value_scale']}, "
            f"MDR={best['MDR']:.3f}, MTD={best['MTD']}, MTFA={best['MTFA']}, MTR={best['MTR']}"
        )


def debug_sanity_check() -> None:
    """简单 sanity check，验证 detector 会触发漂移。"""
    values = [0.0] * 500 + [1.0] * 500
    adwin = drift.ADWIN(delta=0.01)
    ph = drift.PageHinkley(delta=0.005, threshold=5.0, min_instances=30)

    adwin_detect = None
    for idx, val in enumerate(values):
        adwin.update(val)
        if get_drift_flag(adwin):
            adwin_detect = idx
            break
    ph_detect = None
    for idx, val in enumerate(values):
        ph.update(val)
        if get_drift_flag(ph):
            ph_detect = idx
            break
    print(f"[debug] ADWIN first drift at idx={adwin_detect}")
    print(f"[debug] PageHinkley first drift at idx={ph_detect}")


def main() -> None:
    args = parse_args()
    if args.debug_sanity:
        debug_sanity_check()
        return
    datasets = ensure_list(args.datasets)
    if not datasets:
        raise ValueError("至少需要一个数据集")
    grid = build_default_grid()
    results: List[Dict[str, Any]] = []
    for dataset_name in datasets:
        for seed in args.seeds:
            for cfg in grid:
                try:
                    record = evaluate_one_config(
                        dataset_name=dataset_name,
                        model_variant=args.model_variant,
                        seed=seed,
                        cfg=cfg,
                        logs_root=args.logs_root,
                        synth_root=args.synth_root,
                        insects_meta_path=args.insects_meta,
                    )
                    results.append(record)
                except FileNotFoundError as exc:
                    print(f"[skip] {exc}")
                except Exception as exc:
                    print(f"[error] dataset={dataset_name}, seed={seed}, cfg={cfg}: {exc}")
    if not results:
        raise RuntimeError("没有任何可用结果，请确认日志与 meta 已就绪。")
    df = pd.DataFrame(results)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    save_markdown_tables(df, Path(args.out_md_dir), top_k=args.top_k)
    print_best_summary(df)


if __name__ == "__main__":
    main()
