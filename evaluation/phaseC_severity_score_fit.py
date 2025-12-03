"""Phase C2：基于 per_drift_stats 拟合统一的漂移严重度得分。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FEATURE_COLS = ["x_error_pos", "x_div_pos", "x_entropy_pos"]
Z_COLS = ["z_error_pos", "z_div_pos", "z_entropy_pos"]
MANUAL_WEIGHTS = np.array([0.6, 0.3, 0.1], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C2：拟合统一的漂移严重度分数 S")
    parser.add_argument(
        "--per_drift_path",
        type=str,
        default="results/phaseC_severity_analysis/per_drift_stats.csv",
        help="Phase C1 生成的 per_drift 统计 CSV 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phaseC_severity_score",
        help="输出目录，保存带严重度的表与相关性汇总",
    )
    parser.add_argument(
        "--min_drop",
        type=float,
        default=0.0,
        help="拟合与标准化所用样本的 drop_min_acc 下界，用于过滤纯噪声漂移",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        default=True,
        help="是否对特征做标准化（默认开启）",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="关闭特征标准化",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（占位）")
    return parser.parse_args()


def load_per_drift_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"per_drift_stats 不存在：{path}")
    df = pd.read_csv(path)
    required_cols = [
        "delta_student_error_rate",
        "delta_teacher_entropy",
        "delta_divergence_js",
        "drop_min_acc",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"per_drift_stats 缺少必要列：{col}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_error_pos"] = np.maximum(0.0, out["delta_student_error_rate"])
    out["x_div_pos"] = np.maximum(0.0, out["delta_divergence_js"])
    out["x_entropy_pos"] = np.maximum(0.0, -out["delta_teacher_entropy"])
    return out


def select_training_mask(df: pd.DataFrame, min_drop: float) -> pd.Series:
    mask = df["drop_min_acc"].notna() & (df["drop_min_acc"] >= min_drop)
    for col in FEATURE_COLS:
        mask &= df[col].notna()
    return mask


def compute_scaler(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for col in FEATURE_COLS:
        series = train_df[col].dropna()
        if series.empty:
            stats[col] = (0.0, 0.0)
        else:
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            stats[col] = (mean, std)
    return stats


def add_standardized_features(df: pd.DataFrame, scaler: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    for src, dest in zip(FEATURE_COLS, Z_COLS):
        mean, std = scaler.get(src, (0.0, 0.0))
        if std <= 1e-12:
            df[dest] = 0.0
        else:
            df[dest] = (df[src] - mean) / std
    return df


def compute_manual_severity(df: pd.DataFrame, use_standardize: bool) -> pd.Series:
    cols = Z_COLS if use_standardize else FEATURE_COLS
    values = df[cols].to_numpy(dtype=np.float64)
    severity = values @ MANUAL_WEIGHTS
    print(
        f"[info] severity_manual uses standardize={use_standardize}, "
        f"weights={tuple(MANUAL_WEIGHTS)}"
    )
    return severity


def fit_severity_linear(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_drop: float,
) -> Tuple[np.ndarray, int]:
    mask = df["drop_min_acc"].notna() & (df["drop_min_acc"] >= min_drop)
    for col in feature_cols:
        mask &= df[col].notna()
    train_df = df.loc[mask]
    n_train = len(train_df)
    if n_train < len(feature_cols):
        print(
            f"[warn] 训练样本不足（n={n_train}，特征维度={len(feature_cols)}），"
            "回退到手工权重"
        )
        return MANUAL_WEIGHTS.copy(), n_train
    X = train_df[feature_cols].to_numpy(dtype=np.float64)
    y = train_df["drop_min_acc"].to_numpy(dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(
        "[info] fitted severity_lr: "
        f"beta1={beta[0]:.4f}, beta2={beta[1]:.4f}, beta3={beta[2]:.4f}, "
        f"n_train={n_train}"
    )
    return beta, n_train


def compute_linear_severity(df: pd.DataFrame, feature_cols: Sequence[str], beta: np.ndarray) -> pd.Series:
    values = df[feature_cols].to_numpy(dtype=np.float64)
    severity = values @ beta
    return severity


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    arr_a = a.loc[mask].to_numpy(dtype=np.float64)
    arr_b = b.loc[mask].to_numpy(dtype=np.float64)
    if np.allclose(arr_a.std(), 0.0) or np.allclose(arr_b.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    groups = list(sorted(df["dataset_name"].dropna().unique()))
    groups.append("ALL")

    for dataset in groups:
        if dataset == "ALL":
            subset = df
        else:
            subset = df[df["dataset_name"] == dataset]
        if subset.empty:
            continue
        drop_col = subset["drop_min_acc"]
        stats = {
            "dataset_name": dataset,
            "n_drifts_used": int(drop_col.notna().sum()),
            "corr_delta_error__drop_min_acc": pearson_corr(
                subset["delta_student_error_rate"], drop_col
            ),
            "corr_severity_manual__drop_min_acc": pearson_corr(
                subset["severity_manual"], drop_col
            ),
            "corr_severity_lr__drop_min_acc": pearson_corr(
                subset["severity_lr"], drop_col
            ),
        }
        rows.append(stats)
        print(
            f"[info] dataset={dataset}: n={stats['n_drifts_used']}, "
            f"corr(delta_error, drop)={stats['corr_delta_error__drop_min_acc']:.3f}, "
            f"corr(S_manual, drop)={stats['corr_severity_manual__drop_min_acc']:.3f}, "
            f"corr(S_lr, drop)={stats['corr_severity_lr__drop_min_acc']:.3f}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    per_drift_df = load_per_drift_stats(Path(args.per_drift_path))
    per_drift_df = build_features(per_drift_df)
    train_mask = select_training_mask(per_drift_df, args.min_drop)

    if args.standardize:
        scaler = compute_scaler(per_drift_df.loc[train_mask])
        per_drift_df = add_standardized_features(per_drift_df, scaler)
        feature_for_fit = Z_COLS
    else:
        scaler = None
        feature_for_fit = FEATURE_COLS

    per_drift_df["severity_manual"] = compute_manual_severity(per_drift_df, args.standardize)
    beta, n_train = fit_severity_linear(per_drift_df, feature_for_fit, args.min_drop)
    if args.standardize and scaler is None:
        raise RuntimeError("标准化标志开启但未生成 scaler")
    per_drift_df["severity_lr"] = compute_linear_severity(per_drift_df, feature_for_fit, beta)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "severity_scores_per_drift.csv"
    per_drift_df.to_csv(scores_path, index=False)

    corr_df = compute_correlations(per_drift_df)
    summary_path = output_dir / "severity_corr_summary.csv"
    corr_df.to_csv(summary_path, index=False)

    print(f"[done] severity scores saved to {scores_path}")
    print(f"[done] correlation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
