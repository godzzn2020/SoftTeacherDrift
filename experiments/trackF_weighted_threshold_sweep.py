#!/usr/bin/env python
"""Track F：weighted voting 阈值扫描（生成 trade-off 曲线 + 汇总 CSV）。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.streams import generate_default_abrupt_synth_datasets
from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track F：weighted voting 阈值扫描")
    parser.add_argument("--datasets", type=str, default="sea_abrupt4,stagger_abrupt3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--thetas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    parser.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="trackF")
    parser.add_argument("--out_csv", type=str, default="scripts/TRACKF_THRESHOLD_SWEEP.csv")
    parser.add_argument("--fig_dir", type=str, default="scripts/figures")
    parser.add_argument("--match_tolerance", type=int, default=500)
    parser.add_argument("--min_separation", type=int, default=200)
    parser.add_argument("--synthetic_root", type=str, default="data/synthetic")
    return parser.parse_args()


def parse_weights(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--weights 格式错误：{token}（期望 key=value）")
        key, value = token.split("=", 1)
        weights[key.strip()] = float(value.strip())
    if not weights:
        raise ValueError("--weights 不能为空")
    return weights


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]  # type: ignore[arg-type]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]  # type: ignore[arg-type]
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return var**0.5


def _merge_events(events: Sequence[int], min_sep: int) -> List[int]:
    if not events:
        return []
    ev = sorted(int(x) for x in events)
    merged = [ev[0]]
    for x in ev[1:]:
        if x - merged[-1] < min_sep:
            continue
        merged.append(x)
    return merged


def compute_metrics_tolerance(
    gt_drifts: Sequence[int],
    detections: Sequence[int],
    horizon: int,
    tolerance: int,
) -> Dict[str, Optional[float]]:
    gt = sorted(int(d) for d in gt_drifts)
    dets = sorted(int(d) for d in detections)
    if not gt or horizon <= 0:
        return {"MDR": None, "MTD": None, "MTFA": None, "MTR": None}
    matched: set[int] = set()
    delays: List[int] = []
    missed = 0
    for drift in gt:
        match_idx: Optional[int] = None
        for idx, det in enumerate(dets):
            if idx in matched:
                continue
            if det < drift:
                continue
            if det <= drift + tolerance:
                match_idx = idx
                break
            if det > drift + tolerance:
                break
        if match_idx is None:
            missed += 1
        else:
            matched.add(match_idx)
            delays.append(max(0, dets[match_idx] - drift))
    false_dets = [dets[i] for i in range(len(dets)) if i not in matched]
    mdr = missed / len(gt) if gt else None
    mtd = (sum(delays) / len(delays)) if delays else None
    if len(false_dets) >= 2:
        gaps = [b - a for a, b in zip(false_dets[:-1], false_dets[1:])]
        mtfa = sum(gaps) / len(gaps)
    elif len(false_dets) == 1:
        mtfa = float(horizon - false_dets[0]) if horizon > 0 else None
    else:
        mtfa = None
    if mdr is None or mtd is None or mtfa is None or mdr >= 1.0 or mtd <= 0:
        mtr = None
    else:
        mtr = mtfa / (mtd * (1 - mdr))
    return {
        "MDR": float(mdr) if mdr is not None else None,
        "MTD": float(mtd) if mtd is not None else None,
        "MTFA": float(mtfa) if mtfa is not None else None,
        "MTR": float(mtr) if mtr is not None else None,
    }


def load_synth_meta(synth_root: Path, dataset: str, seed: int) -> Tuple[List[int], int]:
    meta_path = synth_root / dataset / f"{dataset}__seed{seed}_meta.json"
    meta = meta_path.read_text(encoding="utf-8")
    obj = __import__("json").loads(meta)
    drifts = [int(d["start"]) for d in obj.get("drifts", [])]
    T = int(obj.get("n_samples", 0) or 0)
    return drifts, T


def build_cfg_map(seed: int) -> Dict[str, ExperimentConfig]:
    mapping: Dict[str, ExperimentConfig] = {}
    for cfg in _default_experiment_configs(seed):
        mapping[cfg.dataset_name.lower()] = cfg
    return mapping


def read_log_metrics(csv_path: Path) -> Dict[str, object]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {
            "acc_final": None,
            "mean_acc": None,
            "acc_min": None,
            "detections": [],
            "x_col": "sample_idx",
        }
    fieldnames = rows[0].keys()
    x_col = "sample_idx" if "sample_idx" in fieldnames else ("seen_samples" if "seen_samples" in fieldnames else "step")
    accs: List[float] = []
    dets: List[int] = []
    for r in rows:
        acc = r.get("metric_accuracy")
        if acc is not None and acc != "":
            try:
                accs.append(float(acc))
            except ValueError:
                pass
        if r.get("drift_flag") == "1":
            x = r.get(x_col)
            if x is not None and x != "":
                try:
                    dets.append(int(float(x)))
                except ValueError:
                    pass
    return {
        "acc_final": accs[-1] if accs else None,
        "mean_acc": (sum(accs) / len(accs)) if accs else None,
        "acc_min": min(accs) if accs else None,
        "detections": dets,
        "x_col": x_col,
    }


def main() -> int:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = list(args.seeds)
    thetas = [float(x) for x in args.thetas]
    weights = parse_weights(args.weights)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    synth_root = Path(args.synthetic_root)

    # 确保合成流 parquet/meta 存在
    generate_default_abrupt_synth_datasets(seeds=seeds, out_root=str(synth_root))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for theta in thetas:
        experiment_run: ExperimentRun = create_experiment_run(
            experiment_name="trackF_threshold_sweep",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{args.run_name}_theta{theta:.2f}",
            run_id=None,
        )
        cfg_map = build_cfg_map(seed=seeds[0])
        for dataset in datasets:
            if dataset.lower() not in cfg_map:
                raise ValueError(f"未知 dataset: {dataset}（请在 _default_experiment_configs 中配置）")
            for seed in seeds:
                cfg_seed_map = build_cfg_map(seed=seed)
                base_cfg = cfg_seed_map[dataset.lower()]
                run_paths = experiment_run.prepare_dataset_run(dataset, "ts_drift_adapt", seed)
                log_path = run_paths.log_csv_path()
                cfg = replace(
                    base_cfg,
                    model_variant="ts_drift_adapt",
                    seed=seed,
                    log_path=str(log_path),
                    monitor_preset=args.monitor_preset,
                    trigger_mode="weighted",
                    trigger_weights=weights,
                    trigger_threshold=float(theta),
                    trigger_k=2,
                    confirm_window=200,
                )
                _ = run_experiment(cfg, device=args.device)
                run_paths.update_legacy_pointer()

                meta_drifts, horizon = load_synth_meta(synth_root, dataset, seed)
                log_stats = read_log_metrics(log_path)
                dets = log_stats["detections"]  # type: ignore[assignment]
                win = compute_detection_metrics(meta_drifts, dets, int(horizon)) if horizon > 0 else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
                dets_merged = _merge_events(dets, int(args.min_separation))
                tol = compute_metrics_tolerance(meta_drifts, dets_merged, int(horizon), int(args.match_tolerance))

                records.append(
                    {
                        "experiment_name": experiment_run.experiment_name,
                        "run_id": experiment_run.run_id,
                        "dataset": dataset,
                        "seed": seed,
                        "model_variant": "ts_drift_adapt",
                        "monitor_preset": args.monitor_preset,
                        "trigger_mode": "weighted",
                        "theta": float(theta),
                        "weights": args.weights,
                        "acc_final": log_stats["acc_final"],
                        "mean_acc": log_stats["mean_acc"],
                        "acc_min": log_stats["acc_min"],
                        "MDR_win": win.get("MDR"),
                        "MTD_win": win.get("MTD"),
                        "MTFA_win": win.get("MTFA"),
                        "MTR_win": win.get("MTR"),
                        "MDR_tol": tol.get("MDR"),
                        "MTD_tol": tol.get("MTD"),
                        "MTFA_tol": tol.get("MTFA"),
                        "MTR_tol": tol.get("MTR"),
                        "match_tolerance": int(args.match_tolerance),
                        "min_separation": int(args.min_separation),
                        "horizon": int(horizon),
                    }
                )

    # 写 CSV
    fieldnames = list(records[0].keys()) if records else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"[done] wrote {out_csv}")

    # 聚合并画图（按 dataset+theta 取均值）
    by_key: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for r in records:
        key = (str(r["dataset"]), float(r["theta"]))
        by_key.setdefault(key, []).append(r)

    def agg(metric: str, dataset: str) -> Tuple[List[float], List[float]]:
        xs = sorted({theta for (ds, theta) in by_key.keys() if ds == dataset})
        ys_mu: List[float] = []
        ys_sd: List[float] = []
        for x in xs:
            rows = by_key.get((dataset, x), [])
            vals = [float(rr[metric]) for rr in rows if rr.get(metric) is not None and not math.isnan(float(rr[metric]))]
            mu = _mean(vals) if vals else math.nan
            sd = _std(vals) if vals else math.nan
            ys_mu.append(float(mu) if mu is not None else math.nan)
            ys_sd.append(float(sd) if sd is not None else math.nan)
        return xs, ys_mu

    # 图1：theta-MDR 与 theta-MTFA（两子图）
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(2, 1, 1)
    for ds in datasets:
        xs, ys = agg("MDR_win", ds)
        ax1.plot(xs, ys, marker="o", label=ds)
    ax1.set_title("TrackF: theta vs MDR (window-based)")
    ax1.set_xlabel("theta")
    ax1.set_ylabel("MDR")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig1.add_subplot(2, 1, 2)
    for ds in datasets:
        xs, ys = agg("MTFA_win", ds)
        ax2.plot(xs, ys, marker="o", label=ds)
    ax2.set_title("TrackF: theta vs MTFA (window-based)")
    ax2.set_xlabel("theta")
    ax2.set_ylabel("MTFA")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig1.tight_layout()
    fig1_path = fig_dir / "trackF_theta_mdr_mtfa.png"
    fig1.savefig(fig1_path, dpi=160)
    plt.close(fig1)
    print(f"[done] wrote {fig1_path}")

    # 图2：theta-acc_final
    fig2 = plt.figure(figsize=(10, 4))
    ax = fig2.add_subplot(1, 1, 1)
    for ds in datasets:
        xs, ys = agg("acc_final", ds)
        ax.plot(xs, ys, marker="o", label=ds)
    ax.set_title("TrackF: theta vs acc_final")
    ax.set_xlabel("theta")
    ax.set_ylabel("acc_final")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2_path = fig_dir / "trackF_theta_acc_final.png"
    fig2.savefig(fig2_path, dpi=160)
    plt.close(fig2)
    print(f"[done] wrote {fig2_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

