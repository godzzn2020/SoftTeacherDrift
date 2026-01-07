#!/usr/bin/env python
"""Track H：INSECTS 上验证 severity v2 gating（confirmed_only）是否缓解负迁移（最小验证）。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.drift_metrics import compute_detection_metrics
from experiments.first_stage_experiments import ExperimentConfig, _default_experiment_configs, run_experiment
from soft_drift.utils.run_paths import ExperimentRun, create_experiment_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track H：severity v2 gating on INSECTS (seeds=1/3/5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--monitor_preset", type=str, default="error_divergence_ph_meta")
    parser.add_argument("--trigger_mode", type=str, default="or", choices=["or", "weighted", "two_stage"])
    parser.add_argument("--weights", type=str, default="error_rate=0.5,divergence=0.3,teacher_entropy=0.2")
    parser.add_argument("--theta_confirm", type=float, default=0.6, help="用于 confirmed_only gating 的票分阈值（复用 trigger_threshold）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="trackH")
    parser.add_argument("--out_csv", type=str, default="scripts/TRACKH_SEVERITY_GATING.csv")
    parser.add_argument("--match_tolerance", type=int, default=500)
    parser.add_argument("--min_separation", type=int, default=200)
    parser.add_argument("--insects_meta", type=str, default="datasets/real/INSECTS_abrupt_balanced.json")
    return parser.parse_args()


def parse_weights(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--weights 格式错误：{token}")
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
    return {"MDR": float(mdr) if mdr is not None else None, "MTD": float(mtd) if mtd is not None else None, "MTFA": float(mtfa) if mtfa is not None else None, "MTR": float(mtr) if mtr is not None else None}


def load_insects_meta(path: Path) -> List[int]:
    obj = __import__("json").loads(path.read_text(encoding="utf-8"))
    return [int(p) for p in obj.get("positions", [])]


def build_insects_cfg(seed: int) -> ExperimentConfig:
    cfg_map = {c.dataset_name.lower(): c for c in _default_experiment_configs(seed)}
    return cfg_map["insects_abrupt_balanced"]


def read_log_metrics(csv_path: Path) -> Dict[str, object]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {"acc_final": None, "mean_acc": None, "acc_min": None, "detections": [], "horizon": 0}
    fieldnames = rows[0].keys()
    x_col = "sample_idx" if "sample_idx" in fieldnames else ("seen_samples" if "seen_samples" in fieldnames else "step")
    accs: List[float] = []
    dets: List[int] = []
    last_x = 0
    for r in rows:
        a = r.get("metric_accuracy")
        if a:
            try:
                accs.append(float(a))
            except ValueError:
                pass
        x = r.get(x_col)
        if x:
            try:
                last_x = int(float(x))
            except ValueError:
                pass
        if r.get("drift_flag") == "1":
            if x:
                try:
                    dets.append(int(float(x)))
                except ValueError:
                    pass
    return {
        "acc_final": accs[-1] if accs else None,
        "mean_acc": (sum(accs) / len(accs)) if accs else None,
        "acc_min": min(accs) if accs else None,
        "detections": dets,
        "horizon": last_x + 1,
    }


def main() -> int:
    args = parse_args()
    seeds = list(args.seeds)
    weights = parse_weights(args.weights)
    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    meta_positions = load_insects_meta(Path(args.insects_meta))

    groups = [
        ("baseline", dict(model_variant="ts_drift_adapt", use_v2=False, gate="none")),
        ("v1", dict(model_variant="ts_drift_adapt_severity", use_v2=False, gate="none")),
        ("v2", dict(model_variant="ts_drift_adapt_severity", use_v2=True, gate="none")),
        ("v2_gate", dict(model_variant="ts_drift_adapt_severity", use_v2=True, gate="confirmed_only")),
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    for tag, gcfg in groups:
        exp_run: ExperimentRun = create_experiment_run(
            experiment_name="trackH_severity_gating",
            results_root=results_root,
            logs_root=logs_root,
            run_name=f"{args.run_name}_{tag}",
        )
        for seed in seeds:
            base_cfg = build_insects_cfg(seed)
            run_paths = exp_run.prepare_dataset_run("INSECTS_abrupt_balanced", str(gcfg["model_variant"]), seed)
            log_path = run_paths.log_csv_path()
            cfg = replace(
                base_cfg,
                model_variant=str(gcfg["model_variant"]),
                seed=seed,
                log_path=str(log_path),
                monitor_preset=args.monitor_preset,
                trigger_mode=str(args.trigger_mode),
                trigger_weights=weights,
                trigger_threshold=float(args.theta_confirm),
                confirm_window=200,
                use_severity_v2=bool(gcfg["use_v2"]),
                severity_gate=str(gcfg["gate"]),
                entropy_mode="uncertain",
                severity_decay=0.95,
                freeze_baseline_steps=5,
                severity_scheduler_scale=2.0,
            )
            _ = run_experiment(cfg, device=args.device)
            run_paths.update_legacy_pointer()

            stats = read_log_metrics(log_path)
            dets = stats["detections"]  # type: ignore[assignment]
            horizon = int(stats["horizon"])
            win = compute_detection_metrics(meta_positions, dets, horizon) if compute_detection_metrics is not None else {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
            dets_merged = _merge_events(dets, int(args.min_separation))
            tol = compute_metrics_tolerance(meta_positions, dets_merged, horizon, int(args.match_tolerance))

            records.append(
                {
                    "experiment_name": exp_run.experiment_name,
                    "run_id": exp_run.run_id,
                    "group": tag,
                    "dataset": "INSECTS_abrupt_balanced",
                    "seed": seed,
                    "model_variant": str(gcfg["model_variant"]),
                    "monitor_preset": args.monitor_preset,
                    "trigger_mode": str(args.trigger_mode),
                    "theta_confirm": float(args.theta_confirm),
                    "weights": args.weights,
                    "use_severity_v2": int(bool(gcfg["use_v2"])),
                    "severity_gate": str(gcfg["gate"]),
                    "acc_final": stats["acc_final"],
                    "mean_acc": stats["mean_acc"],
                    "acc_min": stats["acc_min"],
                    "MDR_win": win.get("MDR"),
                    "MTD_win": win.get("MTD"),
                    "MTFA_win": win.get("MTFA"),
                    "MTR_win": win.get("MTR"),
                    "MDR_tol": tol.get("MDR"),
                    "MTD_tol": tol.get("MTD"),
                    "MTFA_tol": tol.get("MTFA"),
                    "MTR_tol": tol.get("MTR"),
                }
            )

    fieldnames = list(records[0].keys()) if records else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"[done] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

