from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def read_run_index(glob_pattern: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(Path().glob(glob_pattern)):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def parse_eps_list(spec: str) -> List[float]:
    values: List[float] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values or [0.0]


def drift_positions(dataset_name: str) -> List[int]:
    name = str(dataset_name).lower()
    if name == "sea_abrupt4":
        return [10000, 20000, 30000, 40000]
    if name == "sine_abrupt4":
        return [10000, 20000, 30000, 40000]
    if name == "stagger_abrupt3":
        return [20000, 40000]
    return []


def infer_step_size(sample_idxs: Sequence[int]) -> int:
    if not sample_idxs:
        return 1
    xs = sorted(set(int(x) for x in sample_idxs))
    if len(xs) < 2:
        return 1
    diffs = [b - a for a, b in zip(xs[:-1], xs[1:]) if b - a > 0]
    if not diffs:
        return 1
    return max(1, int(np.median(np.asarray(diffs, dtype=np.float64))))


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def fmt_mu_std(mu: Optional[float], sd: Optional[float], digits: int = 4) -> str:
    if mu is None:
        return "N/A"
    if sd is None or math.isnan(float(sd)):
        return f"{mu:.{digits}f}±NaN"
    return f"{mu:.{digits}f}±{sd:.{digits}f}"


def rolling_jitter(values: Sequence[float], window: int) -> Optional[float]:
    if not values:
        return None
    if window <= 1:
        return float(np.std(values))
    series = pd.Series(values)
    roll = series.rolling(window=window).std().dropna()
    return float(roll.mean()) if not roll.empty else None


def smooth_series(
    sample_idxs: Sequence[int],
    accs: Sequence[float],
    *,
    smooth_window_samples: Optional[int] = None,
    smooth_window_points: Optional[int] = None,
) -> List[Tuple[int, float]]:
    if not sample_idxs:
        return []
    roll_rows = 1
    if smooth_window_points is not None and int(smooth_window_points) > 0:
        roll_rows = max(1, int(smooth_window_points))
    elif smooth_window_samples is not None and int(smooth_window_samples) > 0:
        step = infer_step_size(sample_idxs)
        roll_rows = max(1, int(round(float(smooth_window_samples) / float(step))))
    series = pd.Series(accs).rolling(window=roll_rows, center=True, min_periods=1).mean()
    return [(int(x), float(a)) for x, a in zip(sample_idxs, series.to_list())]


def _window_values(
    series: Sequence[Tuple[int, float]],
    start: int,
    end: int,
) -> List[float]:
    return [a for x, a in series if int(start) <= int(x) < int(end)]


def compute_recovery_time(
    series: Sequence[Tuple[int, float]],
    *,
    start: int,
    end: int,
    threshold: float,
    k: int,
    fallback: float,
) -> float:
    if not series:
        return float(fallback)
    count = 0
    start_idx: Optional[int] = None
    for x, a in series:
        if x < start or x >= end:
            continue
        if a >= threshold:
            if count == 0:
                start_idx = int(x)
            count += 1
            if count >= k and start_idx is not None:
                return float(start_idx - start)
        else:
            count = 0
            start_idx = None
    return float(fallback)


def compute_drift_metrics(
    series: Sequence[Tuple[int, float]],
    drifts: Sequence[int],
    *,
    pre_window: int,
    post_window: int,
    eps_list: Sequence[float],
    recovery_k: int,
) -> Dict[str, Any]:
    if not series or not drifts:
        return {
            "pre_acc": None,
            "worst_dip": None,
            "recovery_auc": None,
            "recovery_times": {eps: None for eps in eps_list},
        }
    pre_accs: List[float] = []
    dips: List[float] = []
    aucs: List[float] = []
    rec_times: Dict[float, List[float]] = {eps: [] for eps in eps_list}
    for g in drifts:
        pre_vals = _window_values(series, int(g) - int(pre_window), int(g))
        if not pre_vals:
            continue
        pre_acc = float(np.mean(pre_vals))
        pre_accs.append(pre_acc)
        post_vals = _window_values(series, int(g), int(g) + int(post_window))
        if post_vals:
            post_min = float(min(post_vals))
            dips.append(max(0.0, pre_acc - post_min))
            aucs.append(float(np.mean(post_vals)))
        for eps in eps_list:
            threshold = pre_acc - float(eps)
            rec_time = compute_recovery_time(
                series,
                start=int(g),
                end=int(g) + int(post_window),
                threshold=threshold,
                k=int(recovery_k),
                fallback=float(post_window),
            )
            rec_times[eps].append(rec_time)
    pre_mu, _ = mean_std(pre_accs)
    dip_mu, _ = mean_std(dips)
    auc_mu, _ = mean_std(aucs)
    rec_mu = {eps: mean_std(times)[0] for eps, times in rec_times.items()}
    return {
        "pre_acc": pre_mu,
        "worst_dip": dip_mu,
        "recovery_auc": auc_mu,
        "recovery_times": rec_mu,
    }


def load_log_metrics(
    log_path: Path,
    *,
    pre_window: int,
    post_window: int,
    smooth_window_samples: Optional[int],
    smooth_window_points: Optional[int],
    eps_list: Sequence[float],
    recovery_k: int,
    jitter_window: int,
    dataset_name: str,
    use_all_drifts: bool,
) -> Dict[str, Any]:
    df = pd.read_csv(log_path)
    if df.empty:
        return {}
    df = df.dropna(subset=["metric_accuracy"])
    accs = df["metric_accuracy"].astype(float).to_list()
    samples = df["sample_idx"].astype(int).to_list()
    series = smooth_series(
        samples,
        accs,
        smooth_window_samples=smooth_window_samples,
        smooth_window_points=smooth_window_points,
    )
    drifts = drift_positions(dataset_name)
    if drifts and not use_all_drifts:
        drifts = drifts[:1]
    overall_acc = float(np.mean(accs)) if accs else None
    if drifts:
        metrics = compute_drift_metrics(
            series,
            drifts,
            pre_window=int(pre_window),
            post_window=int(post_window),
            eps_list=eps_list,
            recovery_k=int(recovery_k),
        )
        pre_acc = metrics["pre_acc"]
        worst_dip = metrics["worst_dip"]
        recovery_auc = metrics["recovery_auc"]
        recovery_times = metrics["recovery_times"]
    else:
        pre_acc = overall_acc
        worst_dip = None
        recovery_auc = None
        recovery_times = {eps: None for eps in eps_list}
    jitter = None
    if not drifts:
        jitter = rolling_jitter(accs, window=jitter_window)

    gamma_vals = df["ema_decay"].astype(float).to_numpy() if "ema_decay" in df else np.array([])
    sev_vals = df["ema_severity_ema"].astype(float).to_numpy() if "ema_severity_ema" in df else np.array([])
    last = df.iloc[-1]
    return {
        "overall_acc": overall_acc,
        "pre_acc": pre_acc,
        "worst_dip": worst_dip,
        "recovery_auc": recovery_auc,
        "recovery_times": recovery_times,
        "jitter": jitter,
        "gamma_vals": gamma_vals,
        "sev_vals": sev_vals,
        "candidate_count_total": int(last.get("candidate_count_total", 0)),
        "confirmed_count_total": int(last.get("confirmed_count_total", 0)),
    }


def collect_config_cols(rows: Sequence[Dict[str, str]]) -> List[str]:
    exclude = {
        "dataset_name",
        "dataset_type",
        "dataset_kind",
        "labeled_ratio",
        "variant",
        "model_variant",
        "seed",
        "log_path",
    }
    cols = set()
    for r in rows:
        cols.update(r.keys())
    return sorted([c for c in cols if c not in exclude])


def unique_or_mixed(values: Sequence[Any]) -> str:
    cleaned: List[str] = []
    for v in values:
        if v is None:
            continue
        try:
            if isinstance(v, float) and math.isnan(v):
                continue
        except Exception:
            pass
        s = str(v)
        if s == "" or s.lower() == "nan":
            continue
        cleaned.append(s)
    if not cleaned:
        return ""
    uniq = sorted(set(cleaned))
    if len(uniq) == 1:
        return uniq[0]
    return "|".join(uniq)


def debug_run_metrics(
    log_path: Path,
    *,
    dataset_name: str,
    pre_window: int,
    post_window: int,
    smooth_window_samples: Optional[int],
    smooth_window_points: Optional[int],
) -> None:
    df = pd.read_csv(log_path)
    if df.empty:
        print("[debug] empty log")
        return
    df = df.dropna(subset=["metric_accuracy"])
    accs = df["metric_accuracy"].astype(float).to_list()
    samples = df["sample_idx"].astype(int).to_list()
    series = smooth_series(
        samples,
        accs,
        smooth_window_samples=smooth_window_samples,
        smooth_window_points=smooth_window_points,
    )
    drifts = drift_positions(dataset_name)
    if not drifts:
        print("[debug] no drift anchors for dataset:", dataset_name)
        return
    drift0 = int(drifts[0])
    pre_vals = _window_values(series, drift0 - int(pre_window), drift0)
    post_vals = _window_values(series, drift0, drift0 + int(post_window))
    pre_acc = float(np.mean(pre_vals)) if pre_vals else None
    post_min = float(min(post_vals)) if post_vals else None
    worst_dip = None
    if pre_acc is not None and post_min is not None:
        worst_dip = max(0.0, pre_acc - post_min)
    print("[debug] drift0_sample_idx:", drift0)
    print("[debug] pre_acc:", pre_acc)
    print("[debug] post_min:", post_min)
    print("[debug] worst_dip:", worst_dip)
    print("[debug] post_curve_head:", post_vals[:10])
    if post_vals:
        print("[debug] post_curve_min_max:", min(post_vals), max(post_vals))


def aggregate_curve(
    runs: Sequence[Dict[str, Any]],
    *,
    drift0: Optional[int],
    horizon: int,
    smooth_window_samples: Optional[int],
    smooth_window_points: Optional[int],
) -> Tuple[List[int], List[float]]:
    bucket: Dict[int, List[float]] = defaultdict(list)
    for r in runs:
        df = pd.read_csv(Path(r["log_path"]))
        if df.empty:
            continue
        df = df.dropna(subset=["metric_accuracy", "sample_idx"])
        xs = df["sample_idx"].astype(int).to_list()
        ys = df["metric_accuracy"].astype(float).to_list()
        series = smooth_series(
            xs,
            ys,
            smooth_window_samples=smooth_window_samples,
            smooth_window_points=smooth_window_points,
        )
        for x, a in series:
            if drift0 is None:
                offset = x
            else:
                if x < drift0 or x > drift0 + horizon:
                    continue
                offset = x - drift0
            bucket[offset].append(float(a))
    xs = sorted(bucket.keys())
    ys = [float(np.mean(bucket[x])) for x in xs]
    return xs, ys


def smooth_curve(xs: Sequence[int], ys: Sequence[float], window: int) -> Tuple[List[int], List[float]]:
    if window <= 1:
        return list(xs), list(ys)
    series = pd.Series(ys).rolling(window=window, center=True, min_periods=1).mean()
    return list(xs), series.to_list()


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize ExpA EMA decay results")
    p.add_argument("--run_index_glob", type=str, default="results/expA_ema_decay/run_index_*.csv")
    p.add_argument("--out_detail_csv", type=str, default="results/expA_ema_decay/summary_detail_v2.csv")
    p.add_argument("--out_table_csv", type=str, default="results/expA_ema_decay/summary_table_v2.csv")
    p.add_argument("--plot_dir", type=str, default="results/expA_ema_decay/plots_v2")
    p.add_argument("--pre_window", type=int, default=2000)
    p.add_argument("--post_window", type=int, default=4000)
    p.add_argument("--smooth_window_samples", type=int, default=0)
    p.add_argument("--smooth_window_points", type=int, default=5)
    p.add_argument("--recovery_k", type=int, default=5)
    p.add_argument("--eps_list", type=str, default="0,0.005")
    p.add_argument("--jitter_window", type=int, default=20)
    p.add_argument("--plot_dataset", type=str, default="sea_abrupt4")
    p.add_argument("--plot_nodrift_dataset", type=str, default="sea_nodrift")
    p.add_argument("--plot_ratio", type=float, default=0.01)
    p.add_argument("--recovery_horizon", type=int, default=4000)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--compare_old", action="store_true")
    p.add_argument("--use_all_drifts", action="store_true")
    p.add_argument("--debug_one", action="store_true")
    p.add_argument("--debug_log_path", type=str, default="")
    args = p.parse_args()

    rows = read_run_index(args.run_index_glob)
    if not rows:
        raise SystemExit("未找到 run_index")

    eps_list = parse_eps_list(args.eps_list)
    config_cols = collect_config_cols(rows)
    detail_rows: List[Dict[str, Any]] = []
    group_gamma: Dict[Tuple[str, float, str], List[float]] = defaultdict(list)
    group_sev: Dict[Tuple[str, float, str], List[float]] = defaultdict(list)

    for r in rows:
        log_path = Path(str(r.get("log_path") or ""))
        if not log_path.exists():
            continue
        dataset = str(r.get("dataset_name") or "")
        ratio = float(r.get("labeled_ratio") or 0.0)
        variant = str(r.get("variant") or "")
        metrics = load_log_metrics(
            log_path,
            pre_window=int(args.pre_window),
            post_window=int(args.post_window),
            smooth_window_samples=int(args.smooth_window_samples) if int(args.smooth_window_samples) > 0 else None,
            smooth_window_points=int(args.smooth_window_points) if int(args.smooth_window_points) > 0 else None,
            eps_list=eps_list,
            recovery_k=int(args.recovery_k),
            jitter_window=int(args.jitter_window),
            dataset_name=dataset,
            use_all_drifts=bool(args.use_all_drifts),
        )
        if not metrics:
            continue
        gamma_vals = metrics.pop("gamma_vals", np.array([]))
        sev_vals = metrics.pop("sev_vals", np.array([]))
        recovery_times = metrics.pop("recovery_times", {})
        for eps in eps_list:
            metrics[f"recovery_time_eps{str(eps).replace('.', 'p')}"] = recovery_times.get(eps)
        key = (dataset, ratio, variant)
        if gamma_vals.size > 0:
            group_gamma[key].extend([float(x) for x in gamma_vals])
        if sev_vals.size > 0:
            group_sev[key].extend([float(x) for x in sev_vals])
        base_row = dict(r)
        base_row["dataset_name"] = dataset
        base_row["labeled_ratio"] = ratio
        base_row["variant"] = variant
        base_row["seed"] = int(float(r.get("seed") or 0))
        base_row["log_path"] = str(log_path)
        detail_rows.append({**base_row, **metrics})

    out_detail = Path(args.out_detail_csv)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        fieldnames = sorted({k for row in detail_rows for k in row.keys()})
        with out_detail.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in detail_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})

    by_group: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in detail_rows:
        key = (str(r["dataset_name"]), float(r["labeled_ratio"]), str(r["variant"]))
        by_group[key].append(r)

    table_rows: List[Dict[str, Any]] = []
    for (dataset, ratio, variant), rs in sorted(by_group.items()):
        overall = [r.get("overall_acc") for r in rs]
        pre = [r.get("pre_acc") for r in rs]
        dip = [r.get("worst_dip") for r in rs]
        rec_auc = [r.get("recovery_auc") for r in rs]
        rec_by_eps: Dict[float, List[Optional[float]]] = {eps: [] for eps in eps_list}
        for r in rs:
            for eps in eps_list:
                key = f"recovery_time_eps{str(eps).replace('.', 'p')}"
                rec_by_eps[eps].append(r.get(key))
        jitter = [r.get("jitter") for r in rs]
        overall_mu, overall_sd = mean_std(overall)
        pre_mu, pre_sd = mean_std(pre)
        dip_mu, dip_sd = mean_std(dip)
        auc_mu, auc_sd = mean_std(rec_auc)
        jitter_mu, jitter_sd = mean_std(jitter)
        gamma_vals = group_gamma.get((dataset, ratio, variant), [])
        gamma_mean = float(np.mean(gamma_vals)) if gamma_vals else None
        gamma_p10 = float(np.quantile(gamma_vals, 0.10)) if gamma_vals else None
        gamma_p50 = float(np.quantile(gamma_vals, 0.50)) if gamma_vals else None
        gamma_p90 = float(np.quantile(gamma_vals, 0.90)) if gamma_vals else None
        row: Dict[str, Any] = {
            "dataset": dataset,
            "labeled_ratio": ratio,
            "variant": variant,
            "n_runs": len(rs),
            "overall_acc": fmt_mu_std(overall_mu, overall_sd),
            "pre_acc": fmt_mu_std(pre_mu, pre_sd),
            "worst_dip": fmt_mu_std(dip_mu, dip_sd) if drift_positions(dataset) else "N/A",
            "recovery_auc": fmt_mu_std(auc_mu, auc_sd) if drift_positions(dataset) else "N/A",
            "jitter": fmt_mu_std(jitter_mu, jitter_sd) if not drift_positions(dataset) else "N/A",
            "gamma_mean": gamma_mean if gamma_mean is not None else "",
            "gamma_p10": gamma_p10 if gamma_p10 is not None else "",
            "gamma_p50": gamma_p50 if gamma_p50 is not None else "",
            "gamma_p90": gamma_p90 if gamma_p90 is not None else "",
        }
        for col in config_cols:
            row[col] = unique_or_mixed([r.get(col) for r in rs])
        for eps in eps_list:
            mu, sd = mean_std(rec_by_eps.get(eps, []))
            key = f"recovery_time_eps{str(eps).replace('.', 'p')}"
            row[key] = fmt_mu_std(mu, sd) if drift_positions(dataset) else "N/A"
        table_rows.append(row)

    out_table = Path(args.out_table_csv)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    if table_rows:
        fieldnames = sorted({k for row in table_rows for k in row.keys()})
        with out_table.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in table_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_ratio = float(args.plot_ratio)
    plot_dataset = str(args.plot_dataset)
    plot_nodrift = str(args.plot_nodrift_dataset)
    pick_variants = ["B1", "B3", "P1"]
    eps_label = ",".join(str(eps) for eps in eps_list)

    import matplotlib.pyplot as plt

    # abrupt recovery curve
    abrupt_runs = [
        r
        for r in rows
        if str(r.get("dataset_name")) == plot_dataset
        and float(r.get("labeled_ratio") or 0.0) == plot_ratio
        and str(r.get("variant")) in pick_variants
    ]
    drift0 = drift_positions(plot_dataset)[0] if drift_positions(plot_dataset) else None
    if abrupt_runs and drift0 is not None:
        plt.figure(figsize=(7.2, 4.2))
        for variant in pick_variants:
            vruns = [r for r in abrupt_runs if str(r.get("variant")) == variant]
            xs, ys = aggregate_curve(
                vruns,
                drift0=drift0,
                horizon=int(args.recovery_horizon),
                smooth_window_samples=int(args.smooth_window_samples) if int(args.smooth_window_samples) > 0 else None,
                smooth_window_points=int(args.smooth_window_points) if int(args.smooth_window_points) > 0 else None,
            )
            xs, ys = smooth_curve(xs, ys, window=int(args.smooth_window))
            if xs:
                plt.plot(xs, ys, label=variant)
        plt.xlabel("samples after drift0")
        plt.ylabel("accuracy")
        plt.title(
            f"{plot_dataset} recovery post_drift0 (lr={plot_ratio}, roll_pts={args.smooth_window_points}, "
            f"plot_roll={args.smooth_window}, Wpre={args.pre_window}, Hpost={args.post_window}, "
            f"eps={eps_label}, K={args.recovery_k})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            plot_dir
            / f"{plot_dataset}_recovery_post_drift0_lr{str(plot_ratio).replace('.', 'p')}.png",
            dpi=160,
        )
        plt.close()

    # nodrift curve
    nodrift_runs = [
        r
        for r in rows
        if str(r.get("dataset_name")) == plot_nodrift
        and float(r.get("labeled_ratio") or 0.0) == plot_ratio
        and str(r.get("variant")) in pick_variants
    ]
    if nodrift_runs:
        plt.figure(figsize=(7.2, 4.2))
        for variant in pick_variants:
            vruns = [r for r in nodrift_runs if str(r.get("variant")) == variant]
            xs, ys = aggregate_curve(
                vruns,
                drift0=None,
                horizon=int(args.recovery_horizon),
                smooth_window_samples=int(args.smooth_window_samples) if int(args.smooth_window_samples) > 0 else None,
                smooth_window_points=int(args.smooth_window_points) if int(args.smooth_window_points) > 0 else None,
            )
            xs, ys = smooth_curve(xs, ys, window=int(args.smooth_window))
            if xs:
                plt.plot(xs, ys, label=variant)
        plt.xlabel("sample_idx")
        plt.ylabel("accuracy")
        plt.title(
            f"{plot_nodrift} stability (lr={plot_ratio}, roll_pts={args.smooth_window_points}, "
            f"plot_roll={args.smooth_window})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            plot_dir
            / f"{plot_nodrift}_nodrift_curve_lr{str(plot_ratio).replace('.', 'p')}.png",
            dpi=160,
        )
        plt.close()

    if args.debug_one or args.debug_log_path:
        if args.debug_log_path:
            debug_path = Path(args.debug_log_path)
            match = next((r for r in rows if str(r.get("log_path") or "") == str(debug_path)), None)
            dataset = str(match.get("dataset_name") or "") if match else ""
        else:
            debug_path = Path(str(rows[0].get("log_path") or ""))
            dataset = str(rows[0].get("dataset_name") or "")
        if debug_path.exists():
            debug_run_metrics(
                debug_path,
                dataset_name=dataset,
                pre_window=int(args.pre_window),
                post_window=int(args.post_window),
                smooth_window_samples=int(args.smooth_window_samples) if int(args.smooth_window_samples) > 0 else None,
                smooth_window_points=int(args.smooth_window_points) if int(args.smooth_window_points) > 0 else None,
            )
        else:
            print("[debug] log_path missing:", debug_path)

    if args.compare_old:
        old_path = Path(args.out_table_csv).with_name("summary_table.csv")
        new_path = Path(args.out_table_csv)
        if old_path.exists() and new_path.exists():
            old_df = pd.read_csv(old_path)
            new_df = pd.read_csv(new_path)
            old_vals = (
                old_df[old_df["dataset"] == "sea_abrupt4"]["recovery_time"].dropna().unique().tolist()
                if "recovery_time" in old_df
                else []
            )
            key_eps = f"recovery_time_eps{str(eps_list[0]).replace('.', 'p')}"
            new_vals = (
                new_df[new_df["dataset"] == "sea_abrupt4"][key_eps].dropna().unique().tolist()
                if key_eps in new_df
                else []
            )
            compare_path = Path(args.out_table_csv).with_name("recovery_time_compare.txt")
            compare_path.write_text(
                "old_recovery_time=" + ",".join(str(v) for v in old_vals) + "\n"
                + "new_recovery_time=" + ",".join(str(v) for v in new_vals) + "\n",
                encoding="utf-8",
            )
            print(f"[compare] wrote {compare_path}")
        else:
            print("[compare] skip, missing old/new table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
