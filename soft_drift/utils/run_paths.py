"""Utilities for consistent run_id generation and log/result paths."""

from __future__ import annotations

import datetime as _dt
import random
import re
import shutil
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _sanitize_token(value: str) -> str:
    value = value.strip()
    if not value:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def generate_run_id(run_name: Optional[str] = None) -> str:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=3))
    run_id = f"{timestamp}-{suffix}"
    if run_name:
        run_id = f"{run_id}_{_sanitize_token(run_name)}"
    return run_id


@dataclass
class DatasetRunPaths:
    dataset_name: str
    model_variant: str
    seed: int
    run_id: str
    results_dir: Path
    logs_dir: Path
    legacy_path: Path

    def log_csv_path(self) -> Path:
        filename = f"{self.dataset_name}__{self.model_variant}__seed{self.seed}.csv"
        return self.logs_dir / filename

    def update_legacy_pointer(self) -> None:
        target = self.log_csv_path()
        if not target.exists():
            return
        self.legacy_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.legacy_path.is_symlink() or self.legacy_path.exists():
                self.legacy_path.unlink()
            self.legacy_path.symlink_to(target)
        except OSError:
            shutil.copy2(target, self.legacy_path)


@dataclass
class ExperimentRun:
    experiment_name: str
    run_id: str
    results_root: Path
    logs_root: Path

    def prepare_dataset_run(self, dataset: str, model_variant: str, seed: int) -> DatasetRunPaths:
        dataset_token = _sanitize_token(dataset)
        model_token = _sanitize_token(model_variant)
        seed_token = _sanitize_token(f"seed{seed}")
        result_dir = (
            self.results_root
            / self.experiment_name
            / dataset_token
            / model_token
            / seed_token
            / self.run_id
        )
        log_dir = (
            self.logs_root
            / self.experiment_name
            / dataset_token
            / model_token
            / seed_token
            / self.run_id
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        legacy_dir = self.logs_root / dataset
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = legacy_dir / f"{dataset}__{model_variant}__seed{seed}.csv"
        return DatasetRunPaths(
            dataset_name=dataset,
            model_variant=model_variant,
            seed=seed,
            run_id=self.run_id,
            results_dir=result_dir,
            logs_dir=log_dir,
            legacy_path=legacy_path,
        )

    def summary_dir(self) -> Path:
        path = self.results_root / self.experiment_name / "summary" / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def describe(self) -> str:
        return f"{self.experiment_name}:{self.run_id}"


def create_experiment_run(
    experiment_name: str,
    results_root: str | Path = "results",
    logs_root: str | Path = "logs",
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ExperimentRun:
    rid = run_id or generate_run_id(run_name)
    return ExperimentRun(
        experiment_name=_sanitize_token(experiment_name),
        run_id=rid,
        results_root=Path(results_root),
        logs_root=Path(logs_root),
    )


def resolve_log_path(
    logs_root: str | Path,
    experiment_name: str,
    dataset_name: str,
    model_variant: str,
    seed: int,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    base = (
        Path(logs_root)
        / _sanitize_token(experiment_name)
        / _sanitize_token(dataset_name)
        / _sanitize_token(model_variant)
        / _sanitize_token(f"seed{seed}")
    )
    filename = f"{dataset_name}__{model_variant}__seed{seed}.csv"
    if run_id:
        candidate = base / run_id / filename
        if candidate.exists():
            return candidate
    if base.exists():
        subdirs = sorted([p for p in base.iterdir() if p.is_dir()])
        for run_dir in reversed(subdirs):
            candidate = run_dir / filename
            if candidate.exists():
                return candidate
    legacy = Path(logs_root) / dataset_name / filename
    if legacy.exists():
        return legacy
    return None
