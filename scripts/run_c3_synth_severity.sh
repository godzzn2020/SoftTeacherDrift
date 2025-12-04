#!/usr/bin/env bash
# Multi-GPU launcher for Phase C3 severity scheduling experiments on synthetic datasets.
# This script only issues python commands; edit GPU list / seeds as needed before running.

set -euo pipefail

DATASETS="sea_abrupt4,sine_abrupt4,stagger_abrupt3"
SEEDS_BASELINE="1 2 3 4 5"
SEEDS_SEVERITY="1 2 3 4 5"
GPUS="0,1"
MAX_JOBS=2
MONITOR="error_divergence_ph_meta"
DEVICE="cuda"
SEVERITY_SCALES=("1.0")

echo "[info] Running baseline ts_drift_adapt with seeds: ${SEEDS_BASELINE}"
python experiments/stage1_multi_seed.py \
  --datasets "${DATASETS}" \
  --models ts_drift_adapt \
  --seeds ${SEEDS_BASELINE} \
  --monitor_preset "${MONITOR}" \
  --device "${DEVICE}" \
  --gpus "${GPUS}" \
  --max_jobs_per_gpu ${MAX_JOBS}

for SCALE in "${SEVERITY_SCALES[@]}"; do
  SCALE_TAG="${SCALE//./p}"
  VARIANT="ts_drift_adapt_severity_s${SCALE_TAG}"
  echo "[info] Running severity-aware ${VARIANT} with seeds: ${SEEDS_SEVERITY} (scale=${SCALE})"
  python experiments/stage1_multi_seed.py \
    --datasets "${DATASETS}" \
    --models ${VARIANT} \
    --seeds ${SEEDS_SEVERITY} \
    --monitor_preset "${MONITOR}" \
    --device "${DEVICE}" \
    --gpus "${GPUS}" \
    --max_jobs_per_gpu ${MAX_JOBS} \
    --severity_scheduler_scale ${SCALE}
done

echo "[done] Phase C3 synthetic severity runs submitted."
