#!/usr/bin/env bash
# Phase C3 real-stream severity sweep helper. Edit datasets/seeds/scales before running.
# 建议预先执行：conda activate softteacher

set -euo pipefail

REAL_DATASETS="Electricity,NOAA,INSECTS_abrupt_balanced,Airlines"
SEEDS="1 2 3"
LOGS_ROOT="logs"
MONITOR="error_divergence_ph_meta"
DEVICE="cuda"
SEVERITY_SCALES=("0.5" "1.0" "2.0")
TRAIN_RUN_ID="<fill-training-run-id>"

MODEL_VARIANTS=("ts_drift_adapt")

echo "[info] Running baseline ts_drift_adapt on ${REAL_DATASETS} (seeds: ${SEEDS})"
python experiments/run_real_adaptive.py \
  --datasets "${REAL_DATASETS}" \
  --seeds ${SEEDS} \
  --model_variants ts_drift_adapt \
  --monitor_preset "${MONITOR}" \
  --device "${DEVICE}" \
  --logs_root "${LOGS_ROOT}"

for SCALE in "${SEVERITY_SCALES[@]}"; do
  SCALE_TAG="${SCALE//./p}"
  VARIANT="ts_drift_adapt_severity_s${SCALE_TAG}"
  MODEL_VARIANTS+=("${VARIANT}")
  echo "[info] Running ${VARIANT} with severity scale=${SCALE}"
  python experiments/run_real_adaptive.py \
    --datasets "${REAL_DATASETS}" \
    --seeds ${SEEDS} \
    --model_variants ${VARIANT} \
    --monitor_preset "${MONITOR}" \
    --device "${DEVICE}" \
    --logs_root "${LOGS_ROOT}" \
    --severity_scheduler_scale ${SCALE}
  echo
  sleep 1
done

VARIANT_CSV=$(IFS=','; printf "%s" "${MODEL_VARIANTS[*]}")

echo "[info] Evaluating real-stream Phase C severity sweep"
python evaluation/phaseC_scheduler_ablation_real.py \
  --logs_root "${LOGS_ROOT}" \
  --log_experiment run_real_adaptive \
  --log_run_id "${TRAIN_RUN_ID}" \
  --datasets "${REAL_DATASETS}" \
  --model_variants "${VARIANT_CSV}" \
  --seeds "${SEEDS}" \
  --output_dir results/phaseC_scheduler_ablation_real

echo "[done] Commands queued. Review logs under ${LOGS_ROOT}."
