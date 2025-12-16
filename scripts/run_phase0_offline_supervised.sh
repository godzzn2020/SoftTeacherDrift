#!/usr/bin/env bash
# Phase0 离线监督实验多 GPU 启动脚本。
# 说明：
#   - 默认把 DATASETS 中的真实数据集按顺序平均分配到两张 GPU（可通过 GPU_IDS 配置）；
#   - 每块 GPU 会调用一次 experiments/phase0_offline_supervised.py，并行运行；
#   - 可通过环境变量覆盖默认参数，例如：
#         DATASETS="Airlines Electricity NOAA INSECTS_abrupt_balanced" \
#         GPU_IDS="0,1" \
#         SEEDS="1,2,3" \
#         ./scripts/run_phase0_offline_supervised.sh

set -euo pipefail

DATASETS="${DATASETS:-Airlines Electricity NOAA INSECTS_abrupt_balanced}"
SEEDS="${SEEDS:-1,2,3}"
LABELED_RATIO="${LABELED_RATIO:-1.0}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.2}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
RUN_TAG="${RUN_TAG:-phase0_mlp_full_supervised}"
GPU_IDS="${GPU_IDS:-0,1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=', ' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "[error] 未检测到 GPU_IDS，至少需要一张 GPU"
  exit 1
fi

IFS=', ' read -r -a DATASET_ARRAY <<< "${DATASETS}"
if [[ ${#DATASET_ARRAY[@]} -eq 0 ]]; then
  echo "[error] DATASETS 为空"
  exit 1
fi

declare -a DATASET_TO_GPU

echo "[info] DATASETS=${DATASETS}"
echo "[info] GPU assignment:"
for idx in "${!DATASET_ARRAY[@]}"; do
  ds="${DATASET_ARRAY[$idx]}"
  ds="${ds// /}"
  [[ -z "$ds" ]] && continue
  gpu_idx=$(( idx % ${#GPU_ARRAY[@]} ))
  gpu="${GPU_ARRAY[$gpu_idx]}"
  DATASET_TO_GPU[$idx]="$gpu"
  echo "  dataset ${ds} -> GPU ${gpu}"
done

declare -a PIDS=()

for idx in "${!DATASET_ARRAY[@]}"; do
  ds="${DATASET_ARRAY[$idx]}"
  ds="${ds// /}"
  [[ -z "$ds" ]] && continue
  gpu="${DATASET_TO_GPU[$idx]}"
  echo "[run] GPU ${gpu} running dataset: ${ds}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python experiments/phase0_offline_supervised.py \
      --datasets "${ds}" \
      --seeds "${SEEDS}" \
      --labeled_ratio "${LABELED_RATIO}" \
      --hidden_dim "${HIDDEN_DIM}" \
      --num_layers "${NUM_LAYERS}" \
      --dropout "${DROPOUT}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --batch_size "${BATCH_SIZE}" \
      --max_epochs "${MAX_EPOCHS}" \
      --lr_scheduler "${LR_SCHEDULER}" \
      --run_tag "${RUN_TAG}" \
      ${EXTRA_ARGS} &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "[done] 所有 Phase0 离线监督任务已完成。"
