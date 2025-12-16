#!/usr/bin/env bash
# Phase0 离线监督：按数据集规模加权分配 GPU，数据集 × seed 全并行。
# 使用方式示例：
#   GPU_IDS="0,1" DATASETS="Airlines Electricity NOAA INSECTS_abrupt_balanced" SEEDS="1,2,3" \
#   RUN_TAG="phase0_mlp_full_supervised" ./scripts/run_phase0_offline_supervised_full_parallel.sh

set -euo pipefail

DATASETS="${DATASETS:-Airlines Electricity NOAA INSECTS_abrupt_balanced}"
SEEDS="${SEEDS:-1,2,3}"
GPU_IDS="${GPU_IDS:-0,1}"
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
EXTRA_ARGS="${EXTRA_ARGS:-}"

declare -A DATASET_WEIGHTS=(
    ["Airlines"]=10
    ["INSECTS_abrupt_balanced"]=5
    ["Electricity"]=3
    ["NOAA"]=2
)

parse_to_array() {
    local input="$1"
    local -n out_ref=$2
    local IFS=','
    read -ra tokens <<< "${input// /,}"
    out_ref=()
    for token in "${tokens[@]}"; do
        token="$(echo "$token" | xargs)"
        [[ -z "$token" ]] && continue
        out_ref+=("$token")
    done
}

parse_to_array "$DATASETS" DATASET_ARRAY
parse_to_array "$SEEDS" SEED_ARRAY
parse_to_array "$GPU_IDS" GPU_ARRAY

if [[ ${#DATASET_ARRAY[@]} -eq 0 ]] || [[ ${#SEED_ARRAY[@]} -eq 0 ]] || [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
    echo "[error] DATASETS/SEEDS/GPU_IDS 不可为空"
    exit 1
fi

declare -a WEIGHTED_COMBOS=()
for ds in "${DATASET_ARRAY[@]}"; do
    weight=${DATASET_WEIGHTS[$ds]:-1}
    for seed in "${SEED_ARRAY[@]}"; do
        WEIGHTED_COMBOS+=("${weight}:${ds}:${seed}")
    done
done

if [[ ${#WEIGHTED_COMBOS[@]} -eq 0 ]]; then
    echo "[error] 无待运行组合"
    exit 1
fi

mapfile -t SORTED_COMBOS < <(printf '%s\n' "${WEIGHTED_COMBOS[@]}" | sort -t: -k1,1nr)
declare -A GPU_LOADS=()
for gpu in "${GPU_ARRAY[@]}"; do
    GPU_LOADS["$gpu"]=0
done

declare -a PIDS=()
for entry in "${SORTED_COMBOS[@]}"; do
    weight="${entry%%:*}"
    rest="${entry#*:}"
    dataset="${rest%%:*}"
    seed="${rest##*:}"
    # 选择当前负载最小的 GPU
    target_gpu=""
    min_load=""
    for gpu in "${GPU_ARRAY[@]}"; do
        load=${GPU_LOADS[$gpu]}
        if [[ -z "$target_gpu" ]] || (( load < min_load )); then
            target_gpu=$gpu
            min_load=$load
        fi
    done
    GPU_LOADS["$target_gpu"]=$(( GPU_LOADS["$target_gpu"] + weight ))
    echo "[dispatch] dataset=${dataset} seed=${seed} weight=${weight} -> GPU ${target_gpu}"
    CUDA_VISIBLE_DEVICES="$target_gpu" \
        python experiments/phase0_offline_supervised.py \
        --datasets "${dataset}" \
        --seeds "${seed}" \
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

echo "[done] 并行 Phase0 实验全部完成。"
