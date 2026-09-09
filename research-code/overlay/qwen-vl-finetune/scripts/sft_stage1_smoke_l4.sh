#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
MEDIC_AD_TRAIN_PYTHON=${MEDIC_AD_TRAIN_PYTHON:-/home/envs/medic-ad-train/bin/python}
MEDIC_AD_LINGSHU_MODEL=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
MEDIC_AD_SMOKE_ANNOTATION=${MEDIC_AD_SMOKE_ANNOTATION:-/home/data/medic-ad/training-smoke/stage1_smoke.json}
MEDIC_AD_SMOKE_DATA_ROOT=${MEDIC_AD_SMOKE_DATA_ROOT:-/home/data/medic-ad/training-smoke/images}
MEDIC_AD_TRAIN_OUTPUT=${MEDIC_AD_TRAIN_OUTPUT:-/home/outputs/medic-ad/training-smoke/stage1-one-step-v1}
MEDIC_AD_TRAIN_LOG_DIR=${MEDIC_AD_TRAIN_LOG_DIR:-/home/logs/medic-ad/training-smoke}

export MEDIC_AD_ROOT
export MEDIC_AD_SMOKE_ANNOTATION
export MEDIC_AD_SMOKE_DATA_ROOT
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/home/cache/pip}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/cache/torch_extensions}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TOKENIZERS_PARALLELISM=false

TRAIN_ENTRY=${MEDIC_AD_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py
DEEPSPEED_CONFIG=${MEDIC_AD_ROOT}/qwen-vl-finetune/scripts/zero3_bf16.json
RUN_LOG=${MEDIC_AD_TRAIN_LOG_DIR}/stage1-one-step.log
STATUS_FILE=${MEDIC_AD_TRAIN_LOG_DIR}/stage1-one-step.status
VRAM_CSV=${MEDIC_AD_TRAIN_LOG_DIR}/stage1-one-step-vram.csv
VRAM_PEAK=${MEDIC_AD_TRAIN_LOG_DIR}/stage1-one-step-peak-vram-used-mib.txt

mkdir -p "${MEDIC_AD_TRAIN_LOG_DIR}"

for required_path in \
    "${MEDIC_AD_TRAIN_PYTHON}" \
    "${MEDIC_AD_LINGSHU_MODEL}/config.json" \
    "${MEDIC_AD_SMOKE_ANNOTATION}" \
    "${MEDIC_AD_SMOKE_DATA_ROOT}" \
    "${TRAIN_ENTRY}" \
    "${DEEPSPEED_CONFIG}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 2
    fi
done

if [[ -e "${MEDIC_AD_TRAIN_OUTPUT}" ]] && [[ -n "$(find "${MEDIC_AD_TRAIN_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty output directory: ${MEDIC_AD_TRAIN_OUTPUT}" >&2
    exit 3
fi

mkdir -p "${MEDIC_AD_TRAIN_OUTPUT}"
echo RUNNING > "${STATUS_FILE}"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 0.5
    done
) > "${VRAM_CSV}" 2>&1 &
MONITOR_PID=$!

cleanup_monitor() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

set +e
"${MEDIC_AD_TRAIN_PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29531 \
    "${TRAIN_ENTRY}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MEDIC_AD_LINGSHU_MODEL}" \
    --dataset_use medic_ad_stage1_smoke \
    --train_type default \
    --tune_mm_llm False \
    --tune_mm_mlp False \
    --tune_mm_vision False \
    --tune_mm_vision_decoder False \
    --tune_mm_vpt True \
    --tune_mm_anomaly True \
    --tune_mm_diff False \
    --reset_vpt True \
    --reset_anomaly True \
    --vpt_tokens_number 10 \
    --num_pooling_size 4 \
    --bf16 True \
    --output_dir "${MEDIC_AD_TRAIN_OUTPUT}" \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --log_interval 1 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --data_flatten False \
    --seed 42 \
    --data_seed 42 \
    --report_to none \
    2>&1 | tee "${RUN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

cleanup_monitor
trap - EXIT

awk -F',' '
    BEGIN { peak = 0 }
    {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if (($2 + 0) > peak) peak = $2 + 0
    }
    END { print peak }
' "${VRAM_CSV}" > "${VRAM_PEAK}"

if [[ ${TRAIN_EXIT} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi

exit "${TRAIN_EXIT}"
