#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_RUN_ID=${MEDIC_AD_RUN_ID:-run-a}
if [[ ! "${MEDIC_AD_RUN_ID}" =~ ^run-[a-z0-9-]+$ ]]; then
    echo "Invalid MEDIC_AD_RUN_ID: ${MEDIC_AD_RUN_ID}" >&2
    exit 2
fi

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
MEDIC_AD_TRAIN_PYTHON=${MEDIC_AD_TRAIN_PYTHON:-/home/envs/medic-ad-train/bin/python}
MEDIC_AD_LINGSHU_MODEL=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
MEDIC_AD_VQARAD_TINY_ROOT=${MEDIC_AD_VQARAD_TINY_ROOT:-/home/data/medic-ad/training-tiny-vqarad}
MEDIC_AD_MULTISAMPLE_OUTPUT=${MEDIC_AD_MULTISAMPLE_OUTPUT:-/home/outputs/medic-ad/training-multisample/vqarad-8step-${MEDIC_AD_RUN_ID}}
MEDIC_AD_LOG_DIR=${MEDIC_AD_LOG_DIR:-/home/logs/medic-ad/training-multisample}
MEDIC_AD_GRADIENT_AUDIT=${MEDIC_AD_GRADIENT_AUDIT:-${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}-gradient-audit.jsonl}
MEDIC_AD_LOSS_TRACE=${MEDIC_AD_LOSS_TRACE:-${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}-loss-trace.jsonl}
MEDIC_AD_FULL_DETERMINISM=${MEDIC_AD_FULL_DETERMINISM:-False}
if [[ "${MEDIC_AD_FULL_DETERMINISM}" != True ]] && [[ "${MEDIC_AD_FULL_DETERMINISM}" != False ]]; then
    echo "MEDIC_AD_FULL_DETERMINISM must be True or False." >&2
    exit 2
fi

export MEDIC_AD_RUN_ID MEDIC_AD_VQARAD_TINY_ROOT MEDIC_AD_MULTISAMPLE_OUTPUT
export MEDIC_AD_GRADIENT_AUDIT MEDIC_AD_LOSS_TRACE
export MEDIC_AD_FULL_DETERMINISM
export MEDIC_AD_TINY_VQARAD_TRAIN_ANNOTATION=${MEDIC_AD_VQARAD_TINY_ROOT}/train.json
export MEDIC_AD_TINY_VQARAD_VALIDATION_ANNOTATION=${MEDIC_AD_VQARAD_TINY_ROOT}/validation.json
export MEDIC_AD_TINY_VQARAD_IMAGE_ROOT=${MEDIC_AD_VQARAD_TINY_ROOT}/images
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/home/cache/pip}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/cache/torch_extensions}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TOKENIZERS_PARALLELISM=false

TRAIN_ENTRY=${MEDIC_AD_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py
VERIFY_ENTRY=${MEDIC_AD_ROOT}/reproduction/stage2d/verify_multisample.py
DEEPSPEED_CONFIG=${MEDIC_AD_ROOT}/qwen-vl-finetune/scripts/zero3_bf16.json
RUN_LOG=${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}.log
VERIFY_LOG=${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}-verification.json
STATUS_FILE=${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}.status
VRAM_CSV=${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}-vram.csv
VRAM_PEAK=${MEDIC_AD_LOG_DIR}/vqarad-8step-${MEDIC_AD_RUN_ID}-peak-vram-used-mib.txt

mkdir -p "${MEDIC_AD_LOG_DIR}"
for required_path in \
    "${MEDIC_AD_TRAIN_PYTHON}" \
    "${MEDIC_AD_LINGSHU_MODEL}/config.json" \
    "${MEDIC_AD_VQARAD_TINY_ROOT}/manifest.json" \
    "${MEDIC_AD_VQARAD_TINY_ROOT}/train.json" \
    "${MEDIC_AD_VQARAD_TINY_ROOT}/validation.json" \
    "${TRAIN_ENTRY}" \
    "${VERIFY_ENTRY}" \
    "${DEEPSPEED_CONFIG}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 3
    fi
done

if [[ -e "${MEDIC_AD_MULTISAMPLE_OUTPUT}" ]] && [[ -n "$(find "${MEDIC_AD_MULTISAMPLE_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty output directory: ${MEDIC_AD_MULTISAMPLE_OUTPUT}" >&2
    exit 4
fi
for evidence_path in "${RUN_LOG}" "${VERIFY_LOG}" "${MEDIC_AD_GRADIENT_AUDIT}" "${MEDIC_AD_LOSS_TRACE}" "${VRAM_CSV}" "${VRAM_PEAK}"; do
    if [[ -e "${evidence_path}" ]]; then
        echo "Refusing to overwrite existing evidence: ${evidence_path}" >&2
        exit 5
    fi
done

mkdir -p "${MEDIC_AD_MULTISAMPLE_OUTPUT}"
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
    --master_port=29533 \
    "${TRAIN_ENTRY}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MEDIC_AD_LINGSHU_MODEL}" \
    --dataset_use medic_ad_tiny_vqarad_train \
    --eval_dataset_use medic_ad_tiny_vqarad_validation \
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
    --gradient_audit_interval 4 \
    --gradient_audit_output "${MEDIC_AD_GRADIENT_AUDIT}" \
    --gradient_audit_strict True \
    --loss_trace_output "${MEDIC_AD_LOSS_TRACE}" \
    --bf16 True \
    --output_dir "${MEDIC_AD_MULTISAMPLE_OUTPUT}" \
    --max_steps 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy steps \
    --eval_steps 4 \
    --eval_on_start False \
    --save_strategy no \
    --skip_final_model_save True \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --log_interval 4 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --data_flatten False \
    --seed 42 \
    --data_seed 42 \
    --full_determinism "${MEDIC_AD_FULL_DETERMINISM}" \
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

if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo FAILED > "${STATUS_FILE}"
    exit "${TRAIN_EXIT}"
fi

set +e
"${MEDIC_AD_TRAIN_PYTHON}" "${VERIFY_ENTRY}" 2>&1 | tee "${VERIFY_LOG}"
VERIFY_EXIT=${PIPESTATUS[0]}
set -e

if [[ ${VERIFY_EXIT} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi
exit "${VERIFY_EXIT}"
