#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_SEED=${MEDIC_AD_SEED:?Set MEDIC_AD_SEED to 42, 123, or 2026}
case "${MEDIC_AD_SEED}" in
    42|123|2026) ;;
    *) echo "Unsupported MEDIC_AD_SEED: ${MEDIC_AD_SEED}" >&2; exit 2 ;;
esac
MEDIC_AD_RUN_ID=seed-${MEDIC_AD_SEED}
MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
MODEL_ROOT=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
DATA_ROOT=${MEDIC_AD_BASELINE_VQARAD_ROOT:-/home/data/medic-ad/baseline-vqarad-v1}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/baseline-vqarad}
OUTPUT_ROOT=${MEDIC_AD_BASELINE_OUTPUT:-/home/outputs/medic-ad/baseline-vqarad/${MEDIC_AD_RUN_ID}}
ADAPTER_ROOT=${MEDIC_AD_ADAPTER_ROOT:-/home/checkpoints/medic-ad/baseline-vqarad}
ADAPTER=${ADAPTER_ROOT}/${MEDIC_AD_RUN_ID}.safetensors
ADAPTER_MANIFEST=${ADAPTER_ROOT}/${MEDIC_AD_RUN_ID}.manifest.json
RESULT=${LOG_DIR}/${MEDIC_AD_RUN_ID}-generation.json
TRAIN_LOG=${LOG_DIR}/${MEDIC_AD_RUN_ID}-training.log
TRAIN_VERIFY_LOG=${LOG_DIR}/${MEDIC_AD_RUN_ID}-training-verification.json
GENERATION_LOG=${LOG_DIR}/${MEDIC_AD_RUN_ID}-generation.log
GENERATION_VERIFY_LOG=${LOG_DIR}/${MEDIC_AD_RUN_ID}-generation-verification.json
STATUS_FILE=${LOG_DIR}/${MEDIC_AD_RUN_ID}.status
GRADIENT_AUDIT=${LOG_DIR}/${MEDIC_AD_RUN_ID}-gradient-audit.jsonl
LOSS_TRACE=${LOG_DIR}/${MEDIC_AD_RUN_ID}-loss-trace.jsonl
TRAIN_VRAM_CSV=${LOG_DIR}/${MEDIC_AD_RUN_ID}-training-vram.csv
TRAIN_VRAM_PEAK=${LOG_DIR}/${MEDIC_AD_RUN_ID}-training-peak-vram-used-mib.txt
GENERATION_VRAM_CSV=${LOG_DIR}/${MEDIC_AD_RUN_ID}-generation-vram.csv
GENERATION_VRAM_PEAK=${LOG_DIR}/${MEDIC_AD_RUN_ID}-generation-peak-vram-used-mib.txt
TRAIN_ENTRY=${MEDIC_AD_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py
TRAIN_VERIFY_ENTRY=${MEDIC_AD_ROOT}/reproduction/stage2e/verify_baseline_training.py
GENERATION_ENTRY=${MEDIC_AD_ROOT}/reproduction/stage2e/generate_vqarad_metrics.py
GENERATION_VERIFY_ENTRY=${MEDIC_AD_ROOT}/reproduction/stage2e/verify_generation_result.py
DEEPSPEED_CONFIG=${MEDIC_AD_ROOT}/qwen-vl-finetune/scripts/zero3_bf16.json

mkdir -p "${LOG_DIR}" "${ADAPTER_ROOT}"
for path in "${TRAIN_ENV}/bin/python" "${MODEL_ROOT}/config.json" "${DATA_ROOT}/manifest.json" \
    "${TRAIN_ENTRY}" "${TRAIN_VERIFY_ENTRY}" "${GENERATION_ENTRY}" "${GENERATION_VERIFY_ENTRY}" "${DEEPSPEED_CONFIG}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
if [[ -e "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty output directory: ${OUTPUT_ROOT}" >&2
    exit 4
fi
for path in "${ADAPTER}" "${ADAPTER_MANIFEST}" "${RESULT}" "${TRAIN_LOG}" "${TRAIN_VERIFY_LOG}" \
    "${GENERATION_LOG}" "${GENERATION_VERIFY_LOG}" "${GRADIENT_AUDIT}" "${LOSS_TRACE}" \
    "${TRAIN_VRAM_CSV}" "${TRAIN_VRAM_PEAK}" "${GENERATION_VRAM_CSV}" "${GENERATION_VRAM_PEAK}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to overwrite existing evidence: ${path}" >&2
        exit 5
    fi
done
mkdir -p "${OUTPUT_ROOT}"
echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

export MEDIC_AD_RUN_ID
export MEDIC_AD_BASELINE_OUTPUT=${OUTPUT_ROOT}
export MEDIC_AD_BASELINE_LOG_DIR=${LOG_DIR}
export MEDIC_AD_TRAINABLE_STATE_OUTPUT=${ADAPTER}
export MEDIC_AD_BASELINE_VQARAD_TRAIN_ANNOTATION=${DATA_ROOT}/train.json
export MEDIC_AD_BASELINE_VQARAD_VALIDATION_ANNOTATION=${DATA_ROOT}/validation.json
export MEDIC_AD_BASELINE_VQARAD_IMAGE_ROOT=${DATA_ROOT}/images
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/home/cache/pip}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/cache/torch_extensions}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TOKENIZERS_PARALLELISM=false

start_monitor() {
    local output=$1
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits
            sleep 0.5
        done
    ) > "${output}" 2>&1 &
    MONITOR_PID=$!
}
stop_monitor() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
write_peak() {
    local input=$1
    local output=$2
    awk -F',' '
        BEGIN { peak = 0 }
        {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            if (($2 + 0) > peak) peak = $2 + 0
        }
        END { print peak }
    ' "${input}" > "${output}"
}

start_monitor "${TRAIN_VRAM_CSV}"
trap 'stop_monitor; mark_failed' ERR
trap stop_monitor EXIT
set +e
"${TRAIN_ENV}/bin/python" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29540 \
    "${TRAIN_ENTRY}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_ROOT}" \
    --dataset_use medic_ad_baseline_vqarad_train \
    --eval_dataset_use medic_ad_baseline_vqarad_validation \
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
    --gradient_audit_interval 16 \
    --gradient_audit_output "${GRADIENT_AUDIT}" \
    --gradient_audit_strict True \
    --loss_trace_output "${LOSS_TRACE}" \
    --bf16 True \
    --output_dir "${OUTPUT_ROOT}" \
    --trainable_state_output "${ADAPTER}" \
    --max_steps 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy steps \
    --eval_steps 16 \
    --eval_on_start False \
    --save_strategy no \
    --skip_final_model_save True \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --log_interval 16 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --data_flatten False \
    --seed "${MEDIC_AD_SEED}" \
    --data_seed "${MEDIC_AD_SEED}" \
    --full_determinism False \
    --report_to none \
    2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
stop_monitor
trap - EXIT
trap mark_failed ERR
write_peak "${TRAIN_VRAM_CSV}" "${TRAIN_VRAM_PEAK}"
if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo FAILED > "${STATUS_FILE}"
    exit "${TRAIN_EXIT}"
fi

set +e
"${TRAIN_ENV}/bin/python" "${TRAIN_VERIFY_ENTRY}" 2>&1 | tee "${TRAIN_VERIFY_LOG}"
TRAIN_VERIFY_EXIT=${PIPESTATUS[0]}
set -e
if [[ ${TRAIN_VERIFY_EXIT} -ne 0 ]]; then
    echo FAILED > "${STATUS_FILE}"
    exit "${TRAIN_VERIFY_EXIT}"
fi

start_monitor "${GENERATION_VRAM_CSV}"
trap 'stop_monitor; mark_failed' ERR
trap stop_monitor EXIT
set +e
"${TRAIN_ENV}/bin/python" "${GENERATION_ENTRY}" \
    --base-model "${MODEL_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output "${RESULT}" \
    --adapter "${ADAPTER}" \
    --seed "${MEDIC_AD_SEED}" \
    2>&1 | tee "${GENERATION_LOG}"
GENERATION_EXIT=${PIPESTATUS[0]}
set -e
stop_monitor
trap - EXIT
trap mark_failed ERR
write_peak "${GENERATION_VRAM_CSV}" "${GENERATION_VRAM_PEAK}"
if [[ ${GENERATION_EXIT} -ne 0 ]]; then
    echo FAILED > "${STATUS_FILE}"
    exit "${GENERATION_EXIT}"
fi

set +e
"${TRAIN_ENV}/bin/python" "${GENERATION_VERIFY_ENTRY}" \
    --result "${RESULT}" \
    --expected-mode trained_adapter \
    --expected-seed "${MEDIC_AD_SEED}" \
    --adapter-manifest "${ADAPTER_MANIFEST}" \
    2>&1 | tee "${GENERATION_VERIFY_LOG}"
GENERATION_VERIFY_EXIT=${PIPESTATUS[0]}
set -e
if [[ ${GENERATION_VERIFY_EXIT} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi
exit "${GENERATION_VERIFY_EXIT}"
