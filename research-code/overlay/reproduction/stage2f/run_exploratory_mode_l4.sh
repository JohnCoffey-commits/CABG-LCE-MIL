#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_QUERY_MODE=${MEDIC_AD_QUERY_MODE:?Set MEDIC_AD_QUERY_MODE to single or multiscale}
MEDIC_AD_SEED=${MEDIC_AD_SEED:?Set MEDIC_AD_SEED to 42, 123, or 2026}
MEDIC_AD_REPEAT=${MEDIC_AD_REPEAT:?Set MEDIC_AD_REPEAT to 1 or 2}
case "${MEDIC_AD_QUERY_MODE}" in
    single) MEDIC_AD_METHOD_ID=b0 ;;
    multiscale) MEDIC_AD_METHOD_ID=a3 ;;
    *) echo "Unsupported MEDIC_AD_QUERY_MODE: ${MEDIC_AD_QUERY_MODE}" >&2; exit 2 ;;
esac
case "${MEDIC_AD_SEED}" in 42|123|2026) ;; *) echo "Unsupported seed" >&2; exit 2 ;; esac
case "${MEDIC_AD_REPEAT}" in 1|2) ;; *) echo "Unsupported repeat" >&2; exit 2 ;; esac

MEDIC_AD_RUN_ID=${MEDIC_AD_RUN_ID_OVERRIDE:-seed-${MEDIC_AD_SEED}-repeat-${MEDIC_AD_REPEAT}-${MEDIC_AD_METHOD_ID}}
if ! [[ "${MEDIC_AD_RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid MEDIC_AD_RUN_ID." >&2
    exit 2
fi
MEDIC_AD_EXPLORATORY_EVIDENCE_TAG=${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG:-}
if [[ -n "${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG}" ]] && ! [[ "${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid MEDIC_AD_EXPLORATORY_EVIDENCE_TAG." >&2
    exit 2
fi
MEDIC_AD_EXPLORATORY_GROUP=${MEDIC_AD_EXPLORATORY_GROUP_OVERRIDE:-exploratory}
if [[ -z "${MEDIC_AD_EXPLORATORY_GROUP_OVERRIDE:-}" ]] && [[ -n "${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG}" ]]; then
    MEDIC_AD_EXPLORATORY_GROUP=${MEDIC_AD_EXPLORATORY_GROUP}-${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG}
fi
if ! [[ "${MEDIC_AD_EXPLORATORY_GROUP}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid MEDIC_AD_EXPLORATORY_GROUP." >&2
    exit 2
fi
MEDIC_AD_GRADIENT_AUDIT_INTERVAL=${MEDIC_AD_GRADIENT_AUDIT_INTERVAL:-16}
if ! [[ "${MEDIC_AD_GRADIENT_AUDIT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid MEDIC_AD_GRADIENT_AUDIT_INTERVAL." >&2
    exit 2
fi
MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
MODEL_ROOT=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
DATA_ROOT=${MEDIC_AD_BASELINE_VQARAD_ROOT:-/home/data/medic-ad/baseline-vqarad-v1}
LOG_ROOT=${LOG_ROOT:-/home/logs/medic-ad/stage2f-fb-maq/${MEDIC_AD_EXPLORATORY_GROUP}/${MEDIC_AD_RUN_ID}}
OUTPUT_ROOT=${MEDIC_AD_EXPLORATORY_OUTPUT:-/home/outputs/medic-ad/stage2f-fb-maq/${MEDIC_AD_EXPLORATORY_GROUP}/${MEDIC_AD_RUN_ID}}
ADAPTER_ROOT=${MEDIC_AD_EXPLORATORY_ADAPTER_ROOT:-/home/checkpoints/medic-ad/stage2f-fb-maq/${MEDIC_AD_EXPLORATORY_GROUP}}
ADAPTER=${ADAPTER_ROOT}/${MEDIC_AD_RUN_ID}.safetensors
ADAPTER_MANIFEST=${ADAPTER_ROOT}/${MEDIC_AD_RUN_ID}.manifest.json
STATUS_FILE=${LOG_ROOT}/run.status
TRAIN_LOG=${LOG_ROOT}/training.log
TRAINABLE_SCOPE=${LOG_ROOT}/trainable-scope.json
GRADIENT_AUDIT=${LOG_ROOT}/gradient-audit.jsonl
LOSS_TRACE=${LOG_ROOT}/loss-trace.jsonl
TRAIN_VRAM_CSV=${LOG_ROOT}/training-vram.csv
TRAIN_VRAM_PEAK=${LOG_ROOT}/training-peak-vram-used-mib.txt
TRAIN_ELAPSED=${LOG_ROOT}/training-elapsed-seconds.txt
GENERATION_LOG=${LOG_ROOT}/generation.log
GENERATION_RESULT=${LOG_ROOT}/generation.json
PREDICTIONS_JSONL=${LOG_ROOT}/predictions.jsonl
GENERATION_VRAM_CSV=${LOG_ROOT}/generation-vram.csv
GENERATION_VRAM_PEAK=${LOG_ROOT}/generation-peak-vram-used-mib.txt
GENERATION_ELAPSED=${LOG_ROOT}/generation-elapsed-seconds.txt
RUN_VERIFICATION=${LOG_ROOT}/run-verification.json
RUN_VERIFICATION_LOG=${LOG_ROOT}/run-verification.log
IMPLEMENTATION_FINGERPRINT_FILE=${LOG_ROOT}/implementation-source-fingerprint.txt
TRAIN_ENTRY=${MEDIC_AD_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py
EVALUATOR=${MEDIC_AD_ROOT}/reproduction/stage2f/evaluate_vqarad_exploratory.py
VERIFIER=${MEDIC_AD_ROOT}/reproduction/stage2f/verify_exploratory_run.py
FINGERPRINT_TOOL=${MEDIC_AD_ROOT}/reproduction/stage2f/source_fingerprint.py
DEEPSPEED_CONFIG=${MEDIC_AD_ROOT}/qwen-vl-finetune/scripts/zero3_bf16.json

BASE_MODEL_REVISION=b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9
BASE_CHECKPOINT_FINGERPRINT=9bc1b0de134f6a94f3ddaa6e6b0f3566c4ee55a2dbdccea27e2cb8432a02e16d
DATASET_MANIFEST_SHA256=8c5ebce131590351758a5bb6f1b7d41d894ba9d584df5f6ea5a67c61130ad118

for path in \
    "${TRAIN_ENV}/bin/python" \
    "${MODEL_ROOT}/config.json" \
    "${MODEL_ROOT}/.medic-ad-pinned-revision" \
    "${DATA_ROOT}/manifest.json" \
    "${TRAIN_ENTRY}" \
    "${EVALUATOR}" \
    "${VERIFIER}" \
    "${FINGERPRINT_TOOL}" \
    "${DEEPSPEED_CONFIG}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
if [[ "$(<"${MODEL_ROOT}/.medic-ad-pinned-revision")" != "${BASE_MODEL_REVISION}" ]]; then
    echo "Lingshu pinned revision mismatch." >&2
    exit 4
fi
OBSERVED_DATASET_SHA256=$(sha256sum "${DATA_ROOT}/manifest.json" | awk '{print $1}')
if [[ "${OBSERVED_DATASET_SHA256}" != "${DATASET_MANIFEST_SHA256}" ]]; then
    echo "Stage 2F dataset manifest SHA256 mismatch." >&2
    exit 5
fi
if [[ -e "${LOG_ROOT}" ]] || [[ -e "${OUTPUT_ROOT}" ]] || [[ -e "${ADAPTER}" ]] || [[ -e "${ADAPTER_MANIFEST}" ]]; then
    echo "Refusing to overwrite existing exploratory evidence for ${MEDIC_AD_RUN_ID}." >&2
    exit 6
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${ADAPTER_ROOT}"
echo RUNNING > "${STATUS_FILE}"
mark_failed() { echo FAILED > "${STATUS_FILE}"; }
trap mark_failed ERR

cd "${MEDIC_AD_ROOT}"
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/home/cache/pip}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/cache/torch_extensions}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TOKENIZERS_PARALLELISM=false
export MEDIC_AD_RUN_ID
export MEDIC_AD_REPEAT
export MEDIC_AD_STAGE=2F
export MEDIC_AD_STAGE2F_STRICT_SCHEMA=1
export MEDIC_AD_BASE_MODEL_REVISION=${BASE_MODEL_REVISION}
export MEDIC_AD_BASE_CHECKPOINT_FINGERPRINT=${BASE_CHECKPOINT_FINGERPRINT}
export MEDIC_AD_DATASET_MANIFEST_SHA256=${DATASET_MANIFEST_SHA256}
export MEDIC_AD_IMPLEMENTATION_SOURCE_FINGERPRINT
MEDIC_AD_IMPLEMENTATION_SOURCE_FINGERPRINT=$(
    "${TRAIN_ENV}/bin/python" "${FINGERPRINT_TOOL}" --repo-root "${MEDIC_AD_ROOT}"
)
printf '%s\n' "${MEDIC_AD_IMPLEMENTATION_SOURCE_FINGERPRINT}" > "${IMPLEMENTATION_FINGERPRINT_FILE}"
export MEDIC_AD_TRAINABLE_SCOPE_OUTPUT=${TRAINABLE_SCOPE}
export MEDIC_AD_TRAINABLE_STATE_OUTPUT=${ADAPTER}
export MEDIC_AD_BASELINE_VQARAD_TRAIN_ANNOTATION=${DATA_ROOT}/train.json
export MEDIC_AD_BASELINE_VQARAD_VALIDATION_ANNOTATION=${DATA_ROOT}/validation.json
export MEDIC_AD_BASELINE_VQARAD_IMAGE_ROOT=${DATA_ROOT}/images
if [[ "${MEDIC_AD_ENABLE_ACTIVATION_DIAGNOSTICS:-0}" == "1" ]]; then
    export MEDIC_AD_ACTIVATION_DIAGNOSTIC_OUTPUT=${LOG_ROOT}/activation-diagnostics.jsonl
fi

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
TRAIN_STARTED=$(date +%s)
set +e
"${TRAIN_ENV}/bin/python" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29562 \
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
    --anomaly_query_mode "${MEDIC_AD_QUERY_MODE}" \
    --gradient_audit_interval "${MEDIC_AD_GRADIENT_AUDIT_INTERVAL}" \
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
TRAIN_FINISHED=$(date +%s)
printf '%s\n' "$((TRAIN_FINISHED - TRAIN_STARTED))" > "${TRAIN_ELAPSED}"
stop_monitor
trap - EXIT
trap mark_failed ERR
write_peak "${TRAIN_VRAM_CSV}" "${TRAIN_VRAM_PEAK}"
if [[ ${TRAIN_EXIT} -ne 0 ]]; then exit "${TRAIN_EXIT}"; fi

start_monitor "${GENERATION_VRAM_CSV}"
trap 'stop_monitor; mark_failed' ERR
trap stop_monitor EXIT
GENERATION_STARTED=$(date +%s)
set +e
"${TRAIN_ENV}/bin/python" "${EVALUATOR}" \
    --base-model "${MODEL_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --adapter "${ADAPTER}" \
    --output "${GENERATION_RESULT}" \
    --predictions-jsonl "${PREDICTIONS_JSONL}" \
    --seed "${MEDIC_AD_SEED}" \
    --repeat "${MEDIC_AD_REPEAT}" \
    --run-id "${MEDIC_AD_RUN_ID}" \
    --anomaly-query-mode "${MEDIC_AD_QUERY_MODE}" \
    --expected-global-step 32 \
    2>&1 | tee "${GENERATION_LOG}"
GENERATION_EXIT=${PIPESTATUS[0]}
set -e
GENERATION_FINISHED=$(date +%s)
printf '%s\n' "$((GENERATION_FINISHED - GENERATION_STARTED))" > "${GENERATION_ELAPSED}"
stop_monitor
trap - EXIT
trap mark_failed ERR
write_peak "${GENERATION_VRAM_CSV}" "${GENERATION_VRAM_PEAK}"
if [[ ${GENERATION_EXIT} -ne 0 ]]; then exit "${GENERATION_EXIT}"; fi

set +e
"${TRAIN_ENV}/bin/python" "${VERIFIER}" \
    --log-root "${LOG_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --adapter "${ADAPTER}" \
    --mode "${MEDIC_AD_QUERY_MODE}" \
    --run-id "${MEDIC_AD_RUN_ID}" \
    --seed "${MEDIC_AD_SEED}" \
    --repeat "${MEDIC_AD_REPEAT}" \
    --gradient-audit-interval "${MEDIC_AD_GRADIENT_AUDIT_INTERVAL}" \
    --output "${RUN_VERIFICATION}" \
    2>&1 | tee "${RUN_VERIFICATION_LOG}"
VERIFY_EXIT=${PIPESTATUS[0]}
set -e
if [[ ${VERIFY_EXIT} -eq 0 ]]; then echo SUCCESS > "${STATUS_FILE}"; else echo FAILED > "${STATUS_FILE}"; fi
exit "${VERIFY_EXIT}"
