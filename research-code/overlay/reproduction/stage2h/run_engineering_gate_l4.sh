#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/Medic-AD/worktrees/stage2h-lad-mil-v2}
MEDIC_AD_ENGINEERING_TAG=${MEDIC_AD_ENGINEERING_TAG:-v1}
PROTOCOL_VERSION=${MEDIC_AD_PROTOCOL_VERSION:-2.1}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
MODEL_ROOT=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
IMAGE_ROOT=${MEDIC_AD_STAGE2H_IMAGE_ROOT:-/home/data/medic-ad/official/med_anomaly}
DATA_LOCK=${MEDIC_AD_STAGE2H_DATA_LOCK:-/home/data/medic-ad/stage2h-lad-mil-v2-${MEDIC_AD_ENGINEERING_TAG}}
LOG_ROOT=${LOG_ROOT:-/home/Medic-AD/logs/stage2h-lad-mil-v2/engineering-gate-${MEDIC_AD_ENGINEERING_TAG}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/Medic-AD/outputs/stage2h-lad-mil-v2/engineering-gate-${MEDIC_AD_ENGINEERING_TAG}}
STATUS_FILE=${LOG_ROOT}/run.status
BASE_MODEL_REVISION=b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9
BASE_CHECKPOINT_FINGERPRINT=9bc1b0de134f6a94f3ddaa6e6b0f3566c4ee55a2dbdccea27e2cb8432a02e16d
STAGE2G_MANIFEST_SHA256=f4e3c27042296ef52ab0dd4707452ed238386cbb5dbd7bf843d01b9880be485b

if [[ "${PROTOCOL_VERSION}" != "2.1" && "${PROTOCOL_VERSION}" != "2.2" ]]; then
    echo "Unsupported Stage 2H protocol version: ${PROTOCOL_VERSION}" >&2
    exit 2
fi

for path in \
    "${TRAIN_ENV}/bin/python" \
    "${MODEL_ROOT}/config.json" \
    "${MODEL_ROOT}/.medic-ad-pinned-revision" \
    "${IMAGE_ROOT}" \
    "${MEDIC_AD_ROOT}/reproduction/stage2g/data/evaluation-manifest.jsonl" \
    "${MEDIC_AD_ROOT}/reproduction/stage2h/build_dataset.py" \
    "${MEDIC_AD_ROOT}/reproduction/stage2h/build_recalibration_dataset_v2_2.py" \
    "${MEDIC_AD_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py" \
    "${MEDIC_AD_ROOT}/qwen-vl-finetune/scripts/zero3_bf16.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required Stage 2H path: ${path}" >&2
        exit 3
    fi
done
if [[ "$(<"${MODEL_ROOT}/.medic-ad-pinned-revision")" != "${BASE_MODEL_REVISION}" ]]; then
    echo "Lingshu pinned revision mismatch." >&2
    exit 4
fi
if [[ -e "${LOG_ROOT}" ]] || [[ -e "${OUTPUT_ROOT}" ]] || [[ -e "${DATA_LOCK}" ]]; then
    echo "Refusing to overwrite Stage 2H Engineering Gate evidence." >&2
    exit 5
fi

mkdir -p "${LOG_ROOT}/tests" "${OUTPUT_ROOT}"
echo RUNNING > "${STATUS_FILE}"
CURRENT_PHASE=initialization
mark_failed() {
    local code=${1:-$?}
    printf 'FAILED phase=%s exit=%s\n' "${CURRENT_PHASE}" "${code}" > "${STATUS_FILE}"
    exit "${code}"
}
trap 'failure_code=$?; mark_failed "${failure_code}"' ERR

cd "${MEDIC_AD_ROOT}"
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/home/cache/pip}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/cache/torch_extensions}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TOKENIZERS_PARALLELISM=false
export MEDIC_AD_STAGE2H_IMAGE_ROOT="${IMAGE_ROOT}"
export MEDIC_AD_STAGE2H_TRAIN_ANNOTATION="${DATA_LOCK}/train.json"
export MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION="${DATA_LOCK}/calibration.json"
export MEDIC_AD_STAGE2H_INTERNAL_TEST_ANNOTATION="${DATA_LOCK}/internal-test.json"
export MEDIC_AD_PROTOCOL_VERSION="${PROTOCOL_VERSION}"

MONITOR_PID=
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
    if [[ -n "${MONITOR_PID}" ]]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=
    fi
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
run_monitored() {
    local monitor_file=$1
    local peak_file=$2
    local log_file=$3
    shift 3
    start_monitor "${monitor_file}"
    trap 'failure_code=$?; stop_monitor; mark_failed "${failure_code}"' ERR
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local command_exit=${PIPESTATUS[0]}
    set -e
    stop_monitor
    trap 'failure_code=$?; mark_failed "${failure_code}"' ERR
    write_peak "${monitor_file}" "${peak_file}"
    if [[ ${command_exit} -ne 0 ]]; then
        return "${command_exit}"
    fi
}

CURRENT_PHASE=dataset_lock
if [[ "${PROTOCOL_VERSION}" == "2.2" ]]; then
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/build_recalibration_dataset_v2_2.py \
        --stage2g-manifest reproduction/stage2g/data/evaluation-manifest.jsonl \
        --expected-stage2g-manifest-sha256 "${STAGE2G_MANIFEST_SHA256}" \
        --output-dir "${DATA_LOCK}" \
        2>&1 | tee "${LOG_ROOT}/dataset-lock.log"
else
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/build_dataset.py \
        --stage2g-manifest reproduction/stage2g/data/evaluation-manifest.jsonl \
        --expected-stage2g-manifest-sha256 "${STAGE2G_MANIFEST_SHA256}" \
        --output-dir "${DATA_LOCK}" \
        2>&1 | tee "${LOG_ROOT}/dataset-lock.log"
fi

CURRENT_PHASE=unit_tests
"${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_static_contracts.py \
    --repo-root "${MEDIC_AD_ROOT}" \
    --output "${LOG_ROOT}/tests/static-contracts.json" \
    2>&1 | tee "${LOG_ROOT}/tests/static-contracts.log"
"${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_evidence_math.py \
    --output "${LOG_ROOT}/tests/evidence-math.json" \
    2>&1 | tee "${LOG_ROOT}/tests/evidence-math.log"
"${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_lambda_calibration.py \
    --output "${LOG_ROOT}/tests/lambda-calibration.json" \
    2>&1 | tee "${LOG_ROOT}/tests/lambda-calibration.log"
"${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_dataset_alignment.py \
    --output "${LOG_ROOT}/tests/dataset-alignment.json" \
    2>&1 | tee "${LOG_ROOT}/tests/dataset-alignment.log"
"${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_stage2h_adapter_schema.py \
    --work-dir "${LOG_ROOT}/tests/adapter-schema-work" \
    --output "${LOG_ROOT}/tests/adapter-schema.json" \
    2>&1 | tee "${LOG_ROOT}/tests/adapter-schema.log"
"${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_lad_mil_component.py \
    --fixture-dir reproduction/stage2f/fixtures/pre-change \
    --output "${LOG_ROOT}/tests/lad-mil-component.json" \
    2>&1 | tee "${LOG_ROOT}/tests/lad-mil-component.log"
if [[ "${PROTOCOL_VERSION}" == "2.2" ]]; then
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/tests/test_recalibration_v2_2.py \
        --output "${LOG_ROOT}/tests/recalibration-v2-2.json" \
        2>&1 | tee "${LOG_ROOT}/tests/recalibration-v2-2.log"
fi

CURRENT_PHASE=preflight
if [[ "${PROTOCOL_VERSION}" == "2.2" ]]; then
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/preflight_v2_2.py \
        --repo-root "${MEDIC_AD_ROOT}" \
        --dataset-lock "${DATA_LOCK}" \
        --image-root "${IMAGE_ROOT}" \
        --base-model "${MODEL_ROOT}" \
        --expected-base-revision "${BASE_MODEL_REVISION}" \
        --output "${LOG_ROOT}/preflight.json" \
        2>&1 | tee "${LOG_ROOT}/preflight.log"
else
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/preflight.py \
        --repo-root "${MEDIC_AD_ROOT}" \
        --dataset-lock "${DATA_LOCK}" \
        --image-root "${IMAGE_ROOT}" \
        --base-model "${MODEL_ROOT}" \
        --expected-base-revision "${BASE_MODEL_REVISION}" \
        --output "${LOG_ROOT}/preflight.json" \
        2>&1 | tee "${LOG_ROOT}/preflight.log"
fi

TRAIN_MANIFEST_SHA256=$("${TRAIN_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"]["train"])' "${LOG_ROOT}/preflight.json")
EVALUATION_MANIFEST_SHA256=$("${TRAIN_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"]["internal-test"])' "${LOG_ROOT}/preflight.json")
IMPLEMENTATION_SOURCE_FINGERPRINT=$("${TRAIN_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["implementation_source_fingerprint"])' "${LOG_ROOT}/preflight.json")

CURRENT_PHASE=lambda_calibration
if [[ "${PROTOCOL_VERSION}" == "2.2" ]]; then
    run_monitored \
        "${LOG_ROOT}/calibration-vram.csv" \
        "${LOG_ROOT}/calibration-external-peak-mib.txt" \
        "${LOG_ROOT}/calibration.log" \
        "${TRAIN_ENV}/bin/python" reproduction/stage2h/calibrate_lambda.py \
        --base-model "${MODEL_ROOT}" \
        --expected-images 12 \
        --candidates 0.0001 0.0003 0.001 0.003 \
        --protocol-version 2.2 \
        --output "${LOG_ROOT}/calibration.json"
    CURRENT_PHASE=candidate_artifact
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/candidate_measurements_v2_2.py \
        --calibration "${LOG_ROOT}/calibration.json" \
        --external-peak "${LOG_ROOT}/calibration-external-peak-mib.txt" \
        --output "${LOG_ROOT}/candidate-measurements.json" \
        2>&1 | tee "${LOG_ROOT}/candidate-measurements.log"
else
    run_monitored \
        "${LOG_ROOT}/calibration-vram.csv" \
        "${LOG_ROOT}/calibration-external-peak-mib.txt" \
        "${LOG_ROOT}/calibration.log" \
        "${TRAIN_ENV}/bin/python" reproduction/stage2h/calibrate_lambda.py \
        --base-model "${MODEL_ROOT}" \
        --output "${LOG_ROOT}/calibration.json"
fi
CALIBRATION_STATUS=$("${TRAIN_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${LOG_ROOT}/calibration.json")
if [[ "${CALIBRATION_STATUS}" == "PAUSE" ]]; then
    CURRENT_PHASE=lambda_calibration_pause
    if [[ "${PROTOCOL_VERSION}" == "2.2" ]]; then
        "${TRAIN_ENV}/bin/python" reproduction/stage2h/finalize_calibration_pause_v2_2.py \
            --root "${LOG_ROOT}" \
            --output "${LOG_ROOT}/engineering-gate.json" \
            2>&1 | tee "${LOG_ROOT}/engineering-gate.log"
        printf 'COMPLETE decision=PAUSE terminal_gate=lambda_recalibration\n' > "${STATUS_FILE}"
    else
        "${TRAIN_ENV}/bin/python" reproduction/stage2h/finalize_calibration_pause.py \
            --root "${LOG_ROOT}" \
            --output "${LOG_ROOT}/engineering-gate.json" \
            2>&1 | tee "${LOG_ROOT}/engineering-gate.log"
        printf 'COMPLETE decision=PAUSE terminal_gate=lambda_calibration\n' > "${STATUS_FILE}"
    fi
    trap - ERR
    printf 'Stage 2H-E complete: PAUSE at lambda calibration\n'
    exit 0
fi
if [[ "${CALIBRATION_STATUS}" != "SUCCESS" ]]; then
    echo "Unexpected Stage 2H calibration status: ${CALIBRATION_STATUS}" >&2
    exit 21
fi
SELECTED_LAMBDA=$("${TRAIN_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["calibration"]["selected_lambda"])' "${LOG_ROOT}/calibration.json")
printf '%s\n' "${SELECTED_LAMBDA}" > "${LOG_ROOT}/selected-lambda.txt"

run_training() {
    local method_dir=$1
    local method_id=$2
    local weight=$3
    local port=$4
    local run_root="${LOG_ROOT}/runs/${method_dir}"
    local train_output="${OUTPUT_ROOT}/${method_dir}"
    mkdir -p "${run_root}"
    export MEDIC_AD_RUN_ID="stage2h-e-v${PROTOCOL_VERSION//./}-${method_dir}-seed42-step2"
    export MEDIC_AD_STAGE=2H-E
    export MEDIC_AD_STAGE2H_STRICT_SCHEMA=1
    export MEDIC_AD_BASE_MODEL_REVISION="${BASE_MODEL_REVISION}"
    export MEDIC_AD_BASE_CHECKPOINT_FINGERPRINT="${BASE_CHECKPOINT_FINGERPRINT}"
    export MEDIC_AD_DATASET_MANIFEST_SHA256="${TRAIN_MANIFEST_SHA256}"
    export MEDIC_AD_TRAINING_MANIFEST_SHA256="${TRAIN_MANIFEST_SHA256}"
    export MEDIC_AD_EVALUATION_MANIFEST_SHA256="${EVALUATION_MANIFEST_SHA256}"
    export MEDIC_AD_IMPLEMENTATION_SOURCE_FINGERPRINT="${IMPLEMENTATION_SOURCE_FINGERPRINT}"
    export MEDIC_AD_TRAINABLE_SCOPE_OUTPUT="${run_root}/trainable-scope.json"
    export MEDIC_AD_STAGE2H_INITIAL_STATE_OUTPUT="${run_root}/initial.safetensors"
    export MEDIC_AD_STAGE2H_TRAIN_MEMORY_OUTPUT="${run_root}/training-memory.json"
    run_monitored \
        "${run_root}/training-vram.csv" \
        "${run_root}/training-external-peak-mib.txt" \
        "${run_root}/training.log" \
        "${TRAIN_ENV}/bin/python" -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_addr=127.0.0.1 \
        --master_port="${port}" \
        qwen-vl-finetune/qwenvl/train/train_qwen.py \
        --deepspeed qwen-vl-finetune/scripts/zero3_bf16.json \
        --model_name_or_path "${MODEL_ROOT}" \
        --dataset_use medic_ad_stage2h_train \
        --train_type anomaly_evidence \
        --evidence_loss_weight "${weight}" \
        --evidence_trace_output "${run_root}/evidence-trace.jsonl" \
        --stage2h_step_timing_output "${run_root}/step-timing.jsonl" \
        --tune_mm_llm False \
        --tune_mm_mlp False \
        --tune_mm_vision False \
        --tune_mm_vision_decoder False \
        --tune_mm_vpt True \
        --tune_mm_anomaly True \
        --tune_mm_diff False \
        --diff_only_mode False \
        --reset_vpt True \
        --reset_anomaly True \
        --vpt_tokens_number 10 \
        --num_pooling_size 4 \
        --anomaly_query_mode single \
        --gradient_audit_interval 1 \
        --gradient_audit_output "${run_root}/gradient-audit.jsonl" \
        --gradient_audit_strict True \
        --bf16 True \
        --output_dir "${train_output}" \
        --trainable_state_output "${run_root}/adapter.safetensors" \
        --max_steps 2 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_pixels 50176 \
        --min_pixels 784 \
        --eval_strategy no \
        --save_strategy no \
        --skip_final_model_save True \
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
        --data_packing False \
        --remove_unused_columns False \
        --seed 42 \
        --data_seed 42 \
        --full_determinism False \
        --report_to none
    "${TRAIN_ENV}/bin/python" reproduction/stage2h/compare_states.py \
        --initial "${run_root}/initial.safetensors" \
        --final "${run_root}/adapter.safetensors" \
        --output "${run_root}/parameter-change.json" \
        2>&1 | tee "${run_root}/parameter-change.log"
}

run_evaluation() {
    local method_dir=$1
    local method_id=$2
    local weight=$3
    local run_root="${LOG_ROOT}/runs/${method_dir}"
    run_monitored \
        "${run_root}/evaluation-vram.csv" \
        "${run_root}/evaluation-external-peak-mib.txt" \
        "${run_root}/evaluation.log" \
        "${TRAIN_ENV}/bin/python" reproduction/stage2h/evaluate_stage2h.py \
        --base-model "${MODEL_ROOT}" \
        --adapter "${run_root}/adapter.safetensors" \
        --data-root "${IMAGE_ROOT}" \
        --evaluation-manifest "${DATA_LOCK}/internal-test-manifest.jsonl" \
        --expected-evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA256}" \
        --training-manifest-sha256 "${TRAIN_MANIFEST_SHA256}" \
        --implementation-source-fingerprint "${IMPLEMENTATION_SOURCE_FINGERPRINT}" \
        --base-model-revision "${BASE_MODEL_REVISION}" \
        --base-checkpoint-fingerprint "${BASE_CHECKPOINT_FINGERPRINT}" \
        --run-id "stage2h-e-v${PROTOCOL_VERSION//./}-${method_dir}-seed42-step2" \
        --method-id "${method_id}" \
        --evidence-loss-weight "${weight}" \
        --protocol-version "${PROTOCOL_VERSION}" \
        --predictions "${run_root}/predictions.jsonl" \
        --output "${run_root}/evaluation.json"
}

CURRENT_PHASE=b0_as_training
run_training b0-as medic-ad-b0-as 0.0 29561
CURRENT_PHASE=b0_as_evaluation
run_evaluation b0-as medic-ad-b0-as 0.0
CURRENT_PHASE=lad_mil_training
run_training lad-mil-v2 lad-mil-v2 "${SELECTED_LAMBDA}" 29562
CURRENT_PHASE=lad_mil_evaluation
run_evaluation lad-mil-v2 lad-mil-v2 "${SELECTED_LAMBDA}"

CURRENT_PHASE=engineering_gate_verification
"${TRAIN_ENV}/bin/python" reproduction/stage2h/verify_engineering_gate.py \
    --root "${LOG_ROOT}" \
    --protocol-version "${PROTOCOL_VERSION}" \
    --output "${LOG_ROOT}/engineering-gate.json" \
    2>&1 | tee "${LOG_ROOT}/engineering-gate.log"
DECISION=$("${TRAIN_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${LOG_ROOT}/engineering-gate.json")
printf 'COMPLETE decision=%s\n' "${DECISION}" > "${STATUS_FILE}"
trap - ERR
printf 'Stage 2H-E complete: %s\n' "${DECISION}"
