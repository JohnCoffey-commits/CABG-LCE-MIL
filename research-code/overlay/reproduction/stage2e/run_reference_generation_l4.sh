#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_REFERENCE_SEED=${MEDIC_AD_REFERENCE_SEED:-42}
case "${MEDIC_AD_REFERENCE_SEED}" in
    42|123|2026) ;;
    *) echo "Unsupported MEDIC_AD_REFERENCE_SEED: ${MEDIC_AD_REFERENCE_SEED}" >&2; exit 2 ;;
esac
MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
DATA_ROOT=${MEDIC_AD_BASELINE_VQARAD_ROOT:-/home/data/medic-ad/baseline-vqarad-v1}
MODEL_ROOT=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/baseline-vqarad}
RESULT=${MEDIC_AD_REFERENCE_RESULT:-${LOG_DIR}/reference-seed-${MEDIC_AD_REFERENCE_SEED}-generation.json}
RUN_LOG=${LOG_DIR}/reference-seed-${MEDIC_AD_REFERENCE_SEED}-generation.log
VERIFY_LOG=${LOG_DIR}/reference-seed-${MEDIC_AD_REFERENCE_SEED}-generation-verification.json
STATUS_FILE=${LOG_DIR}/reference-seed-${MEDIC_AD_REFERENCE_SEED}-generation.status
VRAM_CSV=${LOG_DIR}/reference-seed-${MEDIC_AD_REFERENCE_SEED}-generation-vram.csv
VRAM_PEAK=${LOG_DIR}/reference-seed-${MEDIC_AD_REFERENCE_SEED}-generation-peak-vram-used-mib.txt

mkdir -p "${LOG_DIR}"
for path in "${TRAIN_ENV}/bin/python" "${DATA_ROOT}/manifest.json" "${MODEL_ROOT}/config.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
for path in "${RESULT}" "${RUN_LOG}" "${VERIFY_LOG}" "${VRAM_CSV}" "${VRAM_PEAK}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to overwrite existing evidence: ${path}" >&2
        exit 4
    fi
done

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export TOKENIZERS_PARALLELISM=false

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
trap 'cleanup_monitor; mark_failed' ERR
trap cleanup_monitor EXIT

set +e
"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2e/generate_vqarad_metrics.py" \
    --base-model "${MODEL_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output "${RESULT}" \
    --seed "${MEDIC_AD_REFERENCE_SEED}" \
    2>&1 | tee "${RUN_LOG}"
GENERATION_EXIT=${PIPESTATUS[0]}
set -e

cleanup_monitor
trap - EXIT
trap mark_failed ERR
awk -F',' '
    BEGIN { peak = 0 }
    {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if (($2 + 0) > peak) peak = $2 + 0
    }
    END { print peak }
' "${VRAM_CSV}" > "${VRAM_PEAK}"
if [[ ${GENERATION_EXIT} -ne 0 ]]; then
    echo FAILED > "${STATUS_FILE}"
    exit "${GENERATION_EXIT}"
fi

set +e
"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2e/verify_generation_result.py" \
    --result "${RESULT}" \
    --expected-mode untrained_reference \
    --expected-seed "${MEDIC_AD_REFERENCE_SEED}" \
    2>&1 | tee "${VERIFY_LOG}"
VERIFY_EXIT=${PIPESTATUS[0]}
set -e
if [[ ${VERIFY_EXIT} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi
exit "${VERIFY_EXIT}"
