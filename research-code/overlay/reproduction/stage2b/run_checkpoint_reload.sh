#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
MEDIC_AD_TRAIN_OUTPUT=${MEDIC_AD_TRAIN_OUTPUT:-/home/outputs/medic-ad/training-smoke/stage1-one-step-v1}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/training-smoke}
LOG_FILE=${LOG_DIR}/stage1-checkpoint-reload.log
STATUS_FILE=${LOG_DIR}/stage1-checkpoint-reload.status
VRAM_CSV=${LOG_DIR}/stage1-checkpoint-reload-vram.csv
VRAM_PEAK=${LOG_DIR}/stage1-checkpoint-reload-peak-vram-used-mib.txt

mkdir -p "${LOG_DIR}"
echo RUNNING > "${STATUS_FILE}"

export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export MEDIC_AD_TRAIN_OUTPUT
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
trap cleanup_monitor EXIT

set +e
"${TRAIN_ENV}/bin/python" \
    "${MEDIC_AD_ROOT}/qwen-vl-finetune/tools/verify_smoke_checkpoint.py" \
    > >(tee "${LOG_FILE}") 2>&1
VERIFY_EXIT=$?
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

if [[ ${VERIFY_EXIT} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi

exit "${VERIFY_EXIT}"
