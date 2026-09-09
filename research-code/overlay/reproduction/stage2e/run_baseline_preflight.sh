#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
DATA_ROOT=${MEDIC_AD_BASELINE_VQARAD_ROOT:-/home/data/medic-ad/baseline-vqarad-v1}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/baseline-vqarad}
LOG_FILE=${LOG_DIR}/preflight.log
STATUS_FILE=${LOG_DIR}/preflight.status

mkdir -p "${LOG_DIR}"
for path in "${TRAIN_ENV}/bin/python" "${DATA_ROOT}/manifest.json" "${MEDIC_AD_ROOT}/reproduction/stage2e/preflight_vqarad_baseline.py"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
if [[ -e "${LOG_FILE}" ]]; then
    echo "Refusing to overwrite existing evidence: ${LOG_FILE}" >&2
    exit 4
fi
exec > >(tee "${LOG_FILE}") 2>&1

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

export MEDIC_AD_BASELINE_VQARAD_ROOT=${DATA_ROOT}
export MEDIC_AD_LINGSHU_MODEL=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
export MEDIC_AD_BASELINE_VQARAD_TRAIN_ANNOTATION=${DATA_ROOT}/train.json
export MEDIC_AD_BASELINE_VQARAD_VALIDATION_ANNOTATION=${DATA_ROOT}/validation.json
export MEDIC_AD_BASELINE_VQARAD_IMAGE_ROOT=${DATA_ROOT}/images
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export HF_HOME=${HF_HOME:-/home/cache/huggingface}

"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2e/preflight_vqarad_baseline.py"

echo SUCCESS > "${STATUS_FILE}"
trap - ERR
