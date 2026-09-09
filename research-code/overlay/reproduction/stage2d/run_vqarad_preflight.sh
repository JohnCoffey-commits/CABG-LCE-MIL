#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/training-multisample}
LOG_FILE=${LOG_DIR}/vqarad-tiny-preflight.log
STATUS_FILE=${LOG_DIR}/vqarad-tiny-preflight.status

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export MEDIC_AD_VQARAD_TINY_ROOT=${MEDIC_AD_VQARAD_TINY_ROOT:-/home/data/medic-ad/training-tiny-vqarad}
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export TOKENIZERS_PARALLELISM=false

"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2d/preflight_vqarad_tiny.py"

echo SUCCESS > "${STATUS_FILE}"
trap - ERR
