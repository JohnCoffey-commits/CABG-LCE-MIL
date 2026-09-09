#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/baseline-vqarad}
LOG_FILE=${LOG_DIR}/prepare.log
STATUS_FILE=${LOG_DIR}/prepare.status

mkdir -p "${LOG_DIR}"
if [[ -e "${LOG_FILE}" ]]; then
    echo "Refusing to overwrite existing evidence: ${LOG_FILE}" >&2
    exit 2
fi
exec > >(tee "${LOG_FILE}") 2>&1

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export MEDIC_AD_BASELINE_VQARAD_ROOT=${MEDIC_AD_BASELINE_VQARAD_ROOT:-/home/data/medic-ad/baseline-vqarad-v1}

"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2e/prepare_vqarad_baseline.py"

echo SUCCESS > "${STATUS_FILE}"
trap - ERR
