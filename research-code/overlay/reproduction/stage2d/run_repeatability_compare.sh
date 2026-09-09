#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
MEDIC_AD_LOG_DIR=${MEDIC_AD_LOG_DIR:-/home/logs/medic-ad/training-multisample}
MEDIC_AD_COMPARE_NAME=${MEDIC_AD_COMPARE_NAME:-vqarad-repeatability-comparison}
if [[ ! "${MEDIC_AD_COMPARE_NAME}" =~ ^[a-z0-9-]+$ ]]; then
    echo "Invalid MEDIC_AD_COMPARE_NAME: ${MEDIC_AD_COMPARE_NAME}" >&2
    exit 2
fi
LOG_FILE=${MEDIC_AD_LOG_DIR}/${MEDIC_AD_COMPARE_NAME}.json
STATUS_FILE=${MEDIC_AD_LOG_DIR}/${MEDIC_AD_COMPARE_NAME}.status

if [[ -e "${LOG_FILE}" ]]; then
    echo "Refusing to overwrite existing comparison: ${LOG_FILE}" >&2
    exit 2
fi

echo RUNNING > "${STATUS_FILE}"
export MEDIC_AD_LOG_DIR
set +e
"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2d/compare_repeatability.py" \
    2>&1 | tee "${LOG_FILE}"
COMPARE_EXIT=${PIPESTATUS[0]}
set -e

if [[ ${COMPARE_EXIT} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi
exit "${COMPARE_EXIT}"
