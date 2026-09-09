#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/baseline-vqarad}
OUTPUT=${LOG_DIR}/multiseed-aggregate.json
LOG_FILE=${LOG_DIR}/multiseed-aggregate.log
STATUS_FILE=${LOG_DIR}/multiseed-aggregate.status

for seed in 42 123 2026; do
    if [[ ! -e "${LOG_DIR}/seed-${seed}-generation.json" ]]; then
        echo "Missing generation result for seed ${seed}" >&2
        exit 3
    fi
    if [[ ! -e "${LOG_DIR}/reference-seed-${seed}-generation.json" ]]; then
        echo "Missing reference generation result for seed ${seed}" >&2
        exit 3
    fi
done
for path in "${TRAIN_ENV}/bin/python"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
for path in "${OUTPUT}" "${LOG_FILE}"; do
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
"${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2e/aggregate_multiseed.py" \
    --generation-dir "${LOG_DIR}" \
    --reference-dir "${LOG_DIR}" \
    --output "${OUTPUT}" \
    2>&1 | tee "${LOG_FILE}"
echo SUCCESS > "${STATUS_FILE}"
trap - ERR
