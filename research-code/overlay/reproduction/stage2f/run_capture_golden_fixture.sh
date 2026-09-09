#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/stage2f-fb-maq}
FIXTURE_DIR=${FIXTURE_DIR:-${MEDIC_AD_ROOT}/reproduction/stage2f/fixtures/pre-change}
LOG_FILE=${LOG_DIR}/golden-fixture.log
STATUS_FILE=${LOG_DIR}/golden-fixture.status

mkdir -p "${LOG_DIR}"
for path in "${TRAIN_ENV}/bin/python" "${MEDIC_AD_ROOT}/reproduction/stage2f/capture_golden_fixture.py"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
if [[ -e "${LOG_FILE}" ]] || [[ -e "${STATUS_FILE}" ]]; then
    echo "Refusing to overwrite golden-fixture evidence in ${LOG_DIR}" >&2
    exit 4
fi

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

cd "${MEDIC_AD_ROOT}"
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
set +e
"${TRAIN_ENV}/bin/python" reproduction/stage2f/capture_golden_fixture.py \
    --repo-root "${MEDIC_AD_ROOT}" \
    --output-dir "${FIXTURE_DIR}" \
    2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi
exit "${EXIT_CODE}"
