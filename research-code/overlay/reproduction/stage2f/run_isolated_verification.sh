#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_ROOT=${LOG_ROOT:-/home/logs/medic-ad/stage2f-fb-maq/isolated-verification}
FIXTURE_DIR=${FIXTURE_DIR:-${MEDIC_AD_ROOT}/reproduction/stage2f/fixtures/pre-change}
STATUS_FILE=${LOG_ROOT}/isolated-verification.status
LOG_FILE=${LOG_ROOT}/isolated-verification.log

mkdir -p "${LOG_ROOT}"
if [[ -e "${STATUS_FILE}" ]] || [[ -e "${LOG_FILE}" ]]; then
    echo "Refusing to overwrite isolated-verification evidence: ${LOG_ROOT}" >&2
    exit 4
fi
for path in \
    "${TRAIN_ENV}/bin/python" \
    "${FIXTURE_DIR}/golden_manifest.json" \
    "${FIXTURE_DIR}/golden_tensors.safetensors" \
    "${MEDIC_AD_ROOT}/reproduction/stage2f/tests/test_fb_maq_component.py" \
    "${MEDIC_AD_ROOT}/reproduction/stage2f/tests/test_adapter_schema.py"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

cd "${MEDIC_AD_ROOT}"
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
set +e
{
    "${TRAIN_ENV}/bin/python" reproduction/stage2f/tests/test_fb_maq_component.py \
        --fixture-dir "${FIXTURE_DIR}" \
        --output "${LOG_ROOT}/component-test.json"
    "${TRAIN_ENV}/bin/python" reproduction/stage2f/tests/test_adapter_schema.py \
        --work-dir "${LOG_ROOT}/adapter-schema-work" \
        --output "${LOG_ROOT}/adapter-schema-test.json"
} 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo SUCCESS > "${STATUS_FILE}"
else
    echo FAILED > "${STATUS_FILE}"
fi
exit "${EXIT_CODE}"
