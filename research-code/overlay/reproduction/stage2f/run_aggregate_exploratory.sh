#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_ROOT=${LOG_ROOT:-/home/logs/medic-ad/stage2f-fb-maq}
EXPLORATORY_ROOT=${EXPLORATORY_ROOT:-${LOG_ROOT}/exploratory}
ORCHESTRATOR_ROOT=${ORCHESTRATOR_ROOT:-${LOG_ROOT}/exploratory-orchestrator}
AGGREGATE_ROOT=${AGGREGATE_ROOT:-${LOG_ROOT}/exploratory-aggregate}
STATUS_FILE=${AGGREGATE_ROOT}/aggregate.status
LOG_FILE=${AGGREGATE_ROOT}/aggregate.log
RESULT=${AGGREGATE_ROOT}/aggregate.json
TRANSITIONS=${AGGREGATE_ROOT}/paired-transitions.jsonl
ENTRY=${MEDIC_AD_ROOT}/reproduction/stage2f/aggregate_exploratory.py

for path in \
    "${TRAIN_ENV}/bin/python" \
    "${ENTRY}" \
    "${ORCHESTRATOR_ROOT}/run-order.tsv" \
    "${ORCHESTRATOR_ROOT}/orchestrator.status"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
if [[ "$(<"${ORCHESTRATOR_ROOT}/orchestrator.status")" != "SUCCESS" ]]; then
    echo "Exploratory orchestrator did not complete successfully." >&2
    exit 4
fi
if [[ -e "${AGGREGATE_ROOT}" ]]; then
    echo "Refusing to overwrite exploratory aggregate evidence: ${AGGREGATE_ROOT}" >&2
    exit 5
fi

mkdir -p "${AGGREGATE_ROOT}"
echo RUNNING > "${STATUS_FILE}"
mark_failed() { echo FAILED > "${STATUS_FILE}"; }
trap mark_failed ERR
cd "${MEDIC_AD_ROOT}"
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
set +e
"${TRAIN_ENV}/bin/python" "${ENTRY}" \
    --log-root "${EXPLORATORY_ROOT}" \
    --run-order "${ORCHESTRATOR_ROOT}/run-order.tsv" \
    --output "${RESULT}" \
    --transitions "${TRANSITIONS}" \
    2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e
if [[ ${EXIT_CODE} -eq 0 ]]; then echo SUCCESS > "${STATUS_FILE}"; else echo FAILED > "${STATUS_FILE}"; fi
exit "${EXIT_CODE}"
