#!/usr/bin/env bash

set -Eeuo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
ORCHESTRATOR_ROOT=${ORCHESTRATOR_ROOT:-/home/logs/medic-ad/stage2f-fb-maq/exploratory-orchestrator}
ENGINEERING_GATE=${ENGINEERING_GATE:-/home/logs/medic-ad/stage2f-fb-maq/engineering/engineering-gate-v2.json}
RUNNER=${MEDIC_AD_ROOT}/reproduction/stage2f/run_exploratory_mode_l4.sh
STATUS_FILE=${ORCHESTRATOR_ROOT}/orchestrator.status
LOG_FILE=${ORCHESTRATOR_ROOT}/orchestrator.log
ORDER_FILE=${ORCHESTRATOR_ROOT}/run-order.tsv
START_POSITION=${MEDIC_AD_START_POSITION:-1}
MEDIC_AD_EXPLORATORY_EVIDENCE_TAG=${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG:-}
MEDIC_AD_EXPLORATORY_GROUP=exploratory
if [[ -n "${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG}" ]]; then
    MEDIC_AD_EXPLORATORY_GROUP=${MEDIC_AD_EXPLORATORY_GROUP}-${MEDIC_AD_EXPLORATORY_EVIDENCE_TAG}
fi

for path in "${RUNNER}" "${ENGINEERING_GATE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 3
    fi
done
if [[ -e "${ORCHESTRATOR_ROOT}" ]]; then
    echo "Refusing to overwrite exploratory orchestrator evidence: ${ORCHESTRATOR_ROOT}" >&2
    exit 4
fi
if ! [[ "${START_POSITION}" =~ ^([1-9]|1[0-2])$ ]]; then
    echo "MEDIC_AD_START_POSITION must be in 1..12." >&2
    exit 4
fi
if ! grep -q '"engineering_gate": "PASS"' "${ENGINEERING_GATE}"; then
    echo "Exploratory runs require a verified Engineering Gate PASS." >&2
    exit 5
fi

mkdir -p "${ORCHESTRATOR_ROOT}"
echo RUNNING > "${STATUS_FILE}"
mark_failed() { echo FAILED > "${STATUS_FILE}"; }
trap mark_failed ERR
finalize_status() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then echo FAILED > "${STATUS_FILE}"; fi
}
trap finalize_status EXIT
printf '%s\n' \
    $'position\tseed\trepeat\tmode\tmethod' \
    $'1\t42\t1\tsingle\tB0' \
    $'2\t42\t1\tmultiscale\tA3' \
    $'3\t42\t2\tmultiscale\tA3' \
    $'4\t42\t2\tsingle\tB0' \
    $'5\t123\t1\tmultiscale\tA3' \
    $'6\t123\t1\tsingle\tB0' \
    $'7\t123\t2\tsingle\tB0' \
    $'8\t123\t2\tmultiscale\tA3' \
    $'9\t2026\t1\tsingle\tB0' \
    $'10\t2026\t1\tmultiscale\tA3' \
    $'11\t2026\t2\tmultiscale\tA3' \
    $'12\t2026\t2\tsingle\tB0' > "${ORDER_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

run_one() {
    local position=$1
    local seed=$2
    local repeat=$3
    local mode=$4
    local method
    if [[ "${mode}" == "single" ]]; then method=B0; else method=A3; fi
    printf '[Stage2F] START position=%s seed=%s repeat=%s mode=%s method=%s\n' \
        "${position}" "${seed}" "${repeat}" "${mode}" "${method}"
    MEDIC_AD_QUERY_MODE="${mode}" MEDIC_AD_SEED="${seed}" MEDIC_AD_REPEAT="${repeat}" \
        bash "${RUNNER}"
    printf '[Stage2F] SUCCESS position=%s seed=%s repeat=%s mode=%s method=%s\n' \
        "${position}" "${seed}" "${repeat}" "${mode}" "${method}"
}

maybe_run() {
    local position=$1
    local seed=$2
    local repeat=$3
    local mode=$4
    local method_id
    if [[ "${mode}" == "single" ]]; then method_id=b0; else method_id=a3; fi
    if (( position < START_POSITION )); then
        local prior_status=/home/logs/medic-ad/stage2f-fb-maq/${MEDIC_AD_EXPLORATORY_GROUP}/seed-${seed}-repeat-${repeat}-${method_id}/run.status
        if [[ ! -f "${prior_status}" ]] || [[ "$(<"${prior_status}")" != "SUCCESS" ]]; then
            echo "Cannot resume past unverified run at position ${position}." >&2
            return 7
        fi
        printf '[Stage2F] SKIP_VERIFIED position=%s seed=%s repeat=%s mode=%s\n' \
            "${position}" "${seed}" "${repeat}" "${mode}"
        return
    fi
    run_one "${position}" "${seed}" "${repeat}" "${mode}"
}

maybe_run 1 42 1 single
maybe_run 2 42 1 multiscale
maybe_run 3 42 2 multiscale
maybe_run 4 42 2 single
maybe_run 5 123 1 multiscale
maybe_run 6 123 1 single
maybe_run 7 123 2 single
maybe_run 8 123 2 multiscale
maybe_run 9 2026 1 single
maybe_run 10 2026 1 multiscale
maybe_run 11 2026 2 multiscale
maybe_run 12 2026 2 single
echo SUCCESS > "${STATUS_FILE}"
