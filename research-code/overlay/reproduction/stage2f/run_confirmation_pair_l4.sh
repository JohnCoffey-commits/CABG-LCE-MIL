#!/usr/bin/env bash

set -Eeuo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
GROUP=stage2f-confirmation-v1
LOG_BASE=/home/logs/medic-ad/stage2f-fb-maq/${GROUP}
ORCHESTRATOR_ROOT=${MEDIC_AD_CONFIRMATION_ORCHESTRATOR_ROOT:-/home/logs/medic-ad/stage2f-fb-maq/${GROUP}-orchestrator}
START_MODE=${MEDIC_AD_CONFIRMATION_START_MODE:-a3}
PAIR_STATUS=${LOG_BASE}/pair.status
PAIR_RESULT=${LOG_BASE}/pair-verification.json
ORCHESTRATOR_STATUS=${ORCHESTRATOR_ROOT}/orchestrator.status
ORCHESTRATOR_LOG=${ORCHESTRATOR_ROOT}/orchestrator.log
RUNNER=${MEDIC_AD_ROOT}/reproduction/stage2f/run_exploratory_mode_l4.sh
PAIR_VERIFIER=${MEDIC_AD_ROOT}/reproduction/stage2f/verify_confirmation_pair.py
ELIGIBILITY=/home/logs/medic-ad/stage2f-fb-maq/exploratory-aggregate-v3-final/aggregate.json

for path in "${TRAIN_ENV}/bin/python" "${RUNNER}" "${PAIR_VERIFIER}" "${ELIGIBILITY}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing confirmation requirement: ${path}" >&2
        exit 3
    fi
done
case "${START_MODE}" in
    a3)
        if [[ -e "${LOG_BASE}" ]] || [[ -e "${ORCHESTRATOR_ROOT}" ]]; then
            echo "Refusing to overwrite Stage 2F-C confirmation evidence." >&2
            exit 4
        fi
        ;;
    b0)
        if [[ ! -d "${LOG_BASE}" ]] || [[ -e "${ORCHESTRATOR_ROOT}" ]]; then
            echo "Stage 2F-C B0 resume requires existing A3 evidence and a new orchestrator path." >&2
            exit 4
        fi
        if [[ ! -f "${LOG_BASE}/confirmation-seed123-repeat1-a3/run.status" ]] || \
           [[ "$(<"${LOG_BASE}/confirmation-seed123-repeat1-a3/run.status")" != "SUCCESS" ]]; then
            echo "Stage 2F-C B0 resume requires a verified A3 SUCCESS." >&2
            exit 4
        fi
        if [[ -e "${LOG_BASE}/confirmation-seed123-repeat1-b0" ]] || [[ -e "${PAIR_RESULT}" ]]; then
            echo "Refusing to overwrite B0 or pair-level confirmation evidence." >&2
            exit 4
        fi
        ;;
    *)
        echo "MEDIC_AD_CONFIRMATION_START_MODE must be a3 or b0." >&2
        exit 4
        ;;
esac
if ! grep -q '"overall_vqa_classification": "negative"' "${ELIGIBILITY}"; then
    echo "Stage 2F-C requires the verified negative Stage 2F-B result." >&2
    exit 5
fi

mkdir -p "${LOG_BASE}" "${ORCHESTRATOR_ROOT}"
echo RUNNING > "${PAIR_STATUS}"
echo RUNNING > "${ORCHESTRATOR_STATUS}"
mark_failed() {
    echo FAILED > "${PAIR_STATUS}"
    echo FAILED > "${ORCHESTRATOR_STATUS}"
}
trap mark_failed ERR
finalize_status() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then mark_failed; fi
}
trap finalize_status EXIT
exec > >(tee "${ORCHESTRATOR_LOG}") 2>&1

run_mode() {
    local mode=$1
    local method_id=$2
    local run_id=confirmation-seed123-repeat1-${method_id}
    printf '[Stage2F-C] START mode=%s run_id=%s\n' "${mode}" "${run_id}"
    MEDIC_AD_QUERY_MODE="${mode}" \
    MEDIC_AD_SEED=123 \
    MEDIC_AD_REPEAT=1 \
    MEDIC_AD_RUN_ID_OVERRIDE="${run_id}" \
    MEDIC_AD_EXPLORATORY_GROUP_OVERRIDE="${GROUP}" \
    MEDIC_AD_GRADIENT_AUDIT_INTERVAL=1 \
    MEDIC_AD_ENABLE_ACTIVATION_DIAGNOSTICS=1 \
        bash "${RUNNER}"
    printf '[Stage2F-C] SUCCESS mode=%s run_id=%s\n' "${mode}" "${run_id}"
}

if [[ "${START_MODE}" == "a3" ]]; then
    run_mode multiscale a3
else
    echo '[Stage2F-C] SKIP_VERIFIED mode=multiscale run_id=confirmation-seed123-repeat1-a3'
fi
run_mode single b0

cd "${MEDIC_AD_ROOT}"
export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
"${TRAIN_ENV}/bin/python" "${PAIR_VERIFIER}" \
    --a3-root "${LOG_BASE}/confirmation-seed123-repeat1-a3" \
    --b0-root "${LOG_BASE}/confirmation-seed123-repeat1-b0" \
    --output "${PAIR_RESULT}"

echo SUCCESS > "${PAIR_STATUS}"
echo SUCCESS > "${ORCHESTRATOR_STATUS}"
