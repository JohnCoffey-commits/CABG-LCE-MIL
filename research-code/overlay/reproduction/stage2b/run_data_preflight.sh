#!/usr/bin/env bash

set -euo pipefail

MEDIC_AD_ROOT=${MEDIC_AD_ROOT:-/home/workspace/Medic-AD}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/training-smoke}
LOG_FILE=${LOG_DIR}/stage1-data-preflight.log
STATUS_FILE=${LOG_DIR}/stage1-data-preflight.status

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

export PYTHONPATH="${MEDIC_AD_ROOT}:${MEDIC_AD_ROOT}/qwen-vl-finetune"
export MEDIC_AD_LINGSHU_MODEL=${MEDIC_AD_LINGSHU_MODEL:-/home/checkpoints/Lingshu-7B}
export MEDIC_AD_SMOKE_ANNOTATION=${MEDIC_AD_SMOKE_ANNOTATION:-/home/data/medic-ad/training-smoke/stage1_smoke.json}
export MEDIC_AD_SMOKE_DATA_ROOT=${MEDIC_AD_SMOKE_DATA_ROOT:-/home/data/medic-ad/training-smoke/images}
export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export TOKENIZERS_PARALLELISM=false

"${TRAIN_ENV}/bin/python" \
    "${MEDIC_AD_ROOT}/qwen-vl-finetune/tools/preflight_stage1_smoke.py"

echo SUCCESS > "${STATUS_FILE}"
trap - ERR
