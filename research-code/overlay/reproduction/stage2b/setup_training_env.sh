#!/usr/bin/env bash

set -euo pipefail

CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
SOURCE_ENV=${SOURCE_ENV:-/home/envs/medic-ad}
TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/training-smoke}
LOG_FILE=${LOG_DIR}/training-env-setup.log
STATUS_FILE=${LOG_DIR}/training-env-setup.status

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "Conda executable not found: ${CONDA_BIN}" >&2
    exit 2
fi
if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
    echo "Source environment is invalid: ${SOURCE_ENV}" >&2
    exit 3
fi
if [[ -e "${TRAIN_ENV}" ]]; then
    echo "Refusing to overwrite existing training environment: ${TRAIN_ENV}" >&2
    exit 4
fi

"${CONDA_BIN}" create \
    --yes \
    --override-channels \
    --channel conda-forge \
    --prefix "${TRAIN_ENV}" \
    --clone "${SOURCE_ENV}"

export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/home/cache/pip}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export DS_BUILD_OPS=0
"${TRAIN_ENV}/bin/python" -m pip install deepspeed==0.16.4

"${TRAIN_ENV}/bin/python" - <<'PY'
import deepspeed
import flash_attn
import torch
import tokenizers

print("python_environment=", __import__("sys").executable)
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("deepspeed=", deepspeed.__version__)
print("flash_attn=", flash_attn.__version__)
print("tokenizers=", tokenizers.__version__)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the cloned training environment.")
tensor = torch.ones(32, device="cuda", dtype=torch.bfloat16)
print("bf16_cuda_sum=", tensor.sum().item())
PY

"${TRAIN_ENV}/bin/python" -m pip check
"${TRAIN_ENV}/bin/python" -m pip freeze > "${LOG_DIR}/training-env-pip-freeze.txt"

echo SUCCESS > "${STATUS_FILE}"
trap - ERR
