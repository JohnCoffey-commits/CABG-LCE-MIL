#!/usr/bin/env bash

set -euo pipefail

TRAIN_ENV=${TRAIN_ENV:-/home/envs/medic-ad-train}
MODEL_ID=${MODEL_ID:-lingshu-medical-mllm/Lingshu-7B}
REVISION=${REVISION:-b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9}
TARGET_DIR=${TARGET_DIR:-/home/checkpoints/Lingshu-7B}
LOG_DIR=${LOG_DIR:-/home/logs/medic-ad/training-smoke}
LOG_FILE=${LOG_DIR}/lingshu-download.log
STATUS_FILE=${LOG_DIR}/lingshu-download.status
REVISION_FILE=${TARGET_DIR}/.medic-ad-pinned-revision

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo RUNNING > "${STATUS_FILE}"
mark_failed() {
    echo FAILED > "${STATUS_FILE}"
}
trap mark_failed ERR

if [[ ! -x "${TRAIN_ENV}/bin/python" ]]; then
    echo "Training environment is invalid: ${TRAIN_ENV}" >&2
    exit 2
fi

if [[ -d "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ ! -f "${REVISION_FILE}" ]] || [[ "$(<"${REVISION_FILE}")" != "${REVISION}" ]]; then
        echo "Refusing to reuse a non-empty checkpoint directory without the expected revision marker: ${TARGET_DIR}" >&2
        exit 3
    fi
else
    mkdir -p "${TARGET_DIR}"
    printf '%s\n' "${REVISION}" > "${REVISION_FILE}"
fi

export HF_HOME=${HF_HOME:-/home/cache/huggingface}
export HF_HUB_DISABLE_TELEMETRY=1
export MODEL_ID REVISION TARGET_DIR LOG_DIR

"${TRAIN_ENV}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["REVISION"],
    local_dir=os.environ["TARGET_DIR"],
)
print("snapshot_path=", path)
PY

"${TRAIN_ENV}/bin/python" - <<'PY'
import hashlib
import json
import os
import struct
from pathlib import Path

target = Path(os.environ["TARGET_DIR"])
revision = os.environ["REVISION"]
index_path = target / "model.safetensors.index.json"
if not index_path.is_file():
    raise FileNotFoundError(index_path)

index = json.loads(index_path.read_text())
expected_total = 16_584_333_312
actual_total = index.get("metadata", {}).get("total_size")
if actual_total != expected_total:
    raise RuntimeError(f"Unexpected model total_size: {actual_total}; expected {expected_total}")

shards = sorted(set(index.get("weight_map", {}).values()))
if len(shards) != 4:
    raise RuntimeError(f"Expected 4 weight shards, found {len(shards)}: {shards}")

manifest = []
observed_tensor_bytes = 0
for relative_name in shards:
    path = target / relative_name
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        if len(header_length_raw) != 8:
            raise RuntimeError(f"Invalid safetensors header length: {path}")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        header = json.loads(handle.read(header_length))
    tensor_entries = [value for key, value in header.items() if key != "__metadata__"]
    offsets = sorted((entry["data_offsets"][0], entry["data_offsets"][1]) for entry in tensor_entries)
    previous_end = 0
    for start, end in offsets:
        if start < previous_end or end < start:
            raise RuntimeError(f"Invalid or overlapping tensor offsets in {path}: {(start, end)}")
        previous_end = end
    expected_file_size = 8 + header_length + (offsets[-1][1] if offsets else 0)
    if path.stat().st_size != expected_file_size:
        raise RuntimeError(
            f"Safetensors layout mismatch for {path}: file={path.stat().st_size}, expected={expected_file_size}"
        )
    shard_tensor_bytes = sum(end - start for start, end in offsets)
    observed_tensor_bytes += shard_tensor_bytes

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest.append({
        "file": relative_name,
        "bytes": path.stat().st_size,
        "tensor_bytes": shard_tensor_bytes,
        "sha256": digest.hexdigest(),
    })

observed_physical_bytes = sum(item["bytes"] for item in manifest)
if observed_tensor_bytes != expected_total:
    raise RuntimeError(f"Tensor payload bytes sum to {observed_tensor_bytes}; expected {expected_total}")

result = {
    "model_id": os.environ["MODEL_ID"],
    "revision": revision,
    "index_total_size": actual_total,
    "observed_tensor_bytes": observed_tensor_bytes,
    "observed_physical_bytes": observed_physical_bytes,
    "safetensors_header_bytes": observed_physical_bytes - observed_tensor_bytes,
    "shards": manifest,
}
output = Path(os.environ["LOG_DIR"]) / "lingshu-integrity.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

find "${TARGET_DIR}" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "${LOG_DIR}/lingshu-file-sizes.tsv"
printf '%s\n' "${MODEL_ID}@${REVISION}" > "${LOG_DIR}/lingshu-pinned-revision.txt"

echo SUCCESS > "${STATUS_FILE}"
trap - ERR
