"""Atomic strict checkpoint schema for CABG block-boundary state."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.constants import CHECKPOINT_SCHEMA_VERSION
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, tensor_sha256
from reproduction.stage2i.rng_state import RNG_KEYS, rng_state_fingerprint


PAYLOAD_KEYS = {
    "schema_version",
    "adapter_state",
    "master_state",
    "optimizer_state",
    "scheduler_state",
    "cabg_state",
    "sampler_state",
    "rng_state",
    "provenance",
    "trace_state",
}
MANIFEST_KEYS = {
    "schema_version",
    "checkpoint_file",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "tensor_inventory",
    "tensor_inventory_sha256",
    "rng_fingerprint",
    "provenance_sha256",
}
PROVENANCE_KEYS = {
    "source_commit",
    "implementation_fingerprint",
    "base_checkpoint_fingerprint",
    "dataset_manifest_sha256",
    "configuration_sha256",
}
CABG_STATE_KEYS = {
    "m_lm",
    "m_auxiliary",
    "valid_blocks",
    "last_lambda_raw",
    "last_lambda_cap",
    "last_lambda_final",
}
TRACE_STATE_KEYS = {"last_record_sha256", "completed_blocks"}
SAMPLER_STATE_KEYS = {"seed", "epoch", "cursor", "block_permutation_hash"}


def _walk_tensors(value: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_tensors(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            yield from _walk_tensors(child_value, f"{prefix}[{index}]")


def tensor_inventory(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    rows = []
    for name, value in _walk_tensors(payload):
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "elements": int(value.numel()),
                "sha256": tensor_sha256(value),
            }
        )
    rows.sort(key=lambda row: row["name"])
    return rows


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != PAYLOAD_KEYS:
        raise CABGContractError("CABG checkpoint payload schema mismatch.")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CABGContractError("CABG checkpoint schema version mismatch.")
    if not isinstance(payload["adapter_state"], Mapping) or not payload["adapter_state"]:
        raise CABGContractError("CABG adapter state is empty.")
    if set(payload["adapter_state"]) != set(payload["master_state"]):
        raise CABGContractError("CABG adapter/master tensor names differ.")
    for name in payload["adapter_state"]:
        adapter = payload["adapter_state"][name]
        master = payload["master_state"][name]
        if adapter.dtype != torch.bfloat16 or master.dtype != torch.float32:
            raise CABGContractError(f"CABG adapter/master dtype mismatch for {name}.")
        if adapter.shape != master.shape:
            raise CABGContractError(f"CABG adapter/master shape mismatch for {name}.")
    if set(payload["cabg_state"]) != CABG_STATE_KEYS:
        raise CABGContractError("CABG EMA/budget state schema mismatch.")
    cabg = payload["cabg_state"]
    numeric_cabg = [float(cabg[key]) for key in CABG_STATE_KEYS if key != "valid_blocks"]
    if (
        not all(map(math.isfinite, numeric_cabg))
        or int(cabg["valid_blocks"]) < 0
        or float(cabg["last_lambda_final"])
        > min(float(cabg["last_lambda_raw"]), float(cabg["last_lambda_cap"])) + 1e-12
    ):
        raise CABGContractError("CABG EMA/budget state values are invalid.")
    if not isinstance(payload["optimizer_state"], Mapping) or not isinstance(
        payload["scheduler_state"], Mapping
    ):
        raise CABGContractError("CABG optimizer/scheduler states must be objects.")
    if set(payload["sampler_state"]) != SAMPLER_STATE_KEYS:
        raise CABGContractError("CABG block-sampler state schema mismatch.")
    sampler = payload["sampler_state"]
    if int(sampler["cursor"]) < 0 or not re.fullmatch(
        r"[0-9a-f]{64}", str(sampler["block_permutation_hash"])
    ):
        raise CABGContractError("CABG block-sampler state values are invalid.")
    if set(payload["rng_state"]) != RNG_KEYS:
        raise CABGContractError("CABG checkpoint RNG state schema mismatch.")
    if set(payload["provenance"]) != PROVENANCE_KEYS:
        raise CABGContractError("CABG checkpoint provenance schema mismatch.")
    if set(payload["trace_state"]) != TRACE_STATE_KEYS:
        raise CABGContractError("CABG checkpoint trace-state schema mismatch.")
    if int(payload["trace_state"]["completed_blocks"]) < 0:
        raise CABGContractError("CABG completed-block count is invalid.")
    if int(payload["trace_state"]["completed_blocks"]) != int(sampler["cursor"]):
        raise CABGContractError("CABG trace/sampler completed-block cursors differ.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload["trace_state"]["last_record_sha256"])):
        raise CABGContractError("CABG trace-state hash is invalid.")
    provenance = payload["provenance"]
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance["source_commit"])):
        raise CABGContractError("CABG provenance source commit is invalid.")
    for key in PROVENANCE_KEYS - {"source_commit"}:
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance[key])):
            raise CABGContractError(f"CABG provenance hash is invalid: {key}")


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    validate_checkpoint_payload(payload)
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite CABG checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_path = path.with_name(f".{path.name}.{token}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        inventory = tensor_inventory(payload)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_file": path.name,
            "checkpoint_bytes": temporary_path.stat().st_size,
            "checkpoint_sha256": sha256_file(temporary_path),
            "tensor_inventory": inventory,
            "tensor_inventory_sha256": canonical_json_sha256(inventory),
            "rng_fingerprint": rng_state_fingerprint(payload["rng_state"]),
            "provenance_sha256": canonical_json_sha256(payload["provenance"]),
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with temporary_manifest.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return manifest_path


def load_checkpoint_strict(
    path: Path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != MANIFEST_KEYS or manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CABGContractError("CABG checkpoint manifest schema mismatch.")
    if manifest["checkpoint_file"] != path.name:
        raise CABGContractError("CABG checkpoint filename mismatch.")
    if int(manifest["checkpoint_bytes"]) != path.stat().st_size:
        raise CABGContractError("CABG checkpoint byte count mismatch.")
    if manifest["checkpoint_sha256"] != sha256_file(path):
        raise CABGContractError("CABG checkpoint file hash mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CABGContractError("CABG checkpoint payload is not an object.")
    validate_checkpoint_payload(payload)
    inventory = tensor_inventory(payload)
    if inventory != manifest["tensor_inventory"]:
        raise CABGContractError("CABG checkpoint tensor inventory mismatch.")
    if canonical_json_sha256(inventory) != manifest["tensor_inventory_sha256"]:
        raise CABGContractError("CABG checkpoint tensor-inventory hash mismatch.")
    if rng_state_fingerprint(payload["rng_state"]) != manifest["rng_fingerprint"]:
        raise CABGContractError("CABG checkpoint RNG fingerprint mismatch.")
    if canonical_json_sha256(payload["provenance"]) != manifest["provenance_sha256"]:
        raise CABGContractError("CABG checkpoint provenance hash mismatch.")
    if expected_provenance is not None and dict(payload["provenance"]) != dict(expected_provenance):
        raise CABGContractError("CABG checkpoint provenance differs from the expected run.")
    return dict(payload)
