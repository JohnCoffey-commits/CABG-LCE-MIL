"""Atomic, fail-closed full-state checkpoint for the two-phase D4 pilot."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, tensor_sha256
from reproduction.stage2i.rng_state import RNG_KEYS, rng_state_fingerprint
from reproduction.stage2l.constants import CHECKPOINT_SCHEMA_VERSION, TRAINABLE_NAMES

PAYLOAD_KEYS = {
    "schema_version", "adapter_state", "master_state", "optimizer_state", "scheduler_state",
    "controller_state", "sampler_state", "rng_state", "provenance", "trace_state",
}


def _walk(value: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _walk(value[key], f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, member in enumerate(value):
            yield from _walk(member, f"{prefix}[{index}]")


def tensor_inventory(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    return [{
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "elements": value.numel(),
        "sha256": tensor_sha256(value),
    } for name, value in _walk(payload)]


def validate(payload: Mapping[str, Any]) -> None:
    if set(payload) != PAYLOAD_KEYS or payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CABGContractError("D4 checkpoint payload schema/version mismatch.")
    adapter, master = payload["adapter_state"], payload["master_state"]
    if tuple(adapter) != TRAINABLE_NAMES or tuple(master) != TRAINABLE_NAMES:
        raise CABGContractError("D4 checkpoint trainable tensor order changed.")
    for name in TRAINABLE_NAMES:
        if adapter[name].dtype != torch.bfloat16 or master[name].dtype != torch.float32:
            raise CABGContractError(f"D4 checkpoint precision mismatch: {name}")
        if adapter[name].shape != master[name].shape:
            raise CABGContractError(f"D4 checkpoint shape mismatch: {name}")
        if not torch.isfinite(master[name]).all():
            raise CABGContractError(f"D4 checkpoint non-finite master: {name}")
    controller = payload["controller_state"]
    if set(controller) != {"beta", "lm_ema", "lce_ema", "valid_blocks"}:
        raise CABGContractError("D4 checkpoint controller schema mismatch.")
    sampler = payload["sampler_state"]
    sampler_keys = set(sampler)
    base_sampler_keys = {"cursor", "manifest_sha256", "order_sha256"}
    if sampler_keys not in (base_sampler_keys, base_sampler_keys | {"repair_state"}):
        raise CABGContractError("D4 checkpoint sampler schema mismatch.")
    if "repair_state" in sampler and not isinstance(sampler["repair_state"], Mapping):
        raise CABGContractError("D4 checkpoint repair state is invalid.")
    trace = payload["trace_state"]
    if set(trace) != {"completed_blocks", "last_record_sha256"}:
        raise CABGContractError("D4 checkpoint trace schema mismatch.")
    if int(sampler["cursor"]) != int(trace["completed_blocks"]):
        raise CABGContractError("D4 checkpoint sampler/trace cursor mismatch.")
    if set(payload["rng_state"]) != RNG_KEYS:
        raise CABGContractError("D4 checkpoint RNG schema mismatch.")
    if not isinstance(payload["optimizer_state"], Mapping) or not isinstance(payload["scheduler_state"], Mapping):
        raise CABGContractError("D4 checkpoint optimizer/scheduler state missing.")
    if not isinstance(payload["provenance"], Mapping) or not payload["provenance"]:
        raise CABGContractError("D4 checkpoint provenance missing.")


def save_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, object]:
    validate(payload)
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        inventory = tensor_inventory(payload)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_file": path.name,
            "checkpoint_bytes": temporary.stat().st_size,
            "checkpoint_sha256": sha256_file(temporary),
            "tensor_inventory": inventory,
            "tensor_inventory_sha256": canonical_json_sha256(inventory),
            "rng_fingerprint": rng_state_fingerprint(payload["rng_state"]),
            "provenance_sha256": canonical_json_sha256(payload["provenance"]),
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with temporary_manifest.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def load_strict(path: Path, expected_provenance: Mapping[str, object]) -> tuple[dict[str, Any], dict[str, object]]:
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CABGContractError("D4 checkpoint manifest version mismatch.")
    if manifest.get("checkpoint_file") != path.name or manifest.get("checkpoint_bytes") != path.stat().st_size:
        raise CABGContractError("D4 checkpoint file identity mismatch.")
    if manifest.get("checkpoint_sha256") != sha256_file(path):
        raise CABGContractError("D4 checkpoint file hash mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate(payload)
    inventory = tensor_inventory(payload)
    if inventory != manifest.get("tensor_inventory") or canonical_json_sha256(inventory) != manifest.get("tensor_inventory_sha256"):
        raise CABGContractError("D4 checkpoint tensor inventory mismatch.")
    if rng_state_fingerprint(payload["rng_state"]) != manifest.get("rng_fingerprint"):
        raise CABGContractError("D4 checkpoint RNG fingerprint mismatch.")
    if dict(payload["provenance"]) != dict(expected_provenance):
        raise CABGContractError("D4 checkpoint provenance mismatch.")
    return dict(payload), manifest
