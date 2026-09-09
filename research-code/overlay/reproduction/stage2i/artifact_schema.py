"""Strict append-only CABG block-trace schema and hash chain."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Mapping

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import BLOCK_TRACE_SCHEMA_VERSION
from reproduction.stage2i.fingerprint import canonical_json_sha256


HEX64 = re.compile(r"^[0-9a-f]{64}$")
GENESIS_HASH = "0" * 64

BLOCK_TRACE_KEYS = {
    "schema_version",
    "status",
    "run_id",
    "method",
    "seed",
    "epoch",
    "block_id",
    "ordered_samples",
    "class_counts",
    "losses",
    "per_image_shared_norms",
    "ema",
    "budget",
    "aggregate_shared",
    "gradient_norms",
    "optimizer",
    "scheduler",
    "evidence",
    "memory",
    "wall_time_seconds",
    "rng",
    "provenance",
    "checkpoint_sha256",
    "previous_record_sha256",
    "record_sha256",
}

NESTED_KEYS = {
    "class_counts": {"normal", "abnormal"},
    "losses": {"lm_per_image", "auxiliary_per_image", "lm_mean", "auxiliary_class_balanced"},
    "ema": {"raw_lm", "raw_auxiliary", "corrected_lm", "corrected_auxiliary", "valid_blocks"},
    "budget": {"lambda_raw", "lambda_cap", "lambda_final", "rho", "rho_max"},
    "aggregate_shared": {"lm_norm", "auxiliary_norm", "scaled_auxiliary_norm", "cosine", "cap_utilization"},
    "gradient_norms": {"full_preclip", "full_postclip", "clip_scale", "cap_checked_before_clip"},
    "optimizer": {"step", "type"},
    "scheduler": {"step", "learning_rate"},
    "evidence": {
        "score_min",
        "score_max",
        "score_mean",
        "auxiliary_loss_min",
        "auxiliary_loss_max",
        "auxiliary_loss_mean",
        "evidence_min",
        "evidence_max",
        "evidence_mean",
        "evidence_saturation_fraction",
        "abnormal_map_min",
        "abnormal_map_max",
        "abnormal_map_mean",
        "abnormal_map_saturation_fraction",
        "normal_map_min",
        "normal_map_max",
        "normal_map_mean",
        "normal_map_saturation_fraction",
        "spatial_entropy_mean",
        "effective_support_mean",
        "top11_mass_mean",
        "max_pooling_weight",
        "min_nonzero_pooling_weight",
    },
    "memory": {"cuda_allocated_mib", "cuda_reserved_mib", "external_peak_mib"},
    "rng": {"before_block", "after_block"},
    "provenance": {"source_commit", "implementation_fingerprint", "dataset_manifest_sha256", "support_fingerprint"},
}


def _require_exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise CABGContractError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _validate_finite(value: object, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise CABGContractError(f"{context} contains NaN or Inf.")
        return
    if isinstance(value, Mapping):
        for key, member in value.items():
            _validate_finite(member, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, member in enumerate(value):
            _validate_finite(member, f"{context}[{index}]")
        return
    raise CABGContractError(f"{context} contains unsupported value type {type(value).__name__}.")


def block_record_sha256(record: Mapping[str, object]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return canonical_json_sha256(payload)


def validate_block_record(record: Mapping[str, object]) -> None:
    _require_exact_keys(record, BLOCK_TRACE_KEYS, "CABG block trace")
    if record["schema_version"] != BLOCK_TRACE_SCHEMA_VERSION or record["status"] != "SUCCESS":
        raise CABGContractError("CABG block-trace status/version mismatch.")
    samples = record["ordered_samples"]
    if not isinstance(samples, list) or not samples:
        raise CABGContractError("CABG block trace has no ordered samples.")
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise CABGContractError("CABG sample record must be an object.")
        _require_exact_keys(sample, {"sample_id", "sha256", "label"}, f"ordered_samples[{index}]")
        normalize_label(sample["label"])
        if not HEX64.fullmatch(str(sample["sha256"])):
            raise CABGContractError("CABG sample SHA-256 is invalid.")
    for key, expected in NESTED_KEYS.items():
        value = record[key]
        if not isinstance(value, Mapping):
            raise CABGContractError(f"CABG {key} must be an object.")
        _require_exact_keys(value, expected, key)
    counts = record["class_counts"]
    if int(counts["normal"]) <= 0 or int(counts["abnormal"]) <= 0:
        raise CABGContractError("CABG block trace must contain both classes.")
    if int(counts["normal"]) + int(counts["abnormal"]) != len(samples):
        raise CABGContractError("CABG block sample/class counts differ.")
    losses = record["losses"]
    norms = record["per_image_shared_norms"]
    if not isinstance(norms, list) or len(norms) != len(samples):
        raise CABGContractError("CABG per-image norm records differ from sample count.")
    if len(losses["lm_per_image"]) != len(samples) or len(losses["auxiliary_per_image"]) != len(samples):
        raise CABGContractError("CABG per-image losses differ from sample count.")
    for index, value in enumerate(norms):
        if not isinstance(value, Mapping):
            raise CABGContractError("CABG per-image norm must be an object.")
        _require_exact_keys(value, {"sample_id", "label", "lm", "auxiliary"}, f"norms[{index}]")
    budget = record["budget"]
    if float(budget["lambda_final"]) > min(float(budget["lambda_raw"]), float(budget["lambda_cap"])) + 1e-12:
        raise CABGContractError("CABG lambda_final exceeds its locked raw/cap rule.")
    if record["gradient_norms"]["cap_checked_before_clip"] is not True:
        raise CABGContractError("CABG trust cap was not checked before clipping.")
    for hash_key in (
        "checkpoint_sha256",
        "previous_record_sha256",
        "record_sha256",
    ):
        if not HEX64.fullmatch(str(record[hash_key])):
            raise CABGContractError(f"CABG {hash_key} is invalid.")
    for hash_key in ("implementation_fingerprint", "dataset_manifest_sha256", "support_fingerprint"):
        if not HEX64.fullmatch(str(record["provenance"][hash_key])):
            raise CABGContractError(f"CABG provenance {hash_key} is invalid.")
    if not re.fullmatch(r"[0-9a-f]{40}", str(record["provenance"]["source_commit"])):
        raise CABGContractError("CABG provenance source_commit is invalid.")
    for value in record["rng"].values():
        if not HEX64.fullmatch(str(value)):
            raise CABGContractError("CABG RNG fingerprint is invalid.")
    _validate_finite(record, "CABG block trace")
    if str(record["record_sha256"]) != block_record_sha256(record):
        raise CABGContractError("CABG block record hash mismatch.")


def finalize_block_record(record: Mapping[str, object]) -> dict[str, object]:
    if "record_sha256" in record:
        raise CABGContractError("record_sha256 must be assigned only by finalize_block_record().")
    result = dict(record)
    result["record_sha256"] = block_record_sha256(result)
    validate_block_record(result)
    return result


def append_block_record(path: Path, record: Mapping[str, object]) -> None:
    validate_block_record(record)
    previous = GENESIS_HASH
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise CABGContractError("Existing CABG block-trace file is empty.")
        last = json.loads(lines[-1])
        validate_block_record(last)
        previous = str(last["record_sha256"])
    if str(record["previous_record_sha256"]) != previous:
        raise CABGContractError("CABG block-trace previous-record hash mismatch.")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
