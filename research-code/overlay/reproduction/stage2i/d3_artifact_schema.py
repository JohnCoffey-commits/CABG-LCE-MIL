"""Strict artifacts for the Gate D3 read-only mechanism diagnostic."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label


D3_RECORD_SCHEMA = "cabg-v1.1-gate-d3-record-1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

RECORD_KEYS = {
    "schema_version",
    "status",
    "sample_id",
    "image_sha256",
    "relative_path",
    "label",
    "mechanism_id",
    "mechanism",
    "forward_id",
    "evidence_fingerprint",
    "losses",
    "score",
    "gradients",
    "evidence",
    "spatial",
    "memory",
    "runtime",
    "rng",
    "provenance",
}

GRADIENT_ROW_KEYS = {"name", "present", "finite", "max_abs", "l2_norm"}
EVIDENCE_KEYS = {
    "evidence_min",
    "evidence_max",
    "evidence_mean",
    "evidence_saturation_fraction",
    "abnormal_probability_min",
    "abnormal_probability_max",
    "abnormal_probability_mean",
    "abnormal_probability_saturation_fraction",
    "normal_probability_min",
    "normal_probability_max",
    "normal_probability_mean",
    "normal_probability_saturation_fraction",
    "reconstruction_exact",
    "reconstruction_max_abs",
}
SPATIAL_KEYS = {
    "entropy",
    "effective_support",
    "top11_mass",
    "max_pooling_weight",
    "min_nonzero_pooling_weight",
    "nonzero_pooling_weights",
    "exact_one_position_collapse",
}


def _exact(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise CABGContractError(
            f"D3 {context} schema mismatch: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _finite_tree(value: object, context: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise CABGContractError(f"D3 {context} contains NaN or Inf.")
        return
    if isinstance(value, Mapping):
        for key, member in value.items():
            _finite_tree(member, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, member in enumerate(value):
            _finite_tree(member, f"{context}[{index}]")
        return
    raise CABGContractError(f"D3 {context} has unsupported type {type(value).__name__}.")


def _validate_gradient_rows(rows: object, context: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise CABGContractError(f"D3 {context} must be a non-empty list.")
    names = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CABGContractError(f"D3 {context} row must be an object.")
        _exact(row, GRADIENT_ROW_KEYS, context)
        names.append(str(row["name"]))
        if row["present"] and (row["max_abs"] is None or row["l2_norm"] is None):
            raise CABGContractError(f"D3 present {context} gradient lacks numeric metrics.")
        if not row["present"] and (float(row["max_abs"]) != 0.0 or float(row["l2_norm"]) != 0.0):
            raise CABGContractError(f"D3 absent {context} gradient is not zero-valued.")
    if len(names) != len(set(names)):
        raise CABGContractError(f"D3 {context} contains duplicate names.")


def validate_mechanism_record(record: Mapping[str, object]) -> None:
    _exact(record, RECORD_KEYS, "mechanism record")
    if record["schema_version"] != D3_RECORD_SCHEMA or record["status"] != "SUCCESS":
        raise CABGContractError("D3 mechanism record status/schema mismatch.")
    normalize_label(record["label"])
    if record["mechanism_id"] not in {"M0", "M1", "M2"}:
        raise CABGContractError("D3 mechanism ID is invalid.")
    if not HEX64.fullmatch(str(record["image_sha256"])) or not HEX64.fullmatch(
        str(record["evidence_fingerprint"])
    ):
        raise CABGContractError("D3 image/evidence SHA-256 is invalid.")
    losses = record["losses"]
    gradients = record["gradients"]
    evidence = record["evidence"]
    memory = record["memory"]
    runtime = record["runtime"]
    rng = record["rng"]
    provenance = record["provenance"]
    for value, keys, name in (
        (losses, {"lm", "auxiliary"}, "losses"),
        (
            gradients,
            {
                "lm_shared_global_norm",
                "auxiliary_shared_global_norm",
                "cosine_defined",
                "cosine",
                "lm_shared_per_tensor",
                "auxiliary_all_trainable_per_tensor",
                "non_shared_leakage_count",
            },
            "gradients",
        ),
        (evidence, EVIDENCE_KEYS, "evidence"),
        (
            memory,
            {"image_peak_allocated_mib", "image_peak_reserved_mib", "run_external_peak_mib"},
            "memory",
        ),
        (
            runtime,
            {"forward_seconds", "mechanism_loss_seconds", "mechanism_gradient_seconds", "image_seconds"},
            "runtime",
        ),
        (rng, {"before_image", "after_image"}, "rng"),
        (
            provenance,
            {"source_commit", "implementation_fingerprint", "dataset_manifest_sha256", "support_fingerprint"},
            "provenance",
        ),
    ):
        if not isinstance(value, Mapping):
            raise CABGContractError(f"D3 {name} must be an object.")
        _exact(value, keys, name)
    _validate_gradient_rows(gradients["lm_shared_per_tensor"], "LM shared")
    _validate_gradient_rows(
        gradients["auxiliary_all_trainable_per_tensor"], "auxiliary all-trainable"
    )
    if bool(gradients["cosine_defined"]) != (gradients["cosine"] is not None):
        raise CABGContractError("D3 cosine defined/null semantics mismatch.")
    if record["mechanism_id"] == "M2":
        if not isinstance(record["spatial"], Mapping):
            raise CABGContractError("D3 M2 spatial metrics are missing.")
        _exact(record["spatial"], SPATIAL_KEYS, "spatial")
    elif record["spatial"] is not None:
        raise CABGContractError("D3 M0/M1 must not invent M2 spatial metrics.")
    for value in rng.values():
        if not HEX64.fullmatch(str(value)):
            raise CABGContractError("D3 RNG fingerprint is invalid.")
    if not HEX40.fullmatch(str(provenance["source_commit"])):
        raise CABGContractError("D3 source commit is invalid.")
    for key in ("implementation_fingerprint", "dataset_manifest_sha256", "support_fingerprint"):
        if not HEX64.fullmatch(str(provenance[key])):
            raise CABGContractError(f"D3 provenance hash is invalid: {key}")
    _finite_tree(record, "mechanism record")


def write_records_exclusive(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    if len(records) != 36:
        raise CABGContractError(f"D3 requires exactly 36 mechanism records, got {len(records)}.")
    for record in records:
        validate_mechanism_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for record in records:
            payload = json.dumps(
                record, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
            ) + "\n"
            os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
