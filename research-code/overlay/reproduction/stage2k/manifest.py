#!/usr/bin/env python3
"""Build the metadata-only, deterministic, balanced D3R manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from reproduction.stage2k.constants import (
    ABNORMAL_LABEL,
    EXPECTED_OLD_D3_SHA256,
    EXPECTED_STAGE2H_EXCLUDED_SHA256,
    EXPECTED_THRESHOLD_VALIDATION_SHA256,
    EXPECTED_TRAINING_DEVELOPMENT_SHA256,
    NORMAL_LABEL,
    SELECTION_SALT,
)


class ManifestContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ManifestContractError(f"Empty or malformed JSONL: {path}")
    return rows


def normalize_label(value: object) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"normal", "good", "no", "0"}:
        return NORMAL_LABEL
    if lowered in {"abnormal", "ungood", "yes", "1"}:
        return ABNORMAL_LABEL
    raise ManifestContractError(f"Unsupported label: {value!r}")


def identity(row: dict[str, object], field: str) -> str:
    value = str(row.get(field, ""))
    if not value:
        raise ManifestContractError(f"Manifest row missing {field}")
    return value


def selection_key(row: dict[str, object]) -> str:
    label = normalize_label(row.get("label"))
    payload = "\0".join(
        (SELECTION_SALT, label, identity(row, "sha256"), identity(row, "relative_path"))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity_sets(rows: Iterable[dict[str, object]]) -> dict[str, set[str]]:
    material = list(rows)
    return {
        "sha256": {identity(row, "sha256") for row in material},
        "relative_path": {identity(row, "relative_path") for row in material},
        "sample_id": {identity(row, "sample_id") for row in material},
    }


def intersections(left: list[dict[str, object]], right: list[dict[str, object]]) -> dict[str, int]:
    a, b = identity_sets(left), identity_sets(right)
    return {field: len(a[field] & b[field]) for field in a}


def _exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for row in rows:
            raw = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
            os.write(descriptor, (raw + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(
    training_development: Path,
    threshold_validation: Path,
    old_d3: Path,
    stage2h_excluded: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite D3R manifest root: {output_dir}")
    expected = {
        "training_development": (training_development, EXPECTED_TRAINING_DEVELOPMENT_SHA256),
        "threshold_validation": (threshold_validation, EXPECTED_THRESHOLD_VALIDATION_SHA256),
        "old_d3": (old_d3, EXPECTED_OLD_D3_SHA256),
        "stage2h_excluded": (stage2h_excluded, EXPECTED_STAGE2H_EXCLUDED_SHA256),
    }
    observed_hashes = {name: sha256_file(path) for name, (path, _) in expected.items()}
    mismatches = [name for name, (_, digest) in expected.items() if observed_hashes[name] != digest]
    if mismatches:
        raise ManifestContractError(f"Locked source manifest hash mismatch: {mismatches}")
    pools = {name: read_jsonl(path) for name, (path, _) in expected.items()}
    eligible = pools["training_development"]
    if len(eligible) != 132 or Counter(normalize_label(row.get("label")) for row in eligible) != Counter({ABNORMAL_LABEL: 88, NORMAL_LABEL: 44}):
        raise ManifestContractError("Eligible training-development composition changed.")
    if len(pools["threshold_validation"]) != 34 or len(pools["old_d3"]) != 12 or len(pools["stage2h_excluded"]) != 62:
        raise ManifestContractError("Exclusion manifest counts changed.")
    exclusion_overlap = {
        name: intersections(eligible, rows)
        for name, rows in pools.items()
        if name != "training_development"
    }
    if any(value for result in exclusion_overlap.values() for value in result.values()):
        raise ManifestContractError(f"Eligible pool overlaps an exclusion: {exclusion_overlap}")
    if any(value == 0 for value in intersections(pools["old_d3"], pools["threshold_validation"]).values()):
        raise ManifestContractError("Old D3 must be contained in threshold-validation by all identities.")

    selected: list[dict[str, object]] = []
    for label in (ABNORMAL_LABEL, NORMAL_LABEL):
        candidates = [row for row in eligible if normalize_label(row.get("label")) == label]
        ordered = sorted(
            candidates,
            key=lambda row: (
                selection_key(row),
                identity(row, "sha256"),
                identity(row, "relative_path"),
                identity(row, "sample_id"),
            ),
        )
        for rank, row in enumerate(ordered[:12], start=1):
            locked = dict(row)
            locked.update(
                {
                    "cabg_role": "d3r-confirmatory-mechanism",
                    "d3r_label": label,
                    "d3r_class_rank": rank,
                    "d3r_selection_key": selection_key(row),
                    "selection_salt": SELECTION_SALT,
                    "patient_id": None,
                    "patient_id_available": False,
                    "patient_independence_unverified": True,
                    "reserved_from_future_d4_training": True,
                }
            )
            selected.append(locked)
    if len(selected) != 24 or Counter(row["d3r_label"] for row in selected) != Counter({ABNORMAL_LABEL: 12, NORMAL_LABEL: 12}):
        raise ManifestContractError("D3R selection is not exactly balanced 24.")
    selected_ids = identity_sets(selected)
    if any(len(values) != 24 for values in selected_ids.values()):
        raise ManifestContractError("D3R selection contains duplicate identities.")
    selected_overlap = {
        name: intersections(selected, rows)
        for name, rows in pools.items()
        if name != "training_development"
    }
    if any(value for result in selected_overlap.values() for value in result.values()):
        raise ManifestContractError(f"D3R selection overlaps an exclusion: {selected_overlap}")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.jsonl"
    _exclusive_jsonl(manifest_path, selected)
    digest = sha256_file(manifest_path)
    descriptor = os.open(output_dir / "manifest.sha256", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, f"{digest}  manifest.jsonl\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    exclusion_audit = {
        "status": "SUCCESS",
        "schema_version": "cabg-lce-mil-v1.2-d3r-exclusion-1",
        "selection_salt": SELECTION_SALT,
        "source_sha256": observed_hashes,
        "source_counts": {name: len(rows) for name, rows in pools.items()},
        "eligible_pool_overlap_with_exclusions": exclusion_overlap,
        "selected_overlap_with_exclusions": selected_overlap,
        "threshold_validation_fully_excluded": True,
        "old_d3_fully_excluded": True,
        "stage2h_train_calibration_internal_test_fully_excluded": True,
        "protected_internal_test_images_opened": 0,
        "protected_internal_test_outputs_read": 0,
    }
    dataset_audit = {
        "status": "SUCCESS",
        "schema_version": "cabg-lce-mil-v1.2-d3r-dataset-audit-1",
        "claim_scope": "file_level_confirmatory_mechanism_diagnostic_only",
        "eligible_source_role": "training-development",
        "eligible_count": 132,
        "eligible_class_counts": {ABNORMAL_LABEL: 88, NORMAL_LABEL: 44},
        "selection_rule": "per-class deterministic SHA-256 ordering locked before image/model access",
        "selection_salt": SELECTION_SALT,
        "locked_manifest": str(manifest_path.resolve()),
        "locked_manifest_sha256": digest,
        "selected_count": 24,
        "selected_class_counts": {ABNORMAL_LABEL: 12, NORMAL_LABEL: 12},
        "unique_sha256": len(selected_ids["sha256"]),
        "unique_relative_path": len(selected_ids["relative_path"]),
        "unique_sample_id": len(selected_ids["sample_id"]),
        "metadata_files_read": 4,
        "image_files_opened": 0,
        "patient_id_available_count": 0,
        "patient_independence_verified": False,
        "patient_independence_unverified": True,
        "selected_reserved_from_future_d4_training": True,
        "effectiveness_evaluation_authorized": False,
        "internal_test_evaluation_authorized": False,
    }
    _exclusive_json(output_dir / "exclusion-audit.json", exclusion_audit)
    _exclusive_json(output_dir / "dataset-audit.json", dataset_audit)
    result = {
        "status": "SUCCESS",
        "decision": "D3R_MANIFEST_LOCKED_METADATA_ONLY",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "dataset_audit": str((output_dir / "dataset-audit.json").resolve()),
        "exclusion_audit": str((output_dir / "exclusion-audit.json").resolve()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-development", type=Path, required=True)
    parser.add_argument("--threshold-validation", type=Path, required=True)
    parser.add_argument("--old-d3", type=Path, required=True)
    parser.add_argument("--stage2h-excluded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.training_development.resolve(),
        args.threshold_validation.resolve(),
        args.old_d3.resolve(),
        args.stage2h_excluded.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
