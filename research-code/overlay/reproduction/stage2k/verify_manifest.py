#!/usr/bin/env python3
"""Independent metadata-only verifier for a locked D3R manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


SALT = "cabg-lce-mil-v1.2-d3r-v1"
LOCKED_HASHES = {
    "training_development": "c27ea1381d81949c1565412b81df6472be6f08061d11aa140191bbd28dd2810f",
    "threshold_validation": "28ea92d7fe0f957ab4c34d5038bd1cc79db5805bd3ad923c0d1383c5c2e942d5",
    "old_d3": "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c",
    "stage2h_excluded": "98943d6f1f6e333696ad0f38fd3af6d20e82c85239f389a285f9d222de5aeaee",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(value: object) -> str:
    value = str(value).lower()
    if value in {"normal", "good", "no", "0"}:
        return "normal"
    if value in {"abnormal", "ungood", "yes", "1"}:
        return "abnormal"
    raise ValueError(value)


def key(row: dict[str, object]) -> str:
    raw = "\0".join((SALT, label(row["label"]), str(row["sha256"]), str(row["relative_path"])))
    return hashlib.sha256(raw.encode()).hexdigest()


def ids(items: list[dict[str, object]], field: str) -> set[str]:
    return {str(row[field]) for row in items}


def verify(args: argparse.Namespace) -> dict[str, object]:
    sources = {
        "training_development": args.training_development,
        "threshold_validation": args.threshold_validation,
        "old_d3": args.old_d3,
        "stage2h_excluded": args.stage2h_excluded,
    }
    failures: list[str] = []
    for name, path in sources.items():
        if digest(path) != LOCKED_HASHES[name]:
            failures.append(f"source_hash:{name}")
    material = {name: rows(path) for name, path in sources.items()}
    locked = rows(args.manifest)
    expected = []
    for current in ("abnormal", "normal"):
        candidates = [row for row in material["training_development"] if label(row["label"]) == current]
        candidates.sort(key=lambda row: (key(row), str(row["sha256"]), str(row["relative_path"]), str(row["sample_id"])))
        expected.extend(candidates[:12])
    if len(locked) != 24 or Counter(str(row.get("d3r_label")) for row in locked) != Counter({"normal": 12, "abnormal": 12}):
        failures.append("count_or_balance")
    if [(str(row.get("sample_id")), str(row.get("sha256")), str(row.get("relative_path"))) for row in locked] != [(str(row["sample_id"]), str(row["sha256"]), str(row["relative_path"])) for row in expected]:
        failures.append("deterministic_order_or_identity")
    for index, row in enumerate(locked):
        expected_rank = index + 1 if index < 12 else index - 11
        if row.get("d3r_selection_key") != key(row) or row.get("d3r_class_rank") != expected_rank:
            failures.append(f"selection_metadata:{index}")
        if row.get("patient_id_available") is not False or row.get("patient_independence_unverified") is not True:
            failures.append(f"patient_boundary:{index}")
        if row.get("reserved_from_future_d4_training") is not True:
            failures.append(f"future_d4_reservation:{index}")
    for field in ("sample_id", "sha256", "relative_path"):
        if len(ids(locked, field)) != 24:
            failures.append(f"duplicates:{field}")
        for name in ("threshold_validation", "old_d3", "stage2h_excluded"):
            if ids(locked, field) & ids(material[name], field):
                failures.append(f"overlap:{name}:{field}")
    audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusion_audit.read_text(encoding="utf-8"))
    if audit.get("locked_manifest_sha256") != digest(args.manifest) or audit.get("image_files_opened") != 0:
        failures.append("dataset_audit")
    if exclusions.get("protected_internal_test_images_opened") != 0 or exclusions.get("protected_internal_test_outputs_read") != 0:
        failures.append("internal_test_boundary")
    result = {
        "status": "SUCCESS" if not failures else "FAILED",
        "decision": "D3R_MANIFEST_INDEPENDENTLY_VERIFIED" if not failures else "INVALID_QUARANTINED",
        "schema_version": "cabg-lce-mil-v1.2-d3r-manifest-verification-1",
        "manifest_sha256": digest(args.manifest),
        "row_count": len(locked),
        "failures": failures,
        "image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
        "producer_manifest_module_imported": False,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if failures:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-development", type=Path, required=True)
    parser.add_argument("--threshold-validation", type=Path, required=True)
    parser.add_argument("--old-d3", type=Path, required=True)
    parser.add_argument("--stage2h-excluded", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--exclusion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = verify(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
