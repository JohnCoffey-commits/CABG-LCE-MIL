#!/usr/bin/env python3
"""Build the locked CABG-MIL development split without opening image files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from reproduction.stage2i.block_sampler import split_development_pool
from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import DATASET_SCHEMA_VERSION
from reproduction.stage2i.fingerprint import dataset_fingerprint, sha256_file


EXPECTED_SOURCE_COUNTS = {"normal": 87, "abnormal": 141}
EXPECTED_EXCLUDED_COUNTS = {"normal": 31, "abnormal": 31}
EXPECTED_DEVELOPMENT_COUNTS = {"normal": 56, "abnormal": 110}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CABGContractError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    if not rows:
        raise CABGContractError(f"Manifest is empty: {path}")
    return rows


def _identity(record: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(record["sample_id"]),
        str(record["sha256"]),
        str(record["relative_path"]),
    )


def _class_counts(records: Iterable[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(normalize_label(value["label"]) for value in records))


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def build_locked_dataset(
    *,
    source_manifest: Path,
    stage2h_train_manifest: Path,
    internal_test_manifest: Path,
    stage2h_audit_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite CABG dataset lock: {output_dir}")
    audit = json.loads(stage2h_audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "SUCCESS" or audit.get("protocol_version") != "2.2":
        raise CABGContractError("Stage 2H v2.2 audit status/version mismatch.")
    if audit.get("internal_test_used_for_selection") is not False:
        raise CABGContractError("Stage 2H reports internal-test selection access.")
    expected_manifest_hashes = audit.get("manifest_sha256", {})
    actual_hashes = {
        "source": sha256_file(source_manifest),
        "train": sha256_file(stage2h_train_manifest),
        "internal-test": sha256_file(internal_test_manifest),
        "stage2h-audit": sha256_file(stage2h_audit_path),
    }
    if actual_hashes["source"] != audit.get("source_manifest_sha256"):
        raise CABGContractError("Stage 2G source manifest hash differs from Stage 2H provenance.")
    if actual_hashes["train"] != expected_manifest_hashes.get("train"):
        raise CABGContractError("Stage 2H train manifest hash mismatch.")
    if actual_hashes["internal-test"] != expected_manifest_hashes.get("internal-test"):
        raise CABGContractError("Stage 2H internal-test manifest hash mismatch.")

    source = read_jsonl(source_manifest)
    training_exclusion = read_jsonl(stage2h_train_manifest)
    internal_exclusion = read_jsonl(internal_test_manifest)
    if _class_counts(source) != EXPECTED_SOURCE_COUNTS:
        raise CABGContractError(f"Source class counts changed: {_class_counts(source)}")
    if len(training_exclusion) != 16 or len(internal_exclusion) != 46:
        raise CABGContractError("Stage 2H exclusion counts changed.")

    source_by_id = {_identity(value): value for value in source}
    if len(source_by_id) != len(source):
        raise CABGContractError("Source manifest contains duplicate identities.")
    excluded_rows = training_exclusion + internal_exclusion
    excluded_identities = {_identity(value) for value in excluded_rows}
    if len(excluded_identities) != 62:
        raise CABGContractError("Stage 2H train/internal-test exclusions overlap or duplicate.")
    unknown = excluded_identities - set(source_by_id)
    if unknown:
        raise CABGContractError(f"Excluded identities are absent from Stage 2G source: {len(unknown)}")
    if _class_counts(excluded_rows) != EXPECTED_EXCLUDED_COUNTS:
        raise CABGContractError(f"Exclusion class counts changed: {_class_counts(excluded_rows)}")

    development = [value for identity, value in source_by_id.items() if identity not in excluded_identities]
    if len(development) != 166 or _class_counts(development) != EXPECTED_DEVELOPMENT_COUNTS:
        raise CABGContractError(
            f"Development pool changed: count={len(development)}, classes={_class_counts(development)}"
        )
    split = split_development_pool(development)
    development_rows = sorted(
        split["training-development"] + split["threshold-validation"],
        key=lambda value: (value["label"], value["split_key"], value["sample_id"]),
    )
    mechanism_rows = []
    for value in split["mechanism-diagnostic"]:
        row = dict(value)
        row["cabg_role"] = "mechanism-diagnostic"
        mechanism_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "development-pool": output_dir / "development-pool-manifest.jsonl",
        "training-development": output_dir / "training-development-manifest.jsonl",
        "threshold-validation": output_dir / "threshold-validation-manifest.jsonl",
        "mechanism-diagnostic": output_dir / "mechanism-diagnostic-manifest.jsonl",
        "excluded": output_dir / "excluded-stage2h-manifest.jsonl",
    }
    _write_jsonl(paths["development-pool"], development_rows)
    _write_jsonl(paths["training-development"], split["training-development"])
    _write_jsonl(paths["threshold-validation"], split["threshold-validation"])
    _write_jsonl(paths["mechanism-diagnostic"], mechanism_rows)
    _write_jsonl(paths["excluded"], sorted(excluded_rows, key=_identity))
    output_hashes = {name: sha256_file(path) for name, path in paths.items()}
    result = {
        "status": "SUCCESS",
        "schema_version": DATASET_SCHEMA_VERSION,
        "claim_scope": "image_level_internal_engineering_only",
        "patient_independence_verified": False,
        "image_files_opened": 0,
        "internal_test_image_files_opened": 0,
        "internal_test_outputs_read": 0,
        "source_paths": {
            "stage2g_evaluation_manifest": str(source_manifest),
            "stage2h_train_manifest": str(stage2h_train_manifest),
            "stage2h_internal_test_manifest": str(internal_test_manifest),
            "stage2h_dataset_audit": str(stage2h_audit_path),
        },
        "source_sha256": actual_hashes,
        "counts": {
            "stage2g-source": len(source),
            "excluded-stage2h": len(excluded_rows),
            "development-pool": len(development_rows),
            "training-development": len(split["training-development"]),
            "threshold-validation": len(split["threshold-validation"]),
            "mechanism-diagnostic": len(mechanism_rows),
        },
        "class_counts": split["counts"],
        "output_paths": {name: str(path) for name, path in paths.items()},
        "output_sha256": output_hashes,
        "development_fingerprint": dataset_fingerprint(development_rows),
        "split_fingerprint": split["split_fingerprint"],
        "selection_rule": "locked CABG v1.1 SHA-256 split; metadata only; no image content opened",
    }
    audit_output = output_dir / "dataset-audit.json"
    audit_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--stage2h-train-manifest", type=Path, required=True)
    parser.add_argument("--internal-test-manifest", type=Path, required=True)
    parser.add_argument("--stage2h-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_locked_dataset(
        source_manifest=args.source_manifest.resolve(),
        stage2h_train_manifest=args.stage2h_train_manifest.resolve(),
        internal_test_manifest=args.internal_test_manifest.resolve(),
        stage2h_audit_path=args.stage2h_audit.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
