#!/usr/bin/env python3
"""Verify selected image bytes and build the immutable D3R runtime annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from reproduction.stage2k.manifest import ManifestContractError, normalize_label


EXPECTED_MANIFEST_SHA256 = "7e581453de5f4abba2eafa87b982eccb751264f66e86ac10f3b92a160ae20db1"
QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def write_exclusive(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def annotation(row: dict[str, object]) -> dict[str, object]:
    label = normalize_label(row["d3r_label"])
    binary = 1 if label == "abnormal" else 0
    return {
        "id": f"cabg-d3r-{row['sample_id']}",
        "image": row["relative_path"],
        "anomaly_label": binary,
        "cabg_sample_id": row["sample_id"],
        "image_sha256": row["sha256"],
        "d3r_selection_key": row["d3r_selection_key"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{QUESTION}"},
            {"from": "gpt", "value": "Yes" if binary else "No"},
        ],
    }


def build(manifest: Path, image_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite D3R dataset root: {output_dir}")
    if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ManifestContractError("D3R locked manifest hash changed.")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(normalize_label(row["d3r_label"]) for row in rows)
    if len(rows) != 24 or counts != Counter({"normal": 12, "abnormal": 12}):
        raise ManifestContractError("D3R manifest count/balance changed.")
    opened = []
    for row in rows:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestContractError("D3R image path is not canonical relative data.")
        path = image_root / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ManifestContractError(f"D3R image missing or byte count changed: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise ManifestContractError(f"D3R image hash changed: {relative}")
        opened.append(str(path.resolve()))
    output_dir.mkdir(parents=True, exist_ok=False)
    locked_manifest = output_dir / "manifest.jsonl"
    locked_manifest.write_bytes(manifest.read_bytes())
    annotations = [annotation(row) for row in rows]
    annotation_path = output_dir / "d3r.json"
    write_exclusive(annotation_path, annotations)
    result = {
        "status": "SUCCESS",
        "decision": "D3R_SELECTED_IMAGE_BYTES_VERIFIED",
        "schema_version": "cabg-lce-mil-v1.2-d3r-runtime-dataset-1",
        "claim_scope": "file_level_confirmatory_mechanism_diagnostic_only",
        "locked_manifest": str(locked_manifest.resolve()),
        "locked_manifest_sha256": sha256_file(locked_manifest),
        "annotation": str(annotation_path.resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "image_root": str(image_root.resolve()),
        "allowed_image_paths": sorted(opened),
        "image_files_verified": len(opened),
        "selected_count": 24,
        "selected_class_counts": dict(sorted(counts.items())),
        "patient_independence_verified": False,
        "patient_independence_unverified": True,
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
        "effectiveness_evaluation_authorized": False,
    }
    write_exclusive(output_dir / "dataset-audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manifest.resolve(), args.image_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
