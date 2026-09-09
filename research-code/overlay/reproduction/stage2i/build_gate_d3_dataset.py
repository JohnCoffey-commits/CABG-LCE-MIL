#!/usr/bin/env python3
"""Build and verify the immutable 12-image Gate D3 annotation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file


EXPECTED_MANIFEST_SHA256 = "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c"
QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CABGContractError("D3 mechanism manifest is empty or malformed.")
    return rows


def annotation(row: dict[str, object]) -> dict[str, object]:
    label = normalize_label(row["label"])
    binary = 1 if label == "abnormal" else 0
    return {
        "id": f"cabg-d3-{row['sample_id']}",
        "image": row["relative_path"],
        "anomaly_label": binary,
        "cabg_sample_id": row["sample_id"],
        "image_sha256": row["sha256"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{QUESTION}"},
            {"from": "gpt", "value": "Yes" if binary else "No"},
        ],
    }


def build_dataset(manifest: Path, image_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Gate D3 dataset lock: {output_dir}")
    observed = sha256_file(manifest)
    if observed != EXPECTED_MANIFEST_SHA256:
        raise CABGContractError("Gate D3 mechanism manifest SHA-256 mismatch.")
    rows = read_rows(manifest)
    if len(rows) != 12:
        raise CABGContractError(f"Gate D3 requires exactly 12 rows, got {len(rows)}.")
    required = {"sample_id", "sha256", "relative_path", "label", "bytes", "cabg_role"}
    identities = set()
    allowed_paths = []
    counts = Counter()
    for row in rows:
        if not required.issubset(row):
            raise CABGContractError(f"D3 manifest row is missing keys: {sorted(required - set(row))}")
        if row["cabg_role"] != "mechanism-diagnostic":
            raise CABGContractError("D3 manifest contains a non-mechanism row.")
        label = normalize_label(row["label"])
        counts[label] += 1
        identity = (str(row["sample_id"]), str(row["sha256"]), str(row["relative_path"]))
        if identity in identities:
            raise CABGContractError("D3 manifest contains a duplicate identity.")
        identities.add(identity)
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CABGContractError("D3 image path is not canonical relative data.")
        path = image_root / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise CABGContractError(f"D3 image missing or byte count changed: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise CABGContractError(f"D3 image SHA-256 changed: {relative}")
        allowed_paths.append(str(path.resolve()))
    if dict(counts) != {"abnormal": 6, "normal": 6}:
        raise CABGContractError(f"D3 class balance changed: {dict(counts)}")
    annotations = [annotation(row) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=False)
    locked_manifest = output_dir / "mechanism-diagnostic-manifest.jsonl"
    locked_manifest.write_bytes(manifest.read_bytes())
    annotation_path = output_dir / "mechanism-diagnostic.json"
    annotation_path.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "SUCCESS",
        "schema_version": "cabg-v1.1-gate-d3-dataset-1",
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": observed,
        "locked_manifest": str(locked_manifest.resolve()),
        "locked_manifest_sha256": sha256_file(locked_manifest),
        "annotation": str(annotation_path.resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "annotation_fingerprint": canonical_json_sha256(annotations),
        "image_root": str(image_root.resolve()),
        "allowed_image_paths": sorted(allowed_paths),
        "count": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "unique_identities": len(identities),
        "image_files_verified": len(rows),
        "internal_test_image_files_opened": 0,
        "internal_test_outputs_read": 0,
        "patient_independence_verified": False,
        "claim_scope": "image_level_internal_mechanism_diagnostic_only",
    }
    audit = output_dir / "dataset-audit.json"
    audit.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_dataset(args.manifest.resolve(), args.image_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
