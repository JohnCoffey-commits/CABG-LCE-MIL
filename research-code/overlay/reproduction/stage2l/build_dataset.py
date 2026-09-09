#!/usr/bin/env python3
"""Lock the D4 pilot as a deterministic subset of training-development minus D3R."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from reproduction.stage2i.fingerprint import sha256_file
from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2l.constants import (
    ABNORMAL_PER_BLOCK,
    EXPECTED_D3R_SHA256,
    EXPECTED_ELIGIBLE_COUNTS,
    EXPECTED_PILOT_COUNTS,
    EXPECTED_TRAINING_DEVELOPMENT_SHA256,
    NORMAL_PER_BLOCK,
    PILOT_BLOCKS,
    SELECTION_SALT,
)

QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, value: object, *, jsonl: bool = False) -> None:
    if path.exists():
        raise FileExistsError(path)
    if jsonl:
        raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in value)
    else:
        raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, raw.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _key(row: dict[str, object]) -> str:
    raw = f"{SELECTION_SALT}|{row['sample_id']}|{row['sha256']}".encode()
    return hashlib.sha256(raw).hexdigest()


def build(training: Path, d3r: Path, image_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(training) != EXPECTED_TRAINING_DEVELOPMENT_SHA256:
        raise CABGContractError("D4 training-development source hash changed.")
    if sha256_file(d3r) != EXPECTED_D3R_SHA256:
        raise CABGContractError("D4 D3R exclusion manifest hash changed.")
    training_rows, d3r_rows = _rows(training), _rows(d3r)
    d3r_ids = {str(row["sample_id"]) for row in d3r_rows}
    d3r_hashes = {str(row["sha256"]) for row in d3r_rows}
    eligible = [
        row for row in training_rows
        if str(row["sample_id"]) not in d3r_ids and str(row["sha256"]) not in d3r_hashes
    ]
    counts = Counter(normalize_label(row["label"]) for row in eligible)
    if dict(counts) != EXPECTED_ELIGIBLE_COUNTS or len(eligible) != 108:
        raise CABGContractError(f"D4 eligible-pool identity changed: {counts}")
    classes = {
        label: sorted((row for row in eligible if normalize_label(row["label"]) == label), key=_key)
        for label in ("normal", "abnormal")
    }
    selected = []
    for block in range(PILOT_BLOCKS):
        members = [classes["normal"][block * NORMAL_PER_BLOCK]]
        members += classes["abnormal"][block * ABNORMAL_PER_BLOCK:(block + 1) * ABNORMAL_PER_BLOCK]
        for position, source in enumerate(members):
            row = dict(source)
            row.update({
                "d4_role": "development-only-training-feasibility-pilot",
                "d4_block_id": block,
                "d4_block_position": position,
                "d4_selection_key": _key(row),
                "d4_label": normalize_label(row["label"]),
                "selection_salt": SELECTION_SALT,
            })
            selected.append(row)
    selected_counts = Counter(row["d4_label"] for row in selected)
    if dict(selected_counts) != EXPECTED_PILOT_COUNTS or len(selected) != 12:
        raise CABGContractError("D4 pilot class composition changed.")
    if any(str(row["sample_id"]) in d3r_ids or str(row["sha256"]) in d3r_hashes for row in selected):
        raise CABGContractError("D3R leakage into D4 pilot.")
    allowed = []
    for row in selected:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CABGContractError("D4 image path is not canonical relative data.")
        image = image_root / relative
        if not image.is_file() or image.stat().st_size != int(row["bytes"]) or sha256_file(image) != row["sha256"]:
            raise CABGContractError(f"D4 selected image bytes changed: {relative}")
        allowed.append(str(image.resolve()))
    annotations = [{
        "id": f"cabg-d4-{row['sample_id']}",
        "image": row["relative_path"],
        "anomaly_label": 1 if row["d4_label"] == "abnormal" else 0,
        "cabg_sample_id": row["sample_id"],
        "image_sha256": row["sha256"],
        "d4_block_id": row["d4_block_id"],
        "d4_block_position": row["d4_block_position"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{QUESTION}"},
            {"from": "gpt", "value": "Yes" if row["d4_label"] == "abnormal" else "No"},
        ],
    } for row in selected]
    output_dir.mkdir(parents=True)
    _write(output_dir / "manifest.jsonl", selected, jsonl=True)
    _write(output_dir / "train.json", annotations)
    result = {
        "status": "SUCCESS",
        "decision": "D4_PILOT_DATA_LOCKED",
        "claim_scope": "development_only_training_feasibility",
        "training_development_sha256": EXPECTED_TRAINING_DEVELOPMENT_SHA256,
        "d3r_exclusion_sha256": EXPECTED_D3R_SHA256,
        "eligible_count": len(eligible),
        "eligible_class_counts": dict(sorted(counts.items())),
        "selected_count": len(selected),
        "selected_class_counts": dict(sorted(selected_counts.items())),
        "blocks": PILOT_BLOCKS,
        "block_composition": {"normal": NORMAL_PER_BLOCK, "abnormal": ABNORMAL_PER_BLOCK},
        "manifest": str((output_dir / "manifest.jsonl").resolve()),
        "manifest_sha256": sha256_file(output_dir / "manifest.jsonl"),
        "annotation": str((output_dir / "train.json").resolve()),
        "annotation_sha256": sha256_file(output_dir / "train.json"),
        "image_root": str(image_root.resolve()),
        "allowed_image_paths": sorted(allowed),
        "d3r_overlap": 0,
        "old_d3_used": False,
        "threshold_or_hyperparameter_data_used": False,
        "previous_calibration_used": False,
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
        "patient_independence_verified": False,
        "effectiveness_evaluation_authorized": False,
    }
    _write(output_dir / "dataset-audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-development", type=Path, required=True)
    parser.add_argument("--d3r-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.training_development.resolve(), args.d3r_manifest.resolve(), args.image_root.resolve(), args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
