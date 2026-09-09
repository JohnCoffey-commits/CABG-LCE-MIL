#!/usr/bin/env python3
"""Build the deterministic, contamination-resistant D4-Scout split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.fingerprint import sha256_file
from reproduction.stage2m.constants import (
    EVAL_ABNORMAL,
    EVAL_NORMAL,
    EXPECTED_D3R_SHA256,
    EXPECTED_D4_PILOT_SHA256,
    EXPECTED_EVAL_COUNTS,
    EXPECTED_POST_EXCLUSION_COUNTS,
    EXPECTED_TRAIN_EXPOSURE_COUNTS,
    EXPECTED_TRAIN_UNIQUE_COUNTS,
    EXPECTED_TRAINING_DEVELOPMENT_SHA256,
    SCOUT_BLOCKS,
    SELECTION_SALT,
)

QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, value: object, *, jsonl: bool = False) -> None:
    if path.exists():
        raise FileExistsError(path)
    if jsonl:
        raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in value)  # type: ignore[arg-type]
    else:
        raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, raw.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _key(row: Mapping[str, object]) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}|{row['sample_id']}|{row['sha256']}".encode()).hexdigest()


def _members(rows: Iterable[Mapping[str, object]]) -> dict[str, set[str]]:
    values = list(rows)
    return {
        "sample_id": {str(row["sample_id"]) for row in values},
        "sha256": {str(row["sha256"]) for row in values},
        "relative_path": {str(row["relative_path"]) for row in values},
    }


def _overlap(left: Iterable[Mapping[str, object]], right: Iterable[Mapping[str, object]]) -> dict[str, int]:
    a, b = _members(left), _members(right)
    return {key: len(a[key] & b[key]) for key in a}


def select_split(
    training_rows: list[dict[str, object]],
    d3r_rows: list[dict[str, object]],
    pilot_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    excluded = _members([*d3r_rows, *pilot_rows])
    eligible = [
        row for row in training_rows
        if str(row["sample_id"]) not in excluded["sample_id"]
        and str(row["sha256"]) not in excluded["sha256"]
        and str(row["relative_path"]) not in excluded["relative_path"]
    ]
    counts = Counter(normalize_label(row["label"]) for row in eligible)
    if len(eligible) != 96 or dict(counts) != EXPECTED_POST_EXCLUSION_COUNTS:
        raise CABGContractError(f"Scout post-exclusion pool changed: {len(eligible)}, {counts}")
    classes = {
        label: sorted((row for row in eligible if normalize_label(row["label"]) == label), key=_key)
        for label in ("normal", "abnormal")
    }
    eval_rows = [*classes["normal"][:EVAL_NORMAL], *classes["abnormal"][:EVAL_ABNORMAL]]
    train_classes = {
        "normal": classes["normal"][EVAL_NORMAL:],
        "abnormal": classes["abnormal"][EVAL_ABNORMAL:],
    }
    if Counter(normalize_label(row["label"]) for row in eval_rows) != Counter(EXPECTED_EVAL_COUNTS):
        raise CABGContractError("Scout held-out class composition changed.")
    if {label: len(rows) for label, rows in train_classes.items()} != {"normal": 16, "abnormal": 48}:
        raise CABGContractError("Scout unique training pool changed.")
    exposures: list[dict[str, object]] = []
    for block in range(SCOUT_BLOCKS):
        members = [train_classes["normal"][block % len(train_classes["normal"])]]
        members.extend(train_classes["abnormal"][2 * block:2 * block + 2])
        for position, source in enumerate(members):
            row = dict(source)
            row.update({
                "scout_role": "training",
                "scout_block_id": block,
                "scout_block_position": position,
                "scout_exposure_id": f"block-{block:02d}-position-{position}",
                "scout_label": normalize_label(row["label"]),
                "scout_selection_key": _key(row),
                "selection_salt": SELECTION_SALT,
            })
            exposures.append(row)
    evaluation = []
    for rank, source in enumerate(sorted(eval_rows, key=lambda row: (normalize_label(row["label"]), _key(row)))):
        row = dict(source)
        row.update({
            "scout_role": "held-out-development-evaluation",
            "scout_eval_rank": rank,
            "scout_label": normalize_label(row["label"]),
            "scout_selection_key": _key(row),
            "selection_salt": SELECTION_SALT,
        })
        evaluation.append(row)
    unique_train = list({str(row["sample_id"]): row for row in exposures}.values())
    if Counter(row["scout_label"] for row in exposures) != Counter(EXPECTED_TRAIN_EXPOSURE_COUNTS):
        raise CABGContractError("Scout training exposure counts changed.")
    if Counter(normalize_label(row["label"]) for row in unique_train) != Counter(EXPECTED_TRAIN_UNIQUE_COUNTS):
        raise CABGContractError("Scout unique training counts changed.")
    if any(_overlap(left, right)[key] for left, right in ((exposures, evaluation), (exposures, d3r_rows), (exposures, pilot_rows), (evaluation, d3r_rows), (evaluation, pilot_rows)) for key in ("sample_id", "sha256", "relative_path")):
        raise CABGContractError("Scout split or exclusion overlap detected.")
    return exposures, evaluation, unique_train


def _annotation(row: Mapping[str, object], *, identifier: str, block: int | None = None, position: int | None = None) -> dict[str, object]:
    label = str(row["scout_label"])
    value: dict[str, object] = {
        "id": identifier,
        "image": row["relative_path"],
        "anomaly_label": 1 if label == "abnormal" else 0,
        "cabg_sample_id": row["sample_id"],
        "image_sha256": row["sha256"],
        "scout_role": row["scout_role"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{QUESTION}"},
            {"from": "gpt", "value": "Yes" if label == "abnormal" else "No"},
        ],
    }
    if block is not None:
        value.update({"scout_block_id": block, "scout_block_position": position, "scout_exposure_id": row["scout_exposure_id"]})
    else:
        value["scout_eval_rank"] = row["scout_eval_rank"]
    return value


def build(training: Path, d3r: Path, pilot: Path, image_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    identities = {
        "training_development": (training, EXPECTED_TRAINING_DEVELOPMENT_SHA256),
        "d3r": (d3r, EXPECTED_D3R_SHA256),
        "d4_pilot": (pilot, EXPECTED_D4_PILOT_SHA256),
    }
    for name, (path, expected) in identities.items():
        if sha256_file(path) != expected:
            raise CABGContractError(f"Scout locked {name} manifest changed.")
    training_rows, d3r_rows, pilot_rows = _rows(training), _rows(d3r), _rows(pilot)
    exposures, evaluation, unique_train = select_split(training_rows, d3r_rows, pilot_rows)
    for row in [*unique_train, *evaluation]:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CABGContractError("Scout image path is not canonical relative data.")
        image = image_root / relative
        if not image.is_file() or image.stat().st_size != int(row["bytes"]) or sha256_file(image) != row["sha256"]:
            raise CABGContractError(f"Scout image bytes changed: {relative}")
    output_dir.mkdir(parents=True)
    train_annotations = [
        _annotation(row, identifier=f"scout-train-{row['scout_exposure_id']}-{row['sample_id']}", block=int(row["scout_block_id"]), position=int(row["scout_block_position"]))
        for row in exposures
    ]
    eval_annotations = [
        _annotation(row, identifier=f"scout-eval-{row['scout_eval_rank']}-{row['sample_id']}")
        for row in evaluation
    ]
    _write(output_dir / "train-manifest.jsonl", exposures, jsonl=True)
    _write(output_dir / "train-unique-manifest.jsonl", unique_train, jsonl=True)
    _write(output_dir / "eval-manifest.jsonl", evaluation, jsonl=True)
    _write(output_dir / "train.json", train_annotations)
    _write(output_dir / "eval.json", eval_annotations)
    overlap = {
        "train_eval": _overlap(exposures, evaluation),
        "train_d3r": _overlap(exposures, d3r_rows),
        "train_d4_pilot": _overlap(exposures, pilot_rows),
        "eval_d3r": _overlap(evaluation, d3r_rows),
        "eval_d4_pilot": _overlap(evaluation, pilot_rows),
    }
    result = {
        "status": "SUCCESS",
        "decision": "D4_SCOUT_SPLIT_LOCKED",
        "claim_scope": "development_only_exploratory_effect_scout",
        "selection_salt": SELECTION_SALT,
        "source_hashes": {name: expected for name, (_, expected) in identities.items()},
        "source_counts": {"training_development": len(training_rows), "d3r": len(d3r_rows), "d4_pilot": len(pilot_rows)},
        "post_exclusion_count": 96,
        "post_exclusion_class_counts": EXPECTED_POST_EXCLUSION_COUNTS,
        "train_unique_count": len(unique_train),
        "train_unique_class_counts": EXPECTED_TRAIN_UNIQUE_COUNTS,
        "train_exposure_count": len(exposures),
        "train_exposure_class_counts": EXPECTED_TRAIN_EXPOSURE_COUNTS,
        "blocks": SCOUT_BLOCKS,
        "block_composition": {"normal": 1, "abnormal": 2},
        "eval_count": len(evaluation),
        "eval_class_counts": EXPECTED_EVAL_COUNTS,
        "train_manifest": str((output_dir / "train-manifest.jsonl").resolve()),
        "train_manifest_sha256": sha256_file(output_dir / "train-manifest.jsonl"),
        "train_unique_manifest": str((output_dir / "train-unique-manifest.jsonl").resolve()),
        "eval_manifest": str((output_dir / "eval-manifest.jsonl").resolve()),
        "eval_manifest_sha256": sha256_file(output_dir / "eval-manifest.jsonl"),
        "train_annotation": str((output_dir / "train.json").resolve()),
        "train_annotation_sha256": sha256_file(output_dir / "train.json"),
        "eval_annotation": str((output_dir / "eval.json").resolve()),
        "eval_annotation_sha256": sha256_file(output_dir / "eval.json"),
        "image_root": str(image_root.resolve()),
        "overlap": overlap,
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
        "old_d3_used": False,
        "threshold_or_hyperparameter_data_used": False,
        "previous_calibration_used": False,
        "patient_independence_verified": False,
    }
    _write(output_dir / "split-audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-development", type=Path, required=True)
    parser.add_argument("--d3r-manifest", type=Path, required=True)
    parser.add_argument("--d4-pilot-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(*(value.resolve() for value in (args.training_development, args.d3r_manifest, args.d4_pilot_manifest, args.image_root, args.output_dir)))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
