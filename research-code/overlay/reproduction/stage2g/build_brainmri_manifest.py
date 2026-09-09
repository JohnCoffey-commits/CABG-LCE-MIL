#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from reproduction.stage2g.evaluation_core import sha256_file, write_jsonl


ARCHIVE_SHA256 = "5d96dd41aef9f9d35143479791eeaa9f67819e8ceb6c46a34f146197491c5fdf"
ARCHIVE_BYTES = 1_025_990_467
EXPECTED_FILES = {"good": 98, "ungood": 155}
EXPECTED_UNIQUE = {"good": 87, "ungood": 141}
EXPECTED_DUPLICATE_GROUPS = 22
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def build_records(dataset_root: Path):
    candidate_records = []
    decoding_errors = []
    for label in ("good", "ungood"):
        directory = dataset_root / "brain_mri" / "test" / label
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.iterdir(), key=lambda value: value.name):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = path.relative_to(dataset_root).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions = list(image.size)
                    mode = image.mode
                    image_format = image.format
            except Exception as exc:
                decoding_errors.append(
                    {
                        "relative_path": relative_path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            candidate_records.append(
                {
                    "relative_path": relative_path,
                    "label": label,
                    "ground_truth": "yes" if label == "ungood" else "no",
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "dimensions": dimensions,
                    "mode": mode,
                    "format": image_format,
                }
            )
    return candidate_records, decoding_errors


def deduplicate(candidate_records):
    by_hash = defaultdict(list)
    for record in candidate_records:
        by_hash[record["sha256"]].append(record)
    cross_label = [
        records
        for records in by_hash.values()
        if len({record["label"] for record in records}) > 1
    ]
    if cross_label:
        raise RuntimeError(f"Cross-label duplicate content detected: {cross_label}")

    duplicate_groups = []
    canonical_by_key = {}
    for sha256, records in by_hash.items():
        ordered = sorted(records, key=lambda record: record["relative_path"])
        canonical = ordered[0]
        for record in ordered:
            canonical_by_key[record["relative_path"]] = canonical["relative_path"]
        if len(ordered) > 1:
            duplicate_groups.append(
                {
                    "label": canonical["label"],
                    "sha256": sha256,
                    "canonical_relative_path": canonical["relative_path"],
                    "members": [record["relative_path"] for record in ordered],
                    "group_size": len(ordered),
                }
            )

    enriched = []
    for record in candidate_records:
        canonical_relative_path = canonical_by_key[record["relative_path"]]
        group_size = len(by_hash[record["sha256"]])
        enriched.append(
            {
                **record,
                "canonical_relative_path": canonical_relative_path,
                "duplicate_group_size": group_size,
                "is_canonical": record["relative_path"] == canonical_relative_path,
            }
        )
    enriched.sort(key=lambda record: (record["label"], record["sha256"], record["relative_path"]))
    for index, record in enumerate(enriched):
        record["candidate_index"] = index

    evaluation = [
        {
            key: record[key]
            for key in (
                "relative_path",
                "label",
                "ground_truth",
                "sha256",
                "bytes",
                "dimensions",
                "mode",
                "format",
            )
        }
        for record in enriched
        if record["is_canonical"]
    ]
    evaluation.sort(key=lambda record: (record["label"], record["sha256"], record["relative_path"]))
    rank_by_label = Counter()
    for index, record in enumerate(evaluation):
        label = record["label"]
        rank_by_label[label] += 1
        record["sample_index"] = index
        record["sample_id"] = (
            f"brainmri-{label}-{rank_by_label[label]:03d}-{record['sha256'][:12]}"
        )
    duplicate_groups.sort(
        key=lambda record: (record["label"], record["sha256"], record["canonical_relative_path"])
    )
    return enriched, evaluation, duplicate_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite manifest directory: {args.output_dir}")
    if args.archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError("Official MEDIC-AD anomaly archive byte count mismatch.")
    archive_sha256 = sha256_file(args.archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise RuntimeError("Official MEDIC-AD anomaly archive SHA256 mismatch.")

    candidate_records, decoding_errors = build_records(args.dataset_root)
    enriched, evaluation, duplicate_groups = deduplicate(candidate_records)
    candidate_counts = Counter(record["label"] for record in enriched)
    unique_counts = Counter(record["label"] for record in evaluation)
    if dict(candidate_counts) != EXPECTED_FILES:
        raise RuntimeError(f"Candidate counts differ from protocol: {dict(candidate_counts)}")
    if dict(unique_counts) != EXPECTED_UNIQUE:
        raise RuntimeError(f"Unique counts differ from protocol: {dict(unique_counts)}")
    if decoding_errors:
        raise RuntimeError(f"BrainMRI decoding errors detected: {decoding_errors}")
    if len(duplicate_groups) != EXPECTED_DUPLICATE_GROUPS:
        raise RuntimeError(
            f"Duplicate-group count differs from protocol: {len(duplicate_groups)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = args.output_dir / "candidate-manifest.jsonl"
    evaluation_path = args.output_dir / "evaluation-manifest.jsonl"
    duplicate_path = args.output_dir / "duplicate-groups.json"
    audit_path = args.output_dir / "dataset-audit.json"
    write_jsonl(candidate_path, enriched)
    write_jsonl(evaluation_path, evaluation)
    duplicate_path.write_text(
        json.dumps(duplicate_groups, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "status": "SUCCESS",
        "stage": "2G",
        "dataset": "BrainMRI",
        "evaluation_scope": "image_level_internal_pilot_only",
        "patient_metadata_available": False,
        "patient_independence_verified": False,
        "source_archive": str(args.archive),
        "source_archive_bytes": args.archive.stat().st_size,
        "source_archive_sha256": archive_sha256,
        "candidate_records": len(enriched),
        "candidate_counts": dict(candidate_counts),
        "unique_records": len(evaluation),
        "unique_counts": dict(unique_counts),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_instances_beyond_first": len(enriched) - len(evaluation),
        "cross_label_duplicate_groups": 0,
        "decoding_errors": 0,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "evaluation_manifest": str(evaluation_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_path),
        "duplicate_groups_file": str(duplicate_path),
        "duplicate_groups_sha256": sha256_file(duplicate_path),
        "selection_rule": "all canonical records sorted by label+sha256+relative_path",
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
