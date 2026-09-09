#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

from reproduction.stage2h.build_dataset import annotation, read_jsonl, write_jsonl


V21_MANIFEST_SHA256 = {
    "train": "580af0a78aec247004a6367c831594ad6c4325d0a3ed2ad9be8cce8c37d5ed3c",
    "previous-calibration": "c6ff4dba0e53f8a3a185360c395d6b8f262476bdaf97229a85f96a576f922f7a",
    "internal-test": "4d8c419d0a0571fe2d9ba7462beb65fa0fccd31260f9b6f5d2f62be278b01c03",
}
V21_ANNOTATION_SHA256 = {
    "train": "24b6d05f61817213f4ea4f64b7f09eed8b0364adbe33db89c68286e10bffdfb7",
    "previous-calibration": "518961b6390470a686816bee16172b565c504ebe75e5c1320f834ff779805293",
    "internal-test": "51ddb54c0b342b94b73cfd1fd7d6533c7211f3329c48d63e843383b3b6fc5faf",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked(records, split):
    return [
        {**record, "stage2h_split": split, "stage2h_split_index": index}
        for index, record in enumerate(records)
    ]


def write_split(output_dir: Path, name: str, records):
    manifest_path = output_dir / f"{name}-manifest.jsonl"
    annotation_path = output_dir / f"{name}.json"
    write_jsonl(manifest_path, records)
    annotation_path.write_text(
        json.dumps([annotation(record) for record in records], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(manifest_path), sha256_file(annotation_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2g-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage2g-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 2H v2.2 dataset lock: {args.output_dir}")
    observed_source_hash = sha256_file(args.stage2g_manifest)
    if observed_source_hash != args.expected_stage2g_manifest_sha256:
        raise RuntimeError("Stage 2G evaluation manifest SHA256 mismatch.")
    source = read_jsonl(args.stage2g_manifest)
    by_label = {
        label: [record for record in source if record.get("label") == label]
        for label in ("good", "ungood")
    }
    if [len(by_label["good"]), len(by_label["ungood"])] != [87, 141]:
        raise RuntimeError("Unexpected Stage 2G BrainMRI class counts.")

    train_source = by_label["good"][:8] + by_label["ungood"][:8]
    previous_source = by_label["good"][:2] + by_label["ungood"][:2]
    recalibration_source = by_label["good"][2:8] + by_label["ungood"][2:8]
    internal_source = by_label["good"][8:31] + by_label["ungood"][8:31]
    splits = {
        "train": locked(train_source, "train"),
        "previous-calibration": locked(previous_source, "calibration"),
        "calibration": locked(recalibration_source, "calibration"),
        "internal-test": locked(internal_source, "internal-test"),
    }
    expected_counts = {"train": 16, "previous-calibration": 4, "calibration": 12, "internal-test": 46}
    hashes = {name: {record["sha256"] for record in records} for name, records in splits.items()}
    if any(len(hashes[name]) != count for name, count in expected_counts.items()):
        raise RuntimeError("Stage 2H v2.2 split contains duplicate image content.")
    if not hashes["calibration"] < hashes["train"]:
        raise RuntimeError("Recalibration must be a strict training subset.")
    if hashes["calibration"] & hashes["previous-calibration"]:
        raise RuntimeError("Recalibration overlaps the previous calibration subset.")
    if hashes["calibration"] | hashes["previous-calibration"] != hashes["train"]:
        raise RuntimeError("Previous and new calibration subsets do not partition training.")
    if hashes["train"] & hashes["internal-test"]:
        raise RuntimeError("Training/internal-test leakage detected.")

    args.output_dir.mkdir(parents=True)
    manifest_hashes = {}
    annotation_hashes = {}
    for name, records in splits.items():
        manifest_hashes[name], annotation_hashes[name] = write_split(args.output_dir, name, records)
    for name in ("train", "previous-calibration", "internal-test"):
        if manifest_hashes[name] != V21_MANIFEST_SHA256[name]:
            raise RuntimeError(f"v2.1 parent manifest changed: {name}")
        if annotation_hashes[name] != V21_ANNOTATION_SHA256[name]:
            raise RuntimeError(f"v2.1 parent annotation changed: {name}")

    audit = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "protocol_version": "2.2",
        "source_manifest": str(args.stage2g_manifest),
        "source_manifest_sha256": observed_source_hash,
        "selection_rule": "v2.1 training minus v2.1 calibration; inherited order",
        "counts": expected_counts,
        "class_counts": {
            name: {
                "good": sum(record["label"] == "good" for record in records),
                "ungood": sum(record["label"] == "ungood" for record in records),
            }
            for name, records in splits.items()
        },
        "manifest_sha256": manifest_hashes,
        "annotation_sha256": annotation_hashes,
        "v21_parent_manifest_sha256": V21_MANIFEST_SHA256,
        "v21_parent_annotation_sha256": V21_ANNOTATION_SHA256,
        "recalibration_is_strict_training_subset": True,
        "recalibration_previous_calibration_overlap": 0,
        "calibration_partition_equals_training": True,
        "train_internal_test_overlap": 0,
        "recalibration_internal_test_overlap": 0,
        "internal_test_used_for_selection": False,
        "patient_independence_verified": False,
        "claim_scope": "image_level_internal_engineering_only",
    }
    (args.output_dir / "dataset-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
