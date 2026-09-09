#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def annotation(record):
    label = 0 if record["label"] == "good" else 1
    return {
        "id": f"stage2h-{record['sample_id']}",
        "image": record["relative_path"],
        "anomaly_label": label,
        "conversations": [
            {"from": "human", "value": f"<image>\n{QUESTION}"},
            {"from": "gpt", "value": "No" if label == 0 else "Yes"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2g-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage2g-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 2H dataset lock: {args.output_dir}")
    observed_hash = sha256_file(args.stage2g_manifest)
    if observed_hash != args.expected_stage2g_manifest_sha256:
        raise RuntimeError("Stage 2G evaluation manifest SHA256 mismatch.")
    source = read_jsonl(args.stage2g_manifest)
    by_label = {
        label: [record for record in source if record.get("label") == label]
        for label in ("good", "ungood")
    }
    if [len(by_label["good"]), len(by_label["ungood"])] != [87, 141]:
        raise RuntimeError("Unexpected Stage 2G BrainMRI class counts.")

    train = by_label["good"][:8] + by_label["ungood"][:8]
    calibration = by_label["good"][:2] + by_label["ungood"][:2]
    internal_test = by_label["good"][8:31] + by_label["ungood"][8:31]
    splits = {"train": train, "calibration": calibration, "internal-test": internal_test}
    hashes = {name: {record["sha256"] for record in records} for name, records in splits.items()}
    if len(hashes["train"]) != 16 or len(hashes["calibration"]) != 4 or len(hashes["internal-test"]) != 46:
        raise RuntimeError("Stage 2H split contains duplicate image content.")
    if not hashes["calibration"].issubset(hashes["train"]):
        raise RuntimeError("Stage 2H calibration subset is not inside training.")
    if hashes["train"] & hashes["internal-test"]:
        raise RuntimeError("Stage 2H training/internal-test leakage detected.")

    args.output_dir.mkdir(parents=True)
    manifest_hashes = {}
    annotation_hashes = {}
    for name, records in splits.items():
        locked_records = [
            {**record, "stage2h_split": name, "stage2h_split_index": index}
            for index, record in enumerate(records)
        ]
        manifest_path = args.output_dir / f"{name}-manifest.jsonl"
        annotation_path = args.output_dir / f"{name}.json"
        write_jsonl(manifest_path, locked_records)
        annotation_path.write_text(
            json.dumps([annotation(record) for record in locked_records], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_hashes[name] = sha256_file(manifest_path)
        annotation_hashes[name] = sha256_file(annotation_path)

    audit = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "source_manifest": str(args.stage2g_manifest),
        "source_manifest_sha256": observed_hash,
        "selection_rule": "first 8/class train; first 2/class calibration; next 23/class internal-test",
        "counts": {"train": 16, "calibration": 4, "internal-test": 46},
        "class_counts": {
            name: {
                "good": sum(record["label"] == "good" for record in records),
                "ungood": sum(record["label"] == "ungood" for record in records),
            }
            for name, records in splits.items()
        },
        "manifest_sha256": manifest_hashes,
        "annotation_sha256": annotation_hashes,
        "train_internal_test_overlap": 0,
        "calibration_is_training_subset": True,
        "patient_independence_verified": False,
        "claim_scope": "image_level_internal_engineering_only",
    }
    (args.output_dir / "dataset-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
