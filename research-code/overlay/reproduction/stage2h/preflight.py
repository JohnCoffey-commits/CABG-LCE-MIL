#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

from reproduction.stage2h.source_fingerprint import implementation_source_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expected-base-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pinned_revision = (args.base_model / ".medic-ad-pinned-revision").read_text(encoding="utf-8").strip()
    if pinned_revision != args.expected_base_revision:
        raise RuntimeError("Base-model pinned revision mismatch.")
    audit_path = args.dataset_lock / "dataset-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "SUCCESS" or audit.get("counts") != {
        "train": 16,
        "calibration": 4,
        "internal-test": 46,
    }:
        raise RuntimeError("Stage 2H dataset audit is not locked and successful.")
    split_records = {}
    for split, expected_count in (("train", 16), ("calibration", 4), ("internal-test", 46)):
        manifest_path = args.dataset_lock / f"{split}-manifest.jsonl"
        annotation_path = args.dataset_lock / f"{split}.json"
        if sha256_file(manifest_path) != audit["manifest_sha256"][split]:
            raise RuntimeError(f"Stage 2H {split} manifest hash mismatch.")
        if sha256_file(annotation_path) != audit["annotation_sha256"][split]:
            raise RuntimeError(f"Stage 2H {split} annotation hash mismatch.")
        manifest = read_jsonl(manifest_path)
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if len(manifest) != expected_count or len(annotations) != expected_count:
            raise RuntimeError(f"Stage 2H {split} record-count mismatch.")
        for source, annotation in zip(manifest, annotations):
            expected_label = 0 if source["label"] == "good" else 1
            if annotation.get("anomaly_label") != expected_label:
                raise RuntimeError(f"Explicit label mismatch: {annotation.get('id')}")
            if not isinstance(annotation.get("image"), str):
                raise RuntimeError("Stage 2H annotation has no single image path.")
            image_path = args.image_root / annotation["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if image_path.stat().st_size != source["bytes"] or sha256_file(image_path) != source["sha256"]:
                raise RuntimeError(f"Stage 2H image artifact mismatch: {annotation['image']}")
            with Image.open(image_path) as image:
                if list(image.size) != source["dimensions"]:
                    raise RuntimeError(f"Stage 2H image dimensions changed: {annotation['image']}")
                image.verify()
        split_records[split] = manifest
    train_hashes = {record["sha256"] for record in split_records["train"]}
    test_hashes = {record["sha256"] for record in split_records["internal-test"]}
    if train_hashes & test_hashes:
        raise RuntimeError("Stage 2H train/internal-test leakage detected.")

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), use_fast=False)
    token_audit = {}
    for form in ("yes", "no", " yes", " no", "Yes", "No", " Yes", " No"):
        ids = tokenizer.encode(form, add_special_tokens=False)
        token_audit[form] = {"token_ids": ids, "sequence_length": len(ids), "single_token": len(ids) == 1}
    selected_single = token_audit[" Yes"]["single_token"] and token_audit[" No"]["single_token"]
    answer_score_contract = "first_token_margin_permitted" if selected_single else "full_sequence_log_likelihood_required"
    source = implementation_source_record(args.repo_root)
    result = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "base_model_revision": pinned_revision,
        "dataset_audit_sha256": sha256_file(audit_path),
        "manifest_sha256": audit["manifest_sha256"],
        "annotation_sha256": audit["annotation_sha256"],
        "image_artifacts_verified": 66,
        "train_internal_test_overlap": 0,
        "explicit_label_alignment_verified": True,
        "fallback_sample_replacement_permitted": False,
        "tokenizer_audit": token_audit,
        "answer_score_contract": answer_score_contract,
        "implementation_source_fingerprint": source["fingerprint"],
        "implementation_source_files": source["files"],
        "patient_independence_verified": False,
        "claim_scope": "image_level_internal_engineering_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
