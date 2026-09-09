#!/usr/bin/env python3

import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from qwenvl.data.data_qwen import make_supervised_data_module
from reproduction.stage2e.generate_vqarad_metrics import closed_prediction, normalize_answer, token_f1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    model_path = Path(os.environ.get("MEDIC_AD_LINGSHU_MODEL", "/home/checkpoints/Lingshu-7B"))
    data_root = Path(os.environ.get("MEDIC_AD_BASELINE_VQARAD_ROOT", "/home/data/medic-ad/baseline-vqarad-v1"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9":
        raise RuntimeError("Unexpected dataset revision.")
    if manifest.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official test data must remain unused.")
    if manifest.get("train_validation_image_overlap") != []:
        raise RuntimeError("Manifest reports train/validation overlap.")

    records_by_split = {
        split: json.loads((data_root / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "validation")
    }
    if len(records_by_split["train"]) != 32 or len(records_by_split["validation"]) != 16:
        raise RuntimeError("Expected 32 train and 16 validation records.")
    for split in ("train", "validation"):
        if sha256_file(data_root / f"{split}.json") != manifest[f"{split}_annotation_sha256"]:
            raise RuntimeError(f"Annotation SHA mismatch: {split}")

    manifest_by_id = {record["id"]: record for record in manifest["records"]}
    hashes = {"train": set(), "validation": set()}
    counts = {"train": {"closed": 0, "open": 0}, "validation": {"closed": 0, "open": 0}}
    for split, records in records_by_split.items():
        for record in records:
            source = manifest_by_id[record["id"]]
            image_path = data_root / "images" / record["image"]
            if sha256_file(image_path) != source["local_image_sha256"]:
                raise RuntimeError(f"Image SHA mismatch: {record['id']}")
            hashes[split].add(source["source_image_sha256"])
            counts[split][source["answer_type"]] += 1
    if hashes["train"] & hashes["validation"]:
        raise RuntimeError("Train/validation image overlap detected.")
    if counts != {
        "train": {"closed": 16, "open": 16},
        "validation": {"closed": 8, "open": 8},
    }:
        raise RuntimeError(f"Unexpected answer-type balance: {counts}")

    if normalize_answer("Answer: YES.") != "yes":
        raise RuntimeError("Answer normalization self-test failed.")
    if closed_prediction("Yes.") != "yes" or closed_prediction("yes or no") is not None:
        raise RuntimeError("Closed-answer parser self-test failed.")
    if token_f1("left lung", "left lung") != 1.0 or token_f1("left lung", "right kidney") != 0.0:
        raise RuntimeError("Token-F1 self-test failed.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, padding_side="right", use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    data_args = SimpleNamespace(
        dataset_use="medic_ad_baseline_vqarad_train",
        eval_dataset_use="medic_ad_baseline_vqarad_validation",
        model_type="qwen2.5vl",
        image_processor=image_processor,
        max_pixels=50176,
        min_pixels=784,
        load_masks=False,
        use_diff_token=False,
        diff_only_mode=False,
        data_flatten=False,
        video_max_frames=8,
        video_min_frames=4,
        video_max_frame_pixels=32 * 28 * 28,
        video_min_frame_pixels=4 * 28 * 28,
        base_interval=2,
    )
    random.seed(42)
    torch.manual_seed(42)
    module = make_supervised_data_module(tokenizer, data_args, use_anomaly_token=True, num_pooling_size=4)
    train_dataset = module["train_dataset"]
    eval_dataset = module["eval_dataset"]
    collator = module["data_collator"]
    if len(train_dataset) != 32 or eval_dataset is None or len(eval_dataset) != 16:
        raise RuntimeError("Integrated dataset lengths are incorrect.")

    token_summary = {"train": [], "validation": []}
    for split, dataset in (("train", train_dataset), ("validation", eval_dataset)):
        for index in range(len(dataset)):
            item = dataset[index]
            batch = collator([item])
            grid = item["image_grid_thw"]
            base_visual_tokens = int((grid.prod(dim=-1) // image_processor.merge_size**2).sum().item())
            image_token_count = int((item["input_ids"] == 151655).sum().item())
            if image_token_count != base_visual_tokens + 16:
                raise RuntimeError(f"Anomaly-token mismatch: {split}[{index}]")
            supervised_tokens = int((item["labels"] != -100).sum().item())
            if supervised_tokens <= 0:
                raise RuntimeError(f"No supervised tokens: {split}[{index}]")
            token_summary[split].append(
                {
                    "index": index,
                    "input_tokens": int(item["input_ids"].numel()),
                    "base_visual_tokens": base_visual_tokens,
                    "supervised_tokens": supervised_tokens,
                    "batch_shape": list(batch["input_ids"].shape),
                }
            )

    result = {
        "status": "SUCCESS",
        "dataset_revision": manifest["dataset_revision"],
        "official_test_downloaded_or_used": False,
        "train_records": 32,
        "validation_records": 16,
        "answer_types": counts,
        "train_validation_image_overlap": [],
        "items": token_summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
