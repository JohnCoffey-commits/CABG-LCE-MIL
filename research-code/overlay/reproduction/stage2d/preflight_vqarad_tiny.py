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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    model_path = Path(os.environ.get("MEDIC_AD_LINGSHU_MODEL", "/home/checkpoints/Lingshu-7B"))
    data_root = Path(
        os.environ.get(
            "MEDIC_AD_VQARAD_TINY_ROOT",
            "/home/data/medic-ad/training-tiny-vqarad",
        )
    )
    image_root = data_root / "images"
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCESS":
        raise RuntimeError("Prepared-data manifest is not successful.")
    if manifest.get("dataset_revision") != "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9":
        raise RuntimeError("Unexpected VQA-RAD dataset revision.")
    if manifest.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official test data must remain unused.")
    if manifest.get("train_validation_image_overlap") != []:
        raise RuntimeError("Manifest reports train/validation image overlap.")

    records_by_split = {
        split: json.loads((data_root / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "validation")
    }
    if len(records_by_split["train"]) != 8 or len(records_by_split["validation"]) != 4:
        raise RuntimeError("Expected 8 train and 4 validation records.")

    manifest_by_id = {record["id"]: record for record in manifest["records"]}
    split_hashes = {"train": set(), "validation": set()}
    split_answer_types = {"train": {"closed": 0, "open": 0}, "validation": {"closed": 0, "open": 0}}
    for split, records in records_by_split.items():
        for record in records:
            source = manifest_by_id[record["id"]]
            image_path = image_root / record["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            observed_hash = sha256_file(image_path)
            if observed_hash != source["local_image_sha256"]:
                raise RuntimeError(f"Image SHA mismatch: {image_path}")
            if source["derived_split"] != split:
                raise RuntimeError(f"Manifest split mismatch: {record['id']}")
            split_hashes[split].add(observed_hash)
            split_answer_types[split][source["answer_type"]] += 1
            conversations = record.get("conversations", [])
            if len(conversations) != 2 or conversations[0].get("value", "").count("<image>") != 1:
                raise RuntimeError(f"Invalid conversation structure: {record['id']}")
            if not str(conversations[1].get("value", "")).strip():
                raise RuntimeError(f"Empty answer: {record['id']}")

    if split_hashes["train"] & split_hashes["validation"]:
        raise RuntimeError("Train/validation image SHA overlap detected during preflight.")
    if split_answer_types != {
        "train": {"closed": 4, "open": 4},
        "validation": {"closed": 2, "open": 2},
    }:
        raise RuntimeError(f"Unexpected answer-type balance: {split_answer_types}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    data_args = SimpleNamespace(
        dataset_use="medic_ad_tiny_vqarad_train",
        eval_dataset_use="medic_ad_tiny_vqarad_validation",
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
    module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        use_anomaly_token=True,
        num_pooling_size=4,
    )
    train_dataset = module["train_dataset"]
    eval_dataset = module["eval_dataset"]
    collator = module["data_collator"]
    if len(train_dataset) != 8 or eval_dataset is None or len(eval_dataset) != 4:
        raise RuntimeError("Integrated train/eval dataset construction returned unexpected lengths.")

    summaries = {"train": [], "validation": []}
    for split, dataset in (("train", train_dataset), ("validation", eval_dataset)):
        for index in range(len(dataset)):
            item = dataset[index]
            batch = collator([item])
            grid = item["image_grid_thw"]
            base_visual_tokens = int(
                (grid.prod(dim=-1) // image_processor.merge_size**2).sum().item()
            )
            image_token_count = int((item["input_ids"] == 151655).sum().item())
            if image_token_count != base_visual_tokens + 16:
                raise RuntimeError(
                    f"Image/anomaly token mismatch in {split}[{index}]: "
                    f"{image_token_count} != {base_visual_tokens + 16}"
                )
            supervised_tokens = int((item["labels"] != -100).sum().item())
            if supervised_tokens <= 0:
                raise RuntimeError(f"No supervised tokens in {split}[{index}]")
            summaries[split].append(
                {
                    "index": index,
                    "input_tokens": int(item["input_ids"].numel()),
                    "base_visual_tokens": base_visual_tokens,
                    "anomaly_tokens": 16,
                    "supervised_tokens": supervised_tokens,
                    "batch_shape": list(batch["input_ids"].shape),
                }
            )

    result = {
        "status": "SUCCESS",
        "dataset_revision": manifest["dataset_revision"],
        "source_train_file_sha256": manifest["source_train_file_sha256"],
        "official_test_downloaded_or_used": False,
        "train_annotation_sha256": sha256_file(data_root / "train.json"),
        "validation_annotation_sha256": sha256_file(data_root / "validation.json"),
        "train_records": len(train_dataset),
        "validation_records": len(eval_dataset),
        "answer_types": split_answer_types,
        "train_validation_image_overlap": [],
        "items": summaries,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
