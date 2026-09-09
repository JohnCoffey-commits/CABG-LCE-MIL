#!/usr/bin/env python3

import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from qwenvl.data.data_qwen import (
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    expected_records = int(os.environ.get("MEDIC_AD_EXPECTED_RECORDS", "4"))
    expected_yes = int(os.environ.get("MEDIC_AD_EXPECTED_YES", "2"))
    expected_no = int(os.environ.get("MEDIC_AD_EXPECTED_NO", "2"))
    model_path = Path(
        os.environ.get("MEDIC_AD_LINGSHU_MODEL", "/home/checkpoints/Lingshu-7B")
    )
    annotation_path = Path(
        os.environ.get(
            "MEDIC_AD_SMOKE_ANNOTATION",
            "/home/data/medic-ad/training-smoke/stage1_smoke.json",
        )
    )
    image_root = Path(
        os.environ.get(
            "MEDIC_AD_SMOKE_DATA_ROOT",
            "/home/data/medic-ad/training-smoke/images",
        )
    )

    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != expected_records:
        raise ValueError(
            f"Stage 1 smoke annotation must contain exactly {expected_records} records."
        )

    labels = []
    image_hashes = {}
    for record in records:
        image_name = record["image"]
        image_path = image_root / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        conversations = record["conversations"]
        if len(conversations) != 2:
            raise ValueError(f"Expected two conversation turns for {record['id']}")
        prompt = conversations[0]["value"]
        answer = conversations[1]["value"]
        if prompt.count("<image>") != 1:
            raise ValueError(f"Expected one <image> tag for {record['id']}")
        if answer not in {"Yes", "No"}:
            raise ValueError(f"Unexpected label for {record['id']}: {answer}")
        labels.append(answer)
        image_hashes[image_name] = sha256_file(image_path)

    if labels.count("Yes") != expected_yes or labels.count("No") != expected_no:
        raise ValueError(
            f"Expected Yes/No counts {expected_yes}/{expected_no}, got {labels}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    data_args = SimpleNamespace(
        dataset_use="medic_ad_stage1_smoke",
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
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        use_anomaly_token=True,
        num_pooling_size=4,
    )
    if len(dataset) != expected_records:
        raise ValueError(f"Expected {expected_records} dataset items, got {len(dataset)}")

    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        load_masks=False,
    )
    item_summaries = []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = collator([item])
        grid = item["image_grid_thw"]
        base_visual_tokens = int(
            (grid.prod(dim=-1) // image_processor.merge_size**2).sum().item()
        )
        image_token_count = int(
            (item["input_ids"] == 151655).sum().item()
        )
        expected_image_tokens = base_visual_tokens + 16
        if image_token_count != expected_image_tokens:
            raise ValueError(
                "Image/anomaly token mismatch: "
                f"got {image_token_count}, expected {expected_image_tokens}"
            )
        if int((item["labels"] != -100).sum().item()) <= 0:
            raise ValueError("No supervised assistant tokens found.")
        item_summaries.append(
            {
                "index": index,
                "input_shape": list(item["input_ids"].shape),
                "pixel_shape": list(item["pixel_values"].shape),
                "grid": grid.tolist(),
                "base_visual_tokens": base_visual_tokens,
                "anomaly_tokens": 16,
                "batch_input_shape": list(batch["input_ids"].shape),
            }
        )

    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "model_path": str(model_path),
                "annotation_sha256": sha256_file(annotation_path),
                "image_hashes": image_hashes,
                "records": len(dataset),
                "labels": {"Yes": labels.count("Yes"), "No": labels.count("No")},
                "items": item_summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
