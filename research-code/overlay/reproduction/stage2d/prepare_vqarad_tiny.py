#!/usr/bin/env python3

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


DATASET_ID = "flaviagiammarino/vqa-rad"
REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
TRAIN_FILE = "data/train-00000-of-00001-eb8844602202be60.parquet"
TRAIN_FILE_BYTES = 24_183_983
TRAIN_FILE_SHA256 = "b07c3441467b99060e5ec412ddd05be06f86f01f23bfa3debfbbcab47874a06e"
EXPECTED_TRAIN_ROWS = 1_793
SELECTION_SALT = "medic-ad-stage2d-vqarad-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_score(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{value}".encode("utf-8")).hexdigest()


def extract_image_bytes(image_field) -> bytes:
    if not isinstance(image_field, dict):
        raise TypeError(f"Unexpected image field type: {type(image_field)}")
    data = image_field.get("bytes")
    if not isinstance(data, bytes) or not data:
        raise ValueError("Parquet image field does not contain embedded bytes.")
    return data


def choose_row(rows, answer_type: str):
    if answer_type == "closed":
        eligible = [row for row in rows if row["answer"].lower() in {"yes", "no"}]
    else:
        eligible = [row for row in rows if row["answer"].lower() not in {"yes", "no"}]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: deterministic_score(
            f"{row['source_row_index']}:{row['question']}:{row['answer']}"
        ),
    )


def select_candidates(grouped_rows):
    ordered_hashes = sorted(grouped_rows, key=deterministic_score)
    selected = []
    used_image_hashes = set()
    for answer_type in ("closed", "open"):
        category = []
        for image_sha256 in ordered_hashes:
            if image_sha256 in used_image_hashes:
                continue
            row = choose_row(grouped_rows[image_sha256], answer_type)
            if row is None:
                continue
            category.append(row)
            used_image_hashes.add(image_sha256)
            if len(category) == 6:
                break
        if len(category) != 6:
            raise RuntimeError(f"Could not select six unique {answer_type} samples.")
        for index, row in enumerate(category):
            selected.append(
                {
                    **row,
                    "answer_type": answer_type,
                    "derived_split": "train" if index < 4 else "validation",
                    "category_rank": index,
                }
            )
    return selected


def main() -> None:
    output_root = Path(
        os.environ.get(
            "MEDIC_AD_VQARAD_TINY_ROOT",
            "/home/data/medic-ad/training-tiny-vqarad",
        )
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output_root}")
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=TRAIN_FILE,
        )
    )
    if parquet_path.stat().st_size != TRAIN_FILE_BYTES:
        raise RuntimeError(
            f"Unexpected parquet size: {parquet_path.stat().st_size}; expected {TRAIN_FILE_BYTES}"
        )
    observed_parquet_sha256 = sha256_file(parquet_path)
    if observed_parquet_sha256 != TRAIN_FILE_SHA256:
        raise RuntimeError(
            f"Unexpected parquet SHA-256: {observed_parquet_sha256}; expected {TRAIN_FILE_SHA256}"
        )

    table = pq.read_table(parquet_path)
    if table.num_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"Unexpected train row count: {table.num_rows}")
    required_columns = {"image", "question", "answer"}
    if not required_columns.issubset(table.column_names):
        raise RuntimeError(f"Missing parquet columns: {required_columns - set(table.column_names)}")

    grouped_rows = {}
    for source_row_index, raw_row in enumerate(table.to_pylist()):
        image_bytes = extract_image_bytes(raw_row["image"])
        image_sha256 = sha256_bytes(image_bytes)
        question = str(raw_row["question"]).strip()
        answer = str(raw_row["answer"]).strip()
        if not question or not answer:
            continue
        grouped_rows.setdefault(image_sha256, []).append(
            {
                "source_row_index": source_row_index,
                "source_image_sha256": image_sha256,
                "image_bytes": image_bytes,
                "question": question,
                "answer": answer,
            }
        )

    selected = select_candidates(grouped_rows)
    annotations = {"train": [], "validation": []}
    manifest_records = []
    for sample in selected:
        split = sample["derived_split"]
        filename = (
            f"vqarad_{split}_{sample['answer_type']}_{sample['category_rank']:02d}_"
            f"{sample['source_image_sha256'][:12]}.jpg"
        )
        image_path = image_root / filename
        image_path.write_bytes(sample["image_bytes"])
        with Image.open(BytesIO(sample["image_bytes"])) as image:
            image.verify()
        record_id = filename.removesuffix(".jpg")
        annotations[split].append(
            {
                "id": record_id,
                "image": filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"<image>\n{sample['question']}\n"
                            "Answer the question using a single word or phrase."
                        ),
                    },
                    {"from": "gpt", "value": sample["answer"]},
                ],
            }
        )
        manifest_records.append(
            {
                "id": record_id,
                "derived_split": split,
                "answer_type": sample["answer_type"],
                "source_row_index": sample["source_row_index"],
                "source_image_sha256": sample["source_image_sha256"],
                "local_image_sha256": sha256_file(image_path),
                "question": sample["question"],
                "answer": sample["answer"],
            }
        )

    train_hashes = {
        record["source_image_sha256"]
        for record in manifest_records
        if record["derived_split"] == "train"
    }
    validation_hashes = {
        record["source_image_sha256"]
        for record in manifest_records
        if record["derived_split"] == "validation"
    }
    if len(train_hashes) != 8 or len(validation_hashes) != 4:
        raise RuntimeError("Expected 8 unique train images and 4 unique validation images.")
    if train_hashes & validation_hashes:
        raise RuntimeError("Train/validation image SHA overlap detected.")

    for split in ("train", "validation"):
        annotation_path = output_root / f"{split}.json"
        annotation_path.write_text(
            json.dumps(annotations[split], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "status": "SUCCESS",
        "dataset_id": DATASET_ID,
        "dataset_revision": REVISION,
        "license": "CC0-1.0",
        "official_source": "https://osf.io/89kps/",
        "repack_source": "https://huggingface.co/datasets/flaviagiammarino/vqa-rad",
        "source_split_used": "train",
        "official_test_downloaded_or_used": False,
        "source_train_file": TRAIN_FILE,
        "source_train_file_bytes": TRAIN_FILE_BYTES,
        "source_train_file_sha256": observed_parquet_sha256,
        "source_train_rows": table.num_rows,
        "unique_source_images_in_train_repack": len(grouped_rows),
        "selection_salt": SELECTION_SALT,
        "selection_policy": "4 closed + 4 open train; 2 closed + 2 open derived-validation; unique image SHA",
        "train_records": len(annotations["train"]),
        "validation_records": len(annotations["validation"]),
        "train_annotation_sha256": sha256_file(output_root / "train.json"),
        "validation_annotation_sha256": sha256_file(output_root / "validation.json"),
        "train_validation_image_overlap": [],
        "records": sorted(manifest_records, key=lambda record: record["id"]),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
