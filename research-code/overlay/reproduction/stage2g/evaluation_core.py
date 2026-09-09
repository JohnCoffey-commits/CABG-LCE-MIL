#!/usr/bin/env python3

import hashlib
import json
import math
import os
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image


QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."
TERMINAL_PUNCTUATION = frozenset(".!?。！？")
EXPECTED_EVALUATION_COUNTS = {"good": 87, "ungood": 141}
EXPECTED_EVALUATION_RECORDS = 228


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[Dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def normalize_binary_response(value: str) -> Tuple[str, Optional[str]]:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if text and text[-1] in TERMINAL_PUNCTUATION:
        text = text[:-1].rstrip()
    prediction = text if text in {"yes", "no"} else None
    return text, prediction


def validate_evaluation_manifest(
    records: List[Dict],
    expected_records: int = EXPECTED_EVALUATION_RECORDS,
    expected_counts: Optional[Dict[str, int]] = None,
) -> None:
    if expected_counts is None:
        expected_counts = EXPECTED_EVALUATION_COUNTS
    if len(records) != expected_records:
        raise RuntimeError(
            f"Evaluation manifest count mismatch: {len(records)} != {expected_records}"
        )
    sample_ids = [record.get("sample_id") for record in records]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Evaluation manifest sample IDs are missing or duplicated.")
    hashes = [record.get("sha256") for record in records]
    if any(not value for value in hashes) or len(set(hashes)) != len(hashes):
        raise RuntimeError("Evaluation manifest file hashes are missing or duplicated.")
    counts = {
        label: sum(record.get("label") == label for record in records)
        for label in ("good", "ungood")
    }
    if counts != expected_counts:
        raise RuntimeError(f"Evaluation manifest label counts mismatch: {counts}")
    expected_order = sorted(
        records,
        key=lambda record: (
            record["label"],
            record["sha256"],
            record["relative_path"],
        ),
    )
    if records != expected_order:
        raise RuntimeError("Evaluation manifest is not in the registered deterministic order.")
    for index, record in enumerate(records):
        expected_ground_truth = "yes" if record["label"] == "ungood" else "no"
        if record.get("ground_truth") != expected_ground_truth:
            raise RuntimeError(f"Ground-truth mapping mismatch at index {index}.")
        if record.get("sample_index") != index:
            raise RuntimeError(f"Sample index mismatch at index {index}.")


def compute_metrics(predictions: List[Dict]) -> Dict:
    if not predictions:
        raise RuntimeError("Cannot compute metrics for an empty prediction set.")
    positive_total = sum(record["ground_truth"] == "yes" for record in predictions)
    negative_total = sum(record["ground_truth"] == "no" for record in predictions)
    if positive_total == 0 or negative_total == 0:
        raise RuntimeError("Both positive and negative samples are required.")

    tp = sum(
        record["ground_truth"] == "yes" and record["prediction"] == "yes"
        for record in predictions
    )
    tn = sum(
        record["ground_truth"] == "no" and record["prediction"] == "no"
        for record in predictions
    )
    fp = sum(
        record["ground_truth"] == "no" and record["prediction"] == "yes"
        for record in predictions
    )
    fn = sum(
        record["ground_truth"] == "yes" and record["prediction"] == "no"
        for record in predictions
    )
    invalid_positive = sum(
        record["ground_truth"] == "yes" and record["prediction"] is None
        for record in predictions
    )
    invalid_negative = sum(
        record["ground_truth"] == "no" and record["prediction"] is None
        for record in predictions
    )
    invalid = invalid_positive + invalid_negative
    empty = sum(not record["normalized_response"] for record in predictions)

    sensitivity = tp / positive_total
    specificity = tn / negative_total
    balanced_accuracy = (sensitivity + specificity) / 2
    accuracy = (tp + tn) / len(predictions)
    f1_yes_denominator = 2 * tp + fp + fn + invalid_positive
    f1_no_denominator = 2 * tn + fn + fp + invalid_negative
    f1_yes = 2 * tp / f1_yes_denominator if f1_yes_denominator else 0.0
    f1_no = 2 * tn / f1_no_denominator if f1_no_denominator else 0.0
    metrics = {
        "total": len(predictions),
        "positive_total": positive_total,
        "negative_total": negative_total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid_positive": invalid_positive,
        "invalid_negative": invalid_negative,
        "invalid_responses": invalid,
        "invalid_response_rate": invalid / len(predictions),
        "empty_responses": empty,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "f1_yes": f1_yes,
        "f1_no": f1_no,
        "macro_f1": (f1_yes + f1_no) / 2,
    }
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key not in {
            "total",
            "positive_total",
            "negative_total",
            "tp",
            "tn",
            "fp",
            "fn",
            "invalid_positive",
            "invalid_negative",
            "invalid_responses",
            "empty_responses",
        }
    ):
        raise RuntimeError("A Stage 2G metric is non-finite.")
    return metrics


def run_image_evaluation(
    *,
    wrapper,
    data_root: Path,
    manifest_path: Path,
    predictions_path: Path,
    result_path: Path,
    run_metadata: Dict,
    generation: Callable,
    expected_manifest_sha256: str,
    final_metadata: Optional[Callable[[], Dict]] = None,
    expected_records: int = EXPECTED_EVALUATION_RECORDS,
    expected_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    for path in (predictions_path, result_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 2G evidence: {path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"Evaluation manifest hash mismatch: {manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = load_jsonl(manifest_path)
    validate_evaluation_manifest(
        manifest,
        expected_records=expected_records,
        expected_counts=expected_counts,
    )

    predictions: List[Dict] = []
    started = time.perf_counter()
    for source in manifest:
        image_path = data_root / source["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        observed_bytes = image_path.stat().st_size
        observed_sha256 = sha256_file(image_path)
        if observed_bytes != source["bytes"] or observed_sha256 != source["sha256"]:
            raise RuntimeError(f"Dataset artifact mismatch: {source['relative_path']}")
        sample_started = time.perf_counter()
        try:
            with Image.open(image_path) as image:
                if list(image.size) != source["dimensions"]:
                    raise RuntimeError(f"Image dimensions changed: {source['relative_path']}")
                image = image.convert("RGB")
                raw_response = generation(wrapper, image)
        except Exception as exc:
            partial = predictions_path.with_suffix(predictions_path.suffix + ".partial")
            if not partial.exists():
                write_jsonl(partial, predictions)
            raise RuntimeError(
                f"Stage 2G sample failed without retry: {source['relative_path']}: {exc}"
            ) from exc
        latency = time.perf_counter() - sample_started
        normalized_response, prediction = normalize_binary_response(raw_response)
        predictions.append(
            {
                "sample_index": source["sample_index"],
                "sample_id": source["sample_id"],
                "relative_path": source["relative_path"],
                "image_sha256": source["sha256"],
                "label": source["label"],
                "ground_truth": source["ground_truth"],
                "question": QUESTION,
                "raw_response": raw_response,
                "normalized_response": normalized_response,
                "prediction": prediction,
                "valid_response": prediction is not None,
                "correct": prediction == source["ground_truth"],
                "latency_seconds": latency,
                "error_status": None,
            }
        )

    elapsed = time.perf_counter() - started
    metrics = compute_metrics(predictions)
    result = {
        "status": "SUCCESS",
        "stage": "2G",
        "evaluation_role": "anomaly_specific_internal_pilot",
        "claim_scope": "image_level_internal_pilot_only",
        "patient_independence_verified": False,
        "clinical_effectiveness_evaluated": False,
        "paper_level_effectiveness_evaluated": False,
        "process_id": os.getpid(),
        "question": QUESTION,
        "question_sha256": sha256_text(QUESTION),
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": manifest_sha256,
        "evaluation_records": len(predictions),
        "generation_contract": {
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "do_sample": False,
            "max_new_tokens": 16,
            "tune_mode": "default",
            "diff_mode": False,
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            "min_pixels": 784,
            "max_pixels": 50176,
        },
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "predictions_jsonl": str(predictions_path),
        **run_metadata,
    }
    if final_metadata is not None:
        result.update(final_metadata())
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_path, predictions)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
