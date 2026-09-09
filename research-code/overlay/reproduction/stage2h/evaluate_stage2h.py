#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from reproduction.stage2g.evaluation_core import normalize_binary_response
from reproduction.stage2h.adapter_schema import expected_stage2h_metadata, load_stage2h_adapter
from reproduction.stage2h.evidence import compute_evidence_margin_loss
from reproduction.stage2h.runtime import load_stage2h_runtime
from reproduction.stage2h.state_audit import trainable_state_record


QUESTION = "Is there any anomaly in the image?\nAnswer the question using a single word or phrase."
ANO_ATOL = 1e-3
ANO_RTOL = 1e-3
EVIDENCE_ATOL = 1e-6
EVIDENCE_RTOL = 1e-6


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_model_runtime_state(model):
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None


def generation_forward(wrapper, image):
    reset_model_runtime_state(wrapper.llm)
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        response, _ = wrapper.generate_output(
            {"prompt": QUESTION, "image": image},
            tune_mode="default",
            diff_mode=False,
        )
    synchronize()
    return {
        "response": response,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }


def evidence_forward(wrapper, image, label):
    reset_model_runtime_state(wrapper.llm)
    inputs = wrapper.process_messages({"prompt": QUESTION, "image": image}, use_diff_token=False)
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = wrapper.llm(
            **inputs,
            tune_mode="default",
            diff_mode=False,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_anomaly_evidence=True,
        )
    synchronize()
    wall = time.perf_counter() - started
    evidence_result = compute_evidence_margin_loss(
        outputs.anomaly_evidence,
        torch.tensor([label], device=outputs.anomaly_evidence.device, dtype=torch.long),
    )
    evidence = outputs.anomaly_evidence.detach().float().cpu()
    anomaly_tokens = outputs.anomaly_tokens.detach().float().cpu()
    return {
        "wall_seconds": wall,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "evidence": evidence,
        "anomaly_tokens": anomaly_tokens,
        "score": float(evidence_result["image_scores"].detach().cpu().item()),
        "loss": float(evidence_result["loss"].detach().cpu().item()),
        "evidence_min": float(evidence.min().item()),
        "evidence_max": float(evidence.max().item()),
        "evidence_mean": float(evidence.mean().item()),
        "saturation_count": int(evidence_result["saturation_count"].cpu().item()),
        "saturation_total": int(evidence_result["saturation_total"].cpu().item()),
    }


def calculate_metrics(records):
    positive = [row for row in records if row["ground_truth"] == "yes"]
    negative = [row for row in records if row["ground_truth"] == "no"]
    tp = sum(row["prediction"] == "yes" for row in positive)
    tn = sum(row["prediction"] == "no" for row in negative)
    fp = sum(row["prediction"] == "yes" for row in negative)
    fn = sum(row["prediction"] == "no" for row in positive)
    invalid = sum(row["prediction"] is None for row in records)
    empty = sum(not row["normalized_response"] for row in records)
    sensitivity = tp / len(positive)
    specificity = tn / len(negative)
    return {
        "total": len(records),
        "positive_total": len(positive),
        "negative_total": len(negative),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid_responses": invalid,
        "invalid_rate": invalid / len(records),
        "empty_responses": empty,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": (tp + tn) / len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--expected-evaluation-manifest-sha256", required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--implementation-source-fingerprint", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--base-checkpoint-fingerprint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method-id", choices=("medic-ad-b0-as", "lad-mil-v2"), required=True)
    parser.add_argument("--evidence-loss-weight", type=float, required=True)
    parser.add_argument("--protocol-version", default="2.1")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.predictions.exists() or args.output.exists():
        raise FileExistsError("Refusing to overwrite Stage 2H evaluation evidence.")
    observed_manifest_hash = sha256_file(args.evaluation_manifest)
    if observed_manifest_hash != args.expected_evaluation_manifest_sha256:
        raise RuntimeError("Stage 2H evaluation manifest hash mismatch.")
    manifest = read_jsonl(args.evaluation_manifest)
    if len(manifest) != 46:
        raise RuntimeError("Stage 2H internal test requires exactly 46 records.")
    if [row.get("stage2h_split_index") for row in manifest] != list(range(46)):
        raise RuntimeError("Stage 2H internal-test manifest order/index is not locked.")
    if sum(row["label"] == "good" for row in manifest) != 23 or sum(
        row["label"] == "ungood" for row in manifest
    ) != 23:
        raise RuntimeError("Stage 2H internal-test class balance changed.")

    wrapper, runtime_audit = load_stage2h_runtime(args.base_model, seed=42, training=False)
    expected_metadata = expected_stage2h_metadata(
        method_id=args.method_id,
        evidence_loss_weight=args.evidence_loss_weight,
        base_model_revision=args.base_model_revision,
        base_checkpoint_fingerprint=args.base_checkpoint_fingerprint,
        dataset_manifest_sha256=args.training_manifest_sha256,
        training_manifest_sha256=args.training_manifest_sha256,
        evaluation_manifest_sha256=args.expected_evaluation_manifest_sha256,
        implementation_source_fingerprint=args.implementation_source_fingerprint,
        run_id=args.run_id,
        protocol_version=args.protocol_version,
    )
    loaded_manifest = load_stage2h_adapter(
        wrapper.llm,
        str(args.adapter),
        expected_metadata=expected_metadata,
    )
    before_evaluation_state = trainable_state_record(wrapper.llm)

    probe_record = manifest[0]
    probe_path = args.data_root / probe_record["relative_path"]
    with Image.open(probe_path) as source_image:
        probe_image = source_image.convert("RGB")
        before_generation = generation_forward(wrapper, probe_image)
        before_evidence = evidence_forward(
            wrapper, probe_image, 0 if probe_record["label"] == "good" else 1
        )
        reloaded_manifest = load_stage2h_adapter(
            wrapper.llm,
            str(args.adapter),
            expected_metadata=expected_metadata,
        )
        after_generation = generation_forward(wrapper, probe_image)
        after_evidence = evidence_forward(
            wrapper, probe_image, 0 if probe_record["label"] == "good" else 1
        )
    if loaded_manifest["checkpoint_sha256"] != reloaded_manifest["checkpoint_sha256"]:
        raise RuntimeError("Repeated strict reload returned a different adapter identity.")
    if before_generation["response"] != after_generation["response"]:
        raise RuntimeError("Greedy generation changed after strict reload.")
    torch.testing.assert_close(
        before_evidence["anomaly_tokens"],
        after_evidence["anomaly_tokens"],
        atol=ANO_ATOL,
        rtol=ANO_RTOL,
    )
    torch.testing.assert_close(
        before_evidence["evidence"],
        after_evidence["evidence"],
        atol=EVIDENCE_ATOL,
        rtol=EVIDENCE_RTOL,
    )
    if not math.isclose(
        before_evidence["score"],
        after_evidence["score"],
        rel_tol=EVIDENCE_RTOL,
        abs_tol=EVIDENCE_ATOL,
    ):
        raise RuntimeError("Evidence score changed after strict reload.")

    records = []
    max_allocated = 0.0
    max_reserved = 0.0
    generation_total = 0.0
    evidence_total = 0.0
    saturation_count = 0
    saturation_total = 0
    started = time.perf_counter()
    for index, source in enumerate(manifest):
        image_path = args.data_root / source["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.stat().st_size != source["bytes"] or sha256_file(image_path) != source["sha256"]:
            raise RuntimeError(f"Stage 2H internal-test image artifact mismatch: {source['relative_path']}")
        with Image.open(image_path) as image_source:
            if list(image_source.size) != source["dimensions"]:
                raise RuntimeError(f"Stage 2H image dimensions changed: {source['relative_path']}")
            image = image_source.convert("RGB")
            generated = generation_forward(wrapper, image)
            label = 0 if source["label"] == "good" else 1
            evidence = evidence_forward(wrapper, image, label)
        normalized, prediction = normalize_binary_response(generated["response"])
        ground_truth = "no" if label == 0 else "yes"
        if not all(
            math.isfinite(value)
            for value in (
                generated["wall_seconds"],
                evidence["wall_seconds"],
                evidence["score"],
                evidence["loss"],
                evidence["evidence_min"],
                evidence["evidence_max"],
                evidence["evidence_mean"],
            )
        ):
            raise RuntimeError(f"Stage 2H produced a non-finite evaluation value at index {index}.")
        generation_total += generated["wall_seconds"]
        evidence_total += evidence["wall_seconds"]
        max_allocated = max(max_allocated, generated["peak_allocated_mib"], evidence["peak_allocated_mib"])
        max_reserved = max(max_reserved, generated["peak_reserved_mib"], evidence["peak_reserved_mib"])
        saturation_count += evidence["saturation_count"]
        saturation_total += evidence["saturation_total"]
        records.append(
            {
                "stage2h_split_index": source["stage2h_split_index"],
                "sample_id": source["sample_id"],
                "relative_path": source["relative_path"],
                "image_sha256": source["sha256"],
                "anomaly_label": label,
                "ground_truth": ground_truth,
                "raw_response": generated["response"],
                "normalized_response": normalized,
                "prediction": prediction,
                "valid_response": prediction is not None,
                "correct": prediction == ground_truth,
                "generation_wall_seconds": generated["wall_seconds"],
                "generation_peak_allocated_mib": generated["peak_allocated_mib"],
                "generation_peak_reserved_mib": generated["peak_reserved_mib"],
                "evidence_extraction_wall_seconds": evidence["wall_seconds"],
                "evidence_extraction_peak_allocated_mib": evidence["peak_allocated_mib"],
                "evidence_extraction_peak_reserved_mib": evidence["peak_reserved_mib"],
                "evidence_score": evidence["score"],
                "evidence_loss": evidence["loss"],
                "evidence_min": evidence["evidence_min"],
                "evidence_max": evidence["evidence_max"],
                "evidence_mean": evidence["evidence_mean"],
                "saturation_count": evidence["saturation_count"],
                "saturation_total": evidence["saturation_total"],
            }
        )
    total_wall = time.perf_counter() - started
    after_evaluation_state = trainable_state_record(wrapper.llm)
    if before_evaluation_state["trainable_state_fingerprint"] != after_evaluation_state["trainable_state_fingerprint"]:
        raise RuntimeError("Read-only Stage 2H evaluation changed adapter parameters.")
    metrics = calculate_metrics(records)
    result = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "method_id": args.method_id,
        "run_id": args.run_id,
        "claim_scope": "image_level_internal_engineering_only",
        "patient_independence_verified": False,
        "adapter": str(args.adapter),
        "adapter_sha256": loaded_manifest["checkpoint_sha256"],
        "adapter_manifest_sha256": sha256_file(args.adapter.with_suffix(".manifest.json")),
        "runtime_audit": runtime_audit,
        "strict_reload_parity": {
            "status": "SUCCESS",
            "adapter_identity_exact": True,
            "greedy_generation_exact": True,
            "anomaly_tokens_close": True,
            "anomaly_tokens_atol": ANO_ATOL,
            "anomaly_tokens_rtol": ANO_RTOL,
            "evidence_close": True,
            "evidence_atol": EVIDENCE_ATOL,
            "evidence_rtol": EVIDENCE_RTOL,
            "evidence_score_close": True,
            "probe_response": before_generation["response"],
        },
        "read_only_evaluation_state_unchanged": True,
        "evaluation_only_second_forward": True,
        "generation_and_evidence_forward_separately_timed": True,
        "records": len(records),
        "metrics": metrics,
        "generation_wall_seconds_total": generation_total,
        "generation_wall_seconds_mean": generation_total / len(records),
        "evidence_extraction_wall_seconds_total": evidence_total,
        "evidence_extraction_wall_seconds_mean": evidence_total / len(records),
        "total_wall_seconds": total_wall,
        "peak_cuda_memory_allocated_mib": max_allocated,
        "peak_cuda_memory_reserved_mib": max_reserved,
        "evidence_saturation": {
            "numerator": saturation_count,
            "denominator": saturation_total,
            "proportion": saturation_count / saturation_total,
        },
        "evaluation_manifest_sha256": observed_manifest_hash,
        "predictions_jsonl": str(args.predictions),
    }
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.predictions, records)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
