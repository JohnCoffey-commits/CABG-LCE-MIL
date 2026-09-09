#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

from reproduction.stage2g.evaluation_core import (
    QUESTION,
    compute_metrics,
    load_jsonl,
    normalize_binary_response,
    sha256_file,
    sha256_text,
    validate_evaluation_manifest,
)


MAX_EXTERNAL_VRAM_MIB = 22_835


def compare_metrics(expected, observed):
    if expected.keys() != observed.keys():
        raise RuntimeError("Metric key set mismatch.")
    for key in expected:
        if isinstance(expected[key], float):
            if not math.isclose(float(expected[key]), float(observed[key]), rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"Metric mismatch for {key}: {observed[key]} != {expected[key]}")
        elif observed[key] != expected[key]:
            raise RuntimeError(f"Metric mismatch for {key}: {observed[key]} != {expected[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--external-vram", type=Path, required=True)
    parser.add_argument("--wall-elapsed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite run verification: {args.output}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    matching = [run for run in registry["runs"] if run["run_id"] == args.run_id]
    if len(matching) != 1:
        raise RuntimeError(f"Run ID is absent or duplicated in registry: {args.run_id}")
    registered = matching[0]
    result = json.loads(args.result.read_text(encoding="utf-8"))
    predictions = load_jsonl(args.predictions)
    manifest = load_jsonl(args.evaluation_manifest)
    validate_evaluation_manifest(manifest)

    if result.get("status") != "SUCCESS" or result.get("run_id") != args.run_id:
        raise RuntimeError("Stage 2G result status/identity mismatch.")
    if result.get("run_role") != registered["run_role"] or result.get("method") != registered["method"]:
        raise RuntimeError("Stage 2G result role/method mismatch.")
    if result.get("evaluation_manifest_sha256") != sha256_file(args.evaluation_manifest):
        raise RuntimeError("Stage 2G result manifest hash mismatch.")
    if result.get("evaluation_records") != 228 or len(predictions) != 228:
        raise RuntimeError("Stage 2G run must contain exactly 228 predictions.")
    if result.get("question") != QUESTION or result.get("question_sha256") != sha256_text(QUESTION):
        raise RuntimeError("Stage 2G question contract mismatch.")
    if (
        result.get("claim_scope") != "image_level_internal_pilot_only"
        or result.get("patient_independence_verified") is not False
        or result.get("clinical_effectiveness_evaluated") is not False
        or result.get("paper_level_effectiveness_evaluated") is not False
    ):
        raise RuntimeError("Stage 2G claim boundary is invalid.")
    expected_generation = {
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
    }
    if result.get("generation_contract") != expected_generation:
        raise RuntimeError("Stage 2G generation contract mismatch.")

    for source, prediction in zip(manifest, predictions):
        if prediction.get("sample_id") != source["sample_id"]:
            raise RuntimeError("Prediction order/sample identity mismatch.")
        if prediction.get("image_sha256") != source["sha256"]:
            raise RuntimeError("Prediction image hash mismatch.")
        if prediction.get("ground_truth") != source["ground_truth"]:
            raise RuntimeError("Prediction ground truth mismatch.")
        normalized, parsed = normalize_binary_response(prediction.get("raw_response", ""))
        if normalized != prediction.get("normalized_response") or parsed != prediction.get("prediction"):
            raise RuntimeError("Prediction normalization/parser mismatch.")
        if prediction.get("valid_response") != (parsed is not None):
            raise RuntimeError("Prediction validity flag mismatch.")
        if prediction.get("correct") != (parsed == source["ground_truth"]):
            raise RuntimeError("Prediction correctness flag mismatch.")
        if prediction.get("error_status") is not None:
            raise RuntimeError("Successful run contains an error-status record.")
        if not math.isfinite(float(prediction.get("latency_seconds", -1))) or prediction[
            "latency_seconds"
        ] < 0:
            raise RuntimeError("Prediction latency is invalid.")
    recomputed_metrics = compute_metrics(predictions)
    compare_metrics(recomputed_metrics, result.get("metrics", {}))

    if registered["run_role"] == "paired_method":
        for key in ("method", "mode", "seed", "repeat", "adapter_train_run_id"):
            result_key = "anomaly_query_mode" if key == "mode" else key
            if result.get(result_key) != registered[key]:
                raise RuntimeError(f"Paired-method result mismatch: {key}")
        if result.get("adapter_sha256") != registered["adapter_sha256"]:
            raise RuntimeError("Paired-method adapter hash mismatch.")
        if result.get("evaluation_source_commit") != registered["source_commit"]:
            raise RuntimeError("Paired-method source commit mismatch.")
        if result.get("evaluation_source_fingerprint") != registered["source_fingerprint"]:
            raise RuntimeError("Paired-method source fingerprint mismatch.")
    else:
        if result.get("model_revision") != registered["model_revision"]:
            raise RuntimeError("R0 model revision mismatch.")
        if result.get("evaluation_source_commit") != registered["source_commit"]:
            raise RuntimeError("R0 source commit mismatch.")
        if result.get("excluded_from_fb_maq_decision") is not True:
            raise RuntimeError("R0 must be excluded from the FB-MAQ decision.")

    external_peak = int(args.external_vram.read_text(encoding="utf-8").strip())
    if external_peak <= 0 or external_peak > MAX_EXTERNAL_VRAM_MIB:
        raise RuntimeError(f"External Peak VRAM is invalid or exceeds guardrail: {external_peak}")
    wall_elapsed = float(args.wall_elapsed.read_text(encoding="utf-8").strip())
    if not math.isfinite(wall_elapsed) or wall_elapsed <= 0:
        raise RuntimeError("Run wall-clock elapsed time is invalid.")
    verification = {
        "status": "SUCCESS",
        "stage": "2G",
        "run_id": args.run_id,
        "run_role": registered["run_role"],
        "method": registered["method"],
        "evaluation_records": 228,
        "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
        "result_sha256": sha256_file(args.result),
        "predictions_sha256": sha256_file(args.predictions),
        "registry_sha256": sha256_file(args.registry),
        "metrics": recomputed_metrics,
        "external_peak_vram_used_mib": external_peak,
        "wall_elapsed_seconds": wall_elapsed,
        "process_id": result.get("process_id"),
        "fresh_process_evidence": True,
    }
    args.output.write_text(
        json.dumps(verification, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
