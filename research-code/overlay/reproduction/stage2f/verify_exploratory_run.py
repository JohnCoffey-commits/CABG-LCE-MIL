#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

from reproduction.stage2f.adapter_schema import (
    EXPECTED_SCOPE,
    normalize_query_mode,
    require_metadata_fields,
    strict_adapter_metadata,
)
from reproduction.stage2f.gradient_evidence import has_finite_nonzero_element


MAX_EXTERNAL_VRAM_MIB = 22_835
EXPECTED_STEPS = 32
REQUIRED_ADAPTER_FIELDS = (
    "base_model_revision",
    "base_checkpoint_fingerprint",
    "dataset_manifest_sha256",
    "implementation_source_fingerprint",
    "source_base_commit",
    "anomaly_query_mode",
    "output_token_count",
    "method_id",
    "gate_type",
    "run_id",
    "repeat",
    "stage",
    "seed",
    "data_seed",
    "global_step",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    parser.add_argument("--gradient-audit-interval", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mode = normalize_query_mode(args.mode)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite exploratory verification: {args.output}")

    paths = {
        "trainer_state": args.output_root / "trainer_state.json",
        "scope": args.log_root / "trainable-scope.json",
        "gradient": args.log_root / "gradient-audit.jsonl",
        "loss": args.log_root / "loss-trace.jsonl",
        "generation": args.log_root / "generation.json",
        "predictions": args.log_root / "predictions.jsonl",
        "train_vram": args.log_root / "training-peak-vram-used-mib.txt",
        "generation_vram": args.log_root / "generation-peak-vram-used-mib.txt",
        "train_elapsed": args.log_root / "training-elapsed-seconds.txt",
        "generation_elapsed": args.log_root / "generation-elapsed-seconds.txt",
        "implementation_fingerprint": args.log_root / "implementation-source-fingerprint.txt",
        "adapter": args.adapter,
        "adapter_manifest": args.adapter.with_suffix(".manifest.json"),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing exploratory evidence {name}: {path}")

    expected_scope = EXPECTED_SCOPE[mode]
    trainer_state = read_json(paths["trainer_state"])
    if trainer_state.get("global_step") != EXPECTED_STEPS:
        raise RuntimeError("Exploratory training did not reach step 32.")
    training_records = [
        record
        for record in trainer_state.get("log_history", [])
        if "loss" in record and "grad_norm" in record
    ]
    if [record.get("step") for record in training_records] != list(range(1, EXPECTED_STEPS + 1)):
        raise RuntimeError("Exploratory training log is missing a step.")
    if not all(
        math.isfinite(float(record["loss"])) and math.isfinite(float(record["grad_norm"]))
        for record in training_records
    ):
        raise RuntimeError("Exploratory training has a non-finite loss or gradient norm.")
    eval_records = [record for record in trainer_state.get("log_history", []) if "eval_loss" in record]
    if [record.get("step") for record in eval_records] != [16, 32]:
        raise RuntimeError("Exploratory evaluation must occur at steps 16 and 32.")
    if not all(math.isfinite(float(record["eval_loss"])) for record in eval_records):
        raise RuntimeError("Exploratory evaluation loss is non-finite.")

    loss_trace = read_jsonl(paths["loss"])
    if [record.get("global_step_before_update") for record in loss_trace] != list(range(EXPECTED_STEPS)):
        raise RuntimeError("Exploratory unrounded loss trace is incomplete.")
    if not all(math.isfinite(float(record["loss"])) for record in loss_trace):
        raise RuntimeError("Exploratory unrounded loss is non-finite.")

    gradient_records = read_jsonl(paths["gradient"])
    gradient_steps = [record for record in gradient_records if record.get("event") == "gradient_audit_step"]
    gradient_end = [record for record in gradient_records if record.get("event") == "gradient_audit_end"]
    if args.gradient_audit_interval <= 0:
        raise ValueError("Gradient audit interval must be positive.")
    expected_gradient_steps = list(
        range(args.gradient_audit_interval, EXPECTED_STEPS + 1, args.gradient_audit_interval)
    )
    if not expected_gradient_steps or expected_gradient_steps[-1] != EXPECTED_STEPS:
        raise ValueError("Gradient audit interval must include the final optimizer step.")
    if [record.get("optimizer_step") for record in gradient_steps] != expected_gradient_steps:
        raise RuntimeError("Exploratory gradient-audit checkpoints are incomplete.")
    if len(gradient_end) != 1 or gradient_end[0].get("status") != "SUCCESS":
        raise RuntimeError("Exploratory gradient audit did not succeed.")
    if gradient_end[0].get("trainable_parameter_tensors") != expected_scope["tensors"]:
        raise RuntimeError("Exploratory gradient tensor count mismatch.")
    if gradient_end[0].get("trainable_parameter_elements") != expected_scope["elements"]:
        raise RuntimeError("Exploratory gradient element count mismatch.")
    if gradient_end[0].get("missing_gradient") or gradient_end[0].get("never_finite") or gradient_end[0].get("nonfinite"):
        raise RuntimeError("Exploratory gradient integrity audit reported a failure.")

    gate_gradient_evidence = {}
    for suffix in ("fusion_gate.weight", "fusion_gate.bias"):
        observations = [
            parameter
            for step in gradient_steps
            for parameter in step.get("parameters", [])
            if parameter.get("name", "").endswith(suffix)
        ]
        if mode == "single":
            if observations:
                raise RuntimeError(f"B0 unexpectedly contains gate gradient evidence: {suffix}")
            continue
        if len(observations) != len(expected_gradient_steps) or not all(
            record.get("gradient_present") and record.get("finite")
            for record in observations
        ):
            raise RuntimeError(f"A3 exploratory gate gradient failed: {suffix}")
        if not any(has_finite_nonzero_element(record) for record in observations):
            raise RuntimeError(f"A3 exploratory gate gradient was never non-zero: {suffix}")
        gate_gradient_evidence[suffix] = [
            {
                "optimizer_step": step["optimizer_step"],
                "finite": bool(record["finite"]),
                "nonzero": bool(record["nonzero"]),
                "norm": float(record["norm"]),
                "max_abs": float(record["max_abs"]),
            }
            for step, record in zip(gradient_steps, observations)
        ]

    scope = read_json(paths["scope"])
    if (
        scope.get("status") != "SUCCESS"
        or scope.get("anomaly_query_mode") != mode
        or scope.get("missing")
        or scope.get("unexpected")
        or scope.get("trainable_parameter_tensors") != expected_scope["tensors"]
        or scope.get("trainable_parameter_elements") != expected_scope["elements"]
    ):
        raise RuntimeError("Exploratory trainable-scope audit is invalid.")
    expected_gradient_names = set(scope.get("expected_names", []))
    for step in gradient_steps:
        parameters = step.get("parameters", [])
        if {record.get("name") for record in parameters} != expected_gradient_names:
            raise RuntimeError("Exploratory gradient-audit parameter names differ from trainable scope.")
        if not all(record.get("gradient_present") and record.get("finite") for record in parameters):
            raise RuntimeError("Exploratory audited gradient is missing or non-finite at a checkpoint.")

    adapter_manifest = read_json(paths["adapter_manifest"])
    if adapter_manifest.get("status") != "SUCCESS" or adapter_manifest.get("schema_version") != 2:
        raise RuntimeError("Exploratory adapter is not a successful schema-v2 artifact.")
    if adapter_manifest.get("trainable_parameter_tensors") != expected_scope["tensors"]:
        raise RuntimeError("Exploratory adapter tensor count mismatch.")
    if adapter_manifest.get("trainable_parameter_elements") != expected_scope["elements"]:
        raise RuntimeError("Exploratory adapter element count mismatch.")
    adapter_names = sorted(record["name"] for record in adapter_manifest.get("parameters", []))
    if adapter_names != sorted(scope.get("expected_names", [])):
        raise RuntimeError("Exploratory adapter and scope keys differ.")
    metadata = adapter_manifest.get("metadata", {})
    expected_metadata = strict_adapter_metadata(mode)
    expected_metadata.update(
        {
            "seed": args.seed,
            "data_seed": args.seed,
            "global_step": EXPECTED_STEPS,
            "train_type": "default",
            "full_determinism": False,
            "run_id": args.run_id,
            "repeat": args.repeat,
            "stage": "2F",
        }
    )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"Exploratory adapter metadata mismatch: {key}")
    require_metadata_fields(metadata, REQUIRED_ADAPTER_FIELDS)
    implementation_fingerprint = paths["implementation_fingerprint"].read_text(encoding="utf-8").strip()
    if metadata.get("implementation_source_fingerprint") != implementation_fingerprint:
        raise RuntimeError("Exploratory implementation-source fingerprint mismatch.")

    generation = read_json(paths["generation"])
    if generation.get("status") != "SUCCESS" or generation.get("evaluation_role") != "exploratory_vqa_guardrail":
        raise RuntimeError("Exploratory generation result is invalid.")
    if (
        generation.get("run_id") != args.run_id
        or generation.get("anomaly_query_mode") != mode
        or generation.get("seed") != args.seed
        or generation.get("repeat") != args.repeat
    ):
        raise RuntimeError("Exploratory generation identity mismatch.")
    metrics = generation.get("metrics", {})
    metric_keys = (
        "overall_exact_match_accuracy",
        "closed_accuracy",
        "open_exact_match",
        "open_token_f1",
    )
    if generation.get("validation_records") != 16 or metrics.get("total") != 16:
        raise RuntimeError("Exploratory generation must contain 16 samples.")
    if not all(math.isfinite(float(metrics[key])) for key in metric_keys):
        raise RuntimeError("Exploratory VQA metric is non-finite.")
    if generation.get("adapter_sha256") != adapter_manifest.get("checkpoint_sha256"):
        raise RuntimeError("Exploratory evaluator used an unexpected adapter.")
    if generation.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official VQA-RAD test data must remain unused.")
    predictions = read_jsonl(paths["predictions"])
    prediction_ids = [record.get("id") for record in predictions]
    if len(predictions) != 16 or len(set(prediction_ids)) != 16:
        raise RuntimeError("Exploratory prediction JSONL is incomplete or duplicated.")
    if prediction_ids != [record.get("id") for record in generation.get("predictions", [])]:
        raise RuntimeError("Exploratory prediction JSONL order differs from the result JSON.")

    training_peak_mib = int(paths["train_vram"].read_text(encoding="utf-8").strip())
    generation_peak_mib = int(paths["generation_vram"].read_text(encoding="utf-8").strip())
    if max(training_peak_mib, generation_peak_mib) > MAX_EXTERNAL_VRAM_MIB:
        raise RuntimeError("Exploratory run exceeds the external VRAM guardrail.")
    training_elapsed = int(paths["train_elapsed"].read_text(encoding="utf-8").strip())
    generation_elapsed = int(paths["generation_elapsed"].read_text(encoding="utf-8").strip())
    if training_elapsed <= 0 or generation_elapsed <= 0:
        raise RuntimeError("Exploratory elapsed-time evidence is invalid.")
    if list(args.output_root.glob("*.safetensors")):
        raise RuntimeError("Exploratory output unexpectedly contains full-model safetensors.")

    result = {
        "status": "SUCCESS",
        "run_id": args.run_id,
        "seed": args.seed,
        "repeat": args.repeat,
        "method": "B0" if mode == "single" else "A3",
        "anomaly_query_mode": mode,
        "training_losses": [float(record["loss"]) for record in loss_trace],
        "eval_steps": [16, 32],
        "eval_losses": [float(record["eval_loss"]) for record in eval_records],
        "gate_gradient_norms": gate_gradient_evidence,
        "sampled_zero_gradient_parameters": gradient_end[0].get("never_nonzero", []),
        "trainable_parameter_tensors": expected_scope["tensors"],
        "trainable_parameter_elements": expected_scope["elements"],
        "adapter_sha256": adapter_manifest["checkpoint_sha256"],
        "adapter_key_fingerprint": adapter_manifest["trainable_key_fingerprint"],
        "implementation_source_fingerprint": implementation_fingerprint,
        "metrics": metrics,
        "training_peak_vram_used_mib": training_peak_mib,
        "generation_peak_vram_used_mib": generation_peak_mib,
        "training_elapsed_seconds": training_elapsed,
        "generation_elapsed_seconds": generation_elapsed,
        "total_elapsed_seconds": training_elapsed + generation_elapsed,
        "prediction_ids": prediction_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
