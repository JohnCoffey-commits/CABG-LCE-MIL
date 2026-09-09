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
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mode = normalize_query_mode(args.mode)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Engineering Gate verification: {args.output}")

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
            raise FileNotFoundError(f"Missing Engineering Gate evidence {name}: {path}")

    expected_scope = EXPECTED_SCOPE[mode]
    trainer_state = read_json(paths["trainer_state"])
    if trainer_state.get("global_step") != args.expected_steps:
        raise RuntimeError("Engineering training did not reach the expected global step.")
    training_records = [
        record
        for record in trainer_state.get("log_history", [])
        if "loss" in record and "grad_norm" in record
    ]
    if [record.get("step") for record in training_records] != list(range(1, args.expected_steps + 1)):
        raise RuntimeError("Engineering training log does not contain every expected step.")
    if not all(
        math.isfinite(float(record["loss"])) and math.isfinite(float(record["grad_norm"]))
        for record in training_records
    ):
        raise RuntimeError("Engineering training contains a non-finite loss or gradient norm.")

    loss_trace = read_jsonl(paths["loss"])
    if [record.get("global_step_before_update") for record in loss_trace] != list(range(args.expected_steps)):
        raise RuntimeError("Engineering unrounded loss trace has unexpected steps.")
    if not all(math.isfinite(float(record["loss"])) for record in loss_trace):
        raise RuntimeError("Engineering unrounded loss trace contains a non-finite value.")

    gradient_records = read_jsonl(paths["gradient"])
    gradient_start = [record for record in gradient_records if record.get("event") == "gradient_audit_start"]
    gradient_steps = [record for record in gradient_records if record.get("event") == "gradient_audit_step"]
    gradient_end = [record for record in gradient_records if record.get("event") == "gradient_audit_end"]
    if len(gradient_start) != 1 or len(gradient_end) != 1:
        raise RuntimeError("Engineering gradient audit is incomplete.")
    if [record.get("optimizer_step") for record in gradient_steps] != list(
        range(1, args.expected_steps + 1)
    ):
        raise RuntimeError("Engineering gradient audit steps are incomplete.")
    for record in (gradient_start[0], gradient_end[0]):
        if record.get("trainable_parameter_tensors") != expected_scope["tensors"]:
            raise RuntimeError("Gradient audit trainable tensor count mismatch.")
        if record.get("trainable_parameter_elements") != expected_scope["elements"]:
            raise RuntimeError("Gradient audit trainable element count mismatch.")
    if gradient_end[0].get("status") != "SUCCESS":
        raise RuntimeError("Engineering gradient audit did not finish successfully.")
    if gradient_end[0].get("missing_gradient") or gradient_end[0].get("never_finite") or gradient_end[0].get("nonfinite"):
        raise RuntimeError("Engineering gradient integrity audit reported a failure.")

    gate_gradient_evidence = {}
    expected_gate_suffixes = ("fusion_gate.weight", "fusion_gate.bias")
    for suffix in expected_gate_suffixes:
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
        if len(observations) != args.expected_steps:
            raise RuntimeError(f"A3 gate gradient evidence is incomplete: {suffix}")
        if not all(has_finite_nonzero_element(record) for record in observations):
            raise RuntimeError(f"A3 gate gradient is missing, non-finite, or zero: {suffix}")
        gate_gradient_evidence[suffix] = [float(record["norm"]) for record in observations]

    scope = read_json(paths["scope"])
    if scope.get("status") != "SUCCESS" or scope.get("anomaly_query_mode") != mode:
        raise RuntimeError("Trainable-scope audit did not succeed for the requested mode.")
    if scope.get("missing") or scope.get("unexpected"):
        raise RuntimeError("Trainable-scope audit reports a missing or unexpected parameter.")
    if scope.get("trainable_parameter_tensors") != expected_scope["tensors"]:
        raise RuntimeError("Trainable-scope tensor count mismatch.")
    if scope.get("trainable_parameter_elements") != expected_scope["elements"]:
        raise RuntimeError("Trainable-scope element count mismatch.")
    expected_gradient_names = set(scope.get("expected_names", []))
    for step in gradient_steps:
        parameters = step.get("parameters", [])
        if {record.get("name") for record in parameters} != expected_gradient_names:
            raise RuntimeError("Engineering gradient-audit names differ from trainable scope.")
        if not all(record.get("gradient_present") and record.get("finite") for record in parameters):
            raise RuntimeError("Engineering audited gradient is missing or non-finite.")

    adapter_manifest = read_json(paths["adapter_manifest"])
    if adapter_manifest.get("status") != "SUCCESS" or adapter_manifest.get("schema_version") != 2:
        raise RuntimeError("Adapter manifest is not a successful schema-v2 artifact.")
    if adapter_manifest.get("trainable_parameter_tensors") != expected_scope["tensors"]:
        raise RuntimeError("Adapter tensor count mismatch.")
    if adapter_manifest.get("trainable_parameter_elements") != expected_scope["elements"]:
        raise RuntimeError("Adapter element count mismatch.")
    adapter_names = sorted(record["name"] for record in adapter_manifest.get("parameters", []))
    if adapter_names != sorted(scope.get("expected_names", [])):
        raise RuntimeError("Adapter and audited trainable-scope keys differ.")
    metadata = adapter_manifest.get("metadata", {})
    expected_metadata = strict_adapter_metadata(mode)
    expected_metadata.update(
        {
            "seed": 42,
            "data_seed": 42,
            "global_step": args.expected_steps,
            "train_type": "default",
            "full_determinism": False,
            "run_id": args.run_id,
            "repeat": 1,
            "stage": "2F",
        }
    )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Adapter metadata mismatch for {key}: expected {expected!r}, got {metadata.get(key)!r}"
            )
    require_metadata_fields(metadata, REQUIRED_ADAPTER_FIELDS)
    implementation_fingerprint = paths["implementation_fingerprint"].read_text(encoding="utf-8").strip()
    if metadata.get("implementation_source_fingerprint") != implementation_fingerprint:
        raise RuntimeError("Adapter implementation-source fingerprint mismatch.")

    generation = read_json(paths["generation"])
    if generation.get("status") != "SUCCESS" or generation.get("evaluation_role") != "engineering_smoke":
        raise RuntimeError("Engineering generation result is not successful.")
    if generation.get("run_id") != args.run_id or generation.get("anomaly_query_mode") != mode:
        raise RuntimeError("Engineering generation identity or mode mismatch.")
    if generation.get("seed") != 42 or generation.get("repeat") != 1:
        raise RuntimeError("Engineering generation seed/repeat mismatch.")
    if generation.get("validation_records") != 1 or generation.get("metrics", {}).get("total") != 1:
        raise RuntimeError("Engineering generation must contain exactly one validation record.")
    if generation.get("metrics", {}).get("empty_responses") != 0:
        raise RuntimeError("Engineering generation produced an empty response.")
    if generation.get("adapter_sha256") != adapter_manifest.get("checkpoint_sha256"):
        raise RuntimeError("Engineering generation used an unexpected adapter.")
    if generation.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official VQA-RAD test data must remain unused.")
    predictions = read_jsonl(paths["predictions"])
    if len(predictions) != 1 or predictions[0].get("id") != generation["predictions"][0].get("id"):
        raise RuntimeError("Engineering prediction JSONL does not match the generation result.")

    training_peak_mib = int(paths["train_vram"].read_text(encoding="utf-8").strip())
    generation_peak_mib = int(paths["generation_vram"].read_text(encoding="utf-8").strip())
    if max(training_peak_mib, generation_peak_mib) > MAX_EXTERNAL_VRAM_MIB:
        raise RuntimeError(
            f"Engineering run exceeds the external VRAM guardrail: "
            f"train={training_peak_mib}, generation={generation_peak_mib} MiB"
        )
    training_elapsed = int(paths["train_elapsed"].read_text(encoding="utf-8").strip())
    generation_elapsed = int(paths["generation_elapsed"].read_text(encoding="utf-8").strip())
    if training_elapsed <= 0 or generation_elapsed <= 0:
        raise RuntimeError("Engineering elapsed-time evidence is invalid.")
    if list(args.output_root.glob("*.safetensors")):
        raise RuntimeError("Engineering output unexpectedly contains full-model safetensors.")

    result = {
        "status": "SUCCESS",
        "run_id": args.run_id,
        "anomaly_query_mode": mode,
        "global_step": args.expected_steps,
        "training_losses": [float(record["loss"]) for record in loss_trace],
        "gradient_audit_status": gradient_end[0]["status"],
        "gate_gradient_norms": gate_gradient_evidence,
        "trainable_parameter_tensors": expected_scope["tensors"],
        "trainable_parameter_elements": expected_scope["elements"],
        "trainable_scope_fingerprint": scope["scope_fingerprint"],
        "adapter_sha256": adapter_manifest["checkpoint_sha256"],
        "adapter_key_fingerprint": adapter_manifest["trainable_key_fingerprint"],
        "implementation_source_fingerprint": implementation_fingerprint,
        "generation_response_nonempty": True,
        "training_peak_vram_used_mib": training_peak_mib,
        "generation_peak_vram_used_mib": generation_peak_mib,
        "training_elapsed_seconds": training_elapsed,
        "generation_elapsed_seconds": generation_elapsed,
        "total_elapsed_seconds": training_elapsed + generation_elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
