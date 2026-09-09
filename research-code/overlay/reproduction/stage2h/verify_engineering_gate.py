#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


TARGET_PREFIXES = (
    "model.visual.anomaly_qformer.abnormal_prompt",
    "model.visual.anomaly_qformer.normal_prompt",
    "model.visual.anomaly_qformer.anomaly_attention.query_proj.",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def external_peak(path):
    value = float(path.read_text(encoding="utf-8").strip())
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"Invalid external VRAM peak: {path}")
    return value


def check_run(root, method, expected_weight, *, protocol_version="2.1"):
    run = root / "runs" / method
    evidence_trace = load_jsonl(run / "evidence-trace.jsonl")
    step_timing = load_jsonl(run / "step-timing.jsonl")
    gradient = load_jsonl(run / "gradient-audit.jsonl")
    scope = load_json(run / "trainable-scope.json")
    memory = load_json(run / "training-memory.json")
    change = load_json(run / "parameter-change.json")
    adapter_manifest = load_json(run / "adapter.manifest.json")
    evaluation = load_json(run / "evaluation.json")
    initial_manifest = load_json(run / "initial.manifest.json")
    failures = []
    if len(evidence_trace) != 2:
        failures.append(f"{method}: evidence trace has {len(evidence_trace)} records, expected 2")
    for record in evidence_trace:
        numerical = (
            record["lm_loss"],
            record["evidence_loss"],
            record["total_loss"],
            record["evidence_min"],
            record["evidence_max"],
            record["evidence_mean"],
        )
        if not all(math.isfinite(float(value)) for value in numerical):
            failures.append(f"{method}: non-finite loss/evidence trace")
        expected_total = record["lm_loss"] + expected_weight * record["evidence_loss"]
        if not math.isclose(record["total_loss"], expected_total, rel_tol=1e-5, abs_tol=1e-6):
            failures.append(f"{method}: total loss formula mismatch")
        if expected_weight == 0.0 and record["total_loss"] != record["lm_loss"]:
            failures.append(f"{method}: lambda-zero total loss is not exactly LM loss")
        if record["evidence_k"] != 11 or record["evidence_positions"] != 1024:
            failures.append(f"{method}: evidence shape/top-k contract changed")
    if len(step_timing) != 2 or [row["global_step"] for row in step_timing] != [1, 2]:
        failures.append(f"{method}: optimizer-step timing is incomplete")
    gradient_end = [row for row in gradient if row.get("event") == "gradient_audit_end"]
    if len(gradient_end) != 1 or gradient_end[0].get("status") != "SUCCESS":
        failures.append(f"{method}: strict trainable-gradient audit failed")
    gradient_steps = [row for row in gradient if row.get("event") == "gradient_audit_step"]
    target_activity = {}
    for prefix in TARGET_PREFIXES:
        matches = [
            parameter
            for step in gradient_steps
            for parameter in step["parameters"]
            if parameter["name"].startswith(prefix)
        ]
        target_activity[prefix] = any(
            row.get("gradient_present")
            and row.get("finite")
            and float(row.get("max_abs") or 0.0) > 0
            for row in matches
        )
    if not all(target_activity.values()):
        failures.append(f"{method}: an evidence target had no finite non-zero training gradient")
    if scope.get("status") != "SUCCESS" or scope.get("trainable_parameter_tensors") != 21 or scope.get(
        "trainable_parameter_elements"
    ) != 29_561_345:
        failures.append(f"{method}: B0-compatible trainable scope failed")
    if memory.get("status") != "SUCCESS" or memory.get("global_step") != 2:
        failures.append(f"{method}: internal training memory record failed")
    if not change.get("optimizer_changed_at_least_one_target"):
        failures.append(f"{method}: optimizer did not change a target parameter")
    gate = change.get("gate_scale_change", {})
    if not gate.get("initial_finite_positive") or not gate.get("final_finite_positive"):
        failures.append(f"{method}: gate_scale finite/positive gate failed")
    metadata = adapter_manifest.get("metadata", {})
    expected_method = "medic-ad-b0-as" if method == "b0-as" else "lad-mil-v2"
    metadata_expectations = {
        "stage": "2H-E",
        "protocol_version": protocol_version,
        "method_id": expected_method,
        "train_type": "anomaly_evidence",
        "evidence_loss_weight": expected_weight,
        "evidence_definition": "pre_gate_sigmoid_difference",
        "anomaly_query_mode": "single",
        "num_pooling_size": 4,
        "output_token_count": 16,
        "gate_type": "none",
    }
    for key, expected in metadata_expectations.items():
        if metadata.get(key) != expected:
            failures.append(f"{method}: adapter metadata mismatch for {key}")
    if adapter_manifest.get("trainable_parameter_tensors") != 21 or adapter_manifest.get(
        "trainable_parameter_elements"
    ) != 29_561_345:
        failures.append(f"{method}: adapter tensor schema changed")
    if evaluation.get("status") != "SUCCESS" or evaluation.get("records") != 46:
        failures.append(f"{method}: evaluation did not finish all 46 images")
    metrics = evaluation.get("metrics", {})
    if metrics.get("invalid_responses") != 0 or metrics.get("empty_responses") != 0:
        failures.append(f"{method}: invalid/empty generation is non-zero")
    parity = evaluation.get("strict_reload_parity", {})
    if parity.get("status") != "SUCCESS" or not all(
        parity.get(key)
        for key in (
            "adapter_identity_exact",
            "greedy_generation_exact",
            "anomaly_tokens_close",
            "evidence_close",
            "evidence_score_close",
        )
    ):
        failures.append(f"{method}: strict reload/generation parity failed")
    if not evaluation.get("read_only_evaluation_state_unchanged"):
        failures.append(f"{method}: evaluation changed adapter state")
    train_peak = external_peak(run / "training-external-peak-mib.txt")
    eval_peak = external_peak(run / "evaluation-external-peak-mib.txt")
    if train_peak > 22500 or eval_peak > 22500:
        failures.append(f"{method}: external VRAM exceeded 22500 MiB")
    step_mean = sum(row["wall_seconds"] for row in step_timing) / len(step_timing) if step_timing else math.inf
    return {
        "method": method,
        "failures": failures,
        "initial_state_fingerprint": initial_manifest["trainable_state_fingerprint"],
        "scope_fingerprint": scope.get("scope_fingerprint"),
        "target_gradient_activity": target_activity,
        "gate_scale": gate,
        "mean_step_wall_seconds": step_mean,
        "training_internal_peak_allocated_mib": memory.get("max_memory_allocated_mib"),
        "training_internal_peak_reserved_mib": memory.get("max_memory_reserved_mib"),
        "training_external_peak_mib": train_peak,
        "evaluation_external_peak_mib": eval_peak,
        "evaluation_internal_peak_allocated_mib": evaluation.get("peak_cuda_memory_allocated_mib"),
        "evaluation_internal_peak_reserved_mib": evaluation.get("peak_cuda_memory_reserved_mib"),
        "generation": metrics,
        "evidence_saturation": evaluation.get("evidence_saturation"),
        "adapter_sha256": adapter_manifest.get("checkpoint_sha256"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-version", default="2.1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root
    preflight = load_json(root / "preflight.json")
    calibration = load_json(root / "calibration.json")
    test_paths = sorted((root / "tests").glob("*.json"))
    tests = {path.stem: load_json(path) for path in test_paths}
    selected_lambda = float(calibration["calibration"]["selected_lambda"])
    b0 = check_run(root, "b0-as", 0.0, protocol_version=args.protocol_version)
    lad = check_run(root, "lad-mil-v2", selected_lambda, protocol_version=args.protocol_version)
    failures = []
    if preflight.get("status") != "SUCCESS":
        failures.append("preflight failed")
    if preflight.get("protocol_version", "2.1") != args.protocol_version:
        failures.append("preflight protocol version mismatch")
    if any(result.get("status") != "SUCCESS" for result in tests.values()):
        failures.append("one or more unit/integration tests failed")
    if calibration.get("status") != "SUCCESS":
        failures.append("lambda calibration failed")
    if not calibration.get("target_gradient_gate_passed") or not calibration.get(
        "gate_and_downstream_isolation_passed"
    ):
        failures.append("calibration gradient/isolation gate failed")
    calibration_peak = external_peak(root / "calibration-external-peak-mib.txt")
    if calibration_peak > 22500:
        failures.append("calibration external VRAM exceeded 22500 MiB")
    if b0["initial_state_fingerprint"] != lad["initial_state_fingerprint"]:
        failures.append("B0-AS/LAD-MIL matched step-0 initialization differs")
    calibration_state = calibration["runtime_audit"]["step0_state"]["trainable_state_fingerprint"]
    if calibration_state != b0["initial_state_fingerprint"]:
        failures.append("calibration and matched training step-0 initialization differs")
    runtime_ratio = lad["mean_step_wall_seconds"] / b0["mean_step_wall_seconds"]
    if not math.isfinite(runtime_ratio) or runtime_ratio > 1.10:
        failures.append(f"LAD-MIL/B0-AS step runtime ratio exceeds 1.10: {runtime_ratio}")
    failures.extend(b0["failures"])
    failures.extend(lad["failures"])
    lambda_rows = calibration["calibration"]["candidates"]
    selected_rows = [row for row in lambda_rows if row.get("selected")]
    if len(selected_rows) != 1 or not selected_rows[0].get("eligible"):
        failures.append("lambda selection is not unique and eligible")
    if preflight.get("answer_score_contract") not in {
        "first_token_margin_permitted",
        "full_sequence_log_likelihood_required",
    }:
        failures.append("tokenizer answer-score contract is unresolved")
    candidate_measurements = None
    if args.protocol_version == "2.2":
        candidate_measurements = load_json(root / "candidate-measurements.json")
        if candidate_measurements.get("status") != "SUCCESS" or candidate_measurements.get(
            "decision"
        ) != "PROCEED_TO_MATCHED_ENGINEERING_GATE":
            failures.append("v2.2 candidate measurement/selection gate failed")
        if candidate_measurements.get("selected_lambda") != selected_lambda:
            failures.append("v2.2 selected lambda artifact mismatch")
        if candidate_measurements.get("eligible_candidate_count", 0) <= 0:
            failures.append("v2.2 has no eligible candidate")

    decision = "READY FOR STAGE 2H-S" if not failures else "PAUSE"
    result = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "protocol_version": args.protocol_version,
        "decision": decision,
        "stage2h_s_executed": False,
        "promotion_failures": failures,
        "tests": {name: value.get("status") for name, value in tests.items()},
        "preflight": {
            "status": preflight.get("status"),
            "manifest_sha256": preflight.get("manifest_sha256"),
            "tokenizer_audit": preflight.get("tokenizer_audit"),
            "answer_score_contract": preflight.get("answer_score_contract"),
            "claim_scope": preflight.get("claim_scope"),
        },
        "lambda_calibration": calibration["calibration"],
        "candidate_measurements": candidate_measurements,
        "auxiliary_gradient_table": calibration["auxiliary_gradient_table"],
        "calibration_external_peak_mib": calibration_peak,
        "matched_initialization_exact": b0["initial_state_fingerprint"] == lad["initial_state_fingerprint"],
        "calibration_initialization_exact": calibration_state == b0["initial_state_fingerprint"],
        "b0_as": b0,
        "lad_mil_v2": lad,
        "step_runtime_ratio_lad_over_b0": runtime_ratio,
        "external_peak_guardrail_mib": 22500,
        "runtime_ratio_guardrail": 1.10,
        "claim_scope": "image_level_internal_engineering_only",
        "patient_independence_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
