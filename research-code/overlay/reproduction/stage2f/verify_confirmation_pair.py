#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path

from reproduction.stage2f.gradient_evidence import has_finite_nonzero_element


EXPECTED_STEPS = 32
OVERALL_EM_DISCREPANCY = 0.0625
EVAL_LOSS_DISCREPANCY = 0.2094728946685791


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def confirmation_decision(
    *,
    b0_reliability_failure: bool,
    a3_reliability_failure: bool,
    overall_em_delta: float,
    eval_loss_delta: float,
):
    stable_metric_degradation = bool(
        overall_em_delta < -OVERALL_EM_DISCREPANCY
        and eval_loss_delta > EVAL_LOSS_DISCREPANCY
    )
    if b0_reliability_failure:
        return stable_metric_degradation, "COMMON_RUN_FAILURE", "BLOCKED_COMMON_FAILURE"
    if a3_reliability_failure or stable_metric_degradation:
        return (
            stable_metric_degradation,
            "METHOD_SPECIFIC_DEGRADATION_REPRODUCED",
            "PAUSE_FB_MAQ",
        )
    return (
        stable_metric_degradation,
        "COLLAPSE_NOT_REPRODUCED_INSTABILITY_REMAINS",
        "DRAFT_STAGE2G_PILOT_PROTOCOL",
    )


def validate_summary(summary, label: str) -> None:
    if not summary.get("finite"):
        raise RuntimeError(f"Non-finite {label} diagnostic.")
    for key in ("min", "max", "mean", "std", "abs_max"):
        if not math.isfinite(float(summary[key])):
            raise RuntimeError(f"Invalid {label} diagnostic field: {key}")


def validate_run(root: Path, expected_run_id: str, mode: str):
    verification = read_json(root / "run-verification.json")
    generation = read_json(root / "generation.json")
    if (root / "run.status").read_text(encoding="utf-8").strip() != "SUCCESS":
        raise RuntimeError(f"Confirmation run failed: {expected_run_id}")
    if verification.get("status") != "SUCCESS" or generation.get("status") != "SUCCESS":
        raise RuntimeError(f"Confirmation verification failed: {expected_run_id}")
    if verification.get("run_id") != expected_run_id or generation.get("run_id") != expected_run_id:
        raise RuntimeError(f"Confirmation run identity mismatch: {expected_run_id}")
    if verification.get("seed") != 123 or verification.get("repeat") != 1:
        raise RuntimeError(f"Confirmation seed/repeat mismatch: {expected_run_id}")
    if verification.get("anomaly_query_mode") != mode:
        raise RuntimeError(f"Confirmation mode mismatch: {expected_run_id}")
    if generation.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official VQA-RAD test must remain unused.")

    activation = read_jsonl(root / "activation-diagnostics.jsonl")
    starts = [row for row in activation if row.get("event") == "activation_diagnostic_start"]
    steps = [row for row in activation if row.get("event") == "activation_diagnostic_step"]
    ends = [row for row in activation if row.get("event") == "activation_diagnostic_end"]
    if len(starts) != 1 or len(ends) != 1 or ends[0].get("status") != "SUCCESS":
        raise RuntimeError(f"Activation diagnostic lifecycle failed: {expected_run_id}")
    if [row.get("optimizer_step") for row in steps] != list(range(1, EXPECTED_STEPS + 1)):
        raise RuntimeError(f"Activation diagnostic steps are incomplete: {expected_run_id}")

    lambda_means = []
    delta_abs_max = []
    for row in steps:
        delta_observations = row.get("attention_delta_observations", [])
        if not delta_observations:
            raise RuntimeError(f"Missing attention-delta diagnostic: {expected_run_id}")
        for summary in delta_observations:
            validate_summary(summary, "attention-delta")
            delta_abs_max.append(float(summary["abs_max"]))
        lambda_observations = row.get("lambda_observations", [])
        if mode == "multiscale":
            if not lambda_observations or row.get("gate_parameters_before_update") is None:
                raise RuntimeError("A3 activation diagnostic is missing gate evidence.")
            for summary in lambda_observations:
                validate_summary(summary, "fusion-lambda")
                if float(summary["min"]) < 0.0 or float(summary["max"]) > 1.0:
                    raise RuntimeError("Fusion lambda is outside [0, 1].")
                lambda_means.append(float(summary["mean"]))
        elif lambda_observations or row.get("gate_parameters_before_update") is not None:
            raise RuntimeError("B0 unexpectedly contains gate activation evidence.")

    gradient = read_jsonl(root / "gradient-audit.jsonl")
    gradient_steps = [row for row in gradient if row.get("event") == "gradient_audit_step"]
    if [row.get("optimizer_step") for row in gradient_steps] != list(
        range(1, EXPECTED_STEPS + 1)
    ):
        raise RuntimeError(f"Dense gradient audit is incomplete: {expected_run_id}")
    gate_gradient_active_steps = {}
    for suffix in ("fusion_gate.weight", "fusion_gate.bias"):
        observations = [
            (row["optimizer_step"], parameter)
            for row in gradient_steps
            for parameter in row.get("parameters", [])
            if parameter.get("name", "").endswith(suffix)
        ]
        if mode == "single":
            if observations:
                raise RuntimeError("B0 unexpectedly contains gate gradient evidence.")
            continue
        if len(observations) != EXPECTED_STEPS or not all(
            parameter.get("gradient_present") and parameter.get("finite")
            for _, parameter in observations
        ):
            raise RuntimeError(f"A3 dense gate gradient evidence failed: {suffix}")
        active = [step for step, record in observations if has_finite_nonzero_element(record)]
        if not active:
            raise RuntimeError(f"A3 gate gradient never had a non-zero element: {suffix}")
        gate_gradient_active_steps[suffix] = active

    predictions = generation.get("predictions", [])
    if len(predictions) != 16:
        raise RuntimeError(f"Confirmation predictions are incomplete: {expected_run_id}")
    character_counts = [len(str(row.get("response", ""))) for row in predictions]
    whitespace_token_counts = [len(str(row.get("response", "")).split()) for row in predictions]
    metrics = verification["metrics"]
    return {
        "verification": verification,
        "generation": generation,
        "diagnostics": {
            "activation_steps": len(steps),
            "lambda_observation_count": len(lambda_means),
            "lambda_mean_min": min(lambda_means) if lambda_means else None,
            "lambda_mean_max": max(lambda_means) if lambda_means else None,
            "lambda_mean_over_observations": statistics.fmean(lambda_means) if lambda_means else None,
            "attention_delta_abs_max_over_observations": max(delta_abs_max),
            "gate_gradient_active_steps": gate_gradient_active_steps,
            "response_character_counts": character_counts,
            "response_whitespace_token_counts": whitespace_token_counts,
            "mean_response_characters": statistics.fmean(character_counts),
            "mean_response_whitespace_tokens": statistics.fmean(whitespace_token_counts),
        },
        "reliability_failure": bool(
            metrics["empty_responses"] > 0 or metrics["invalid_closed_responses"] > 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-root", type=Path, required=True)
    parser.add_argument("--b0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite confirmation result: {args.output}")

    a3 = validate_run(args.a3_root, "confirmation-seed123-repeat1-a3", "multiscale")
    b0 = validate_run(args.b0_root, "confirmation-seed123-repeat1-b0", "single")
    a3_verification = a3["verification"]
    b0_verification = b0["verification"]
    if (
        a3_verification["implementation_source_fingerprint"]
        != b0_verification["implementation_source_fingerprint"]
    ):
        raise RuntimeError("Confirmation runs used different source fingerprints.")

    overall_em_delta = (
        a3_verification["metrics"]["overall_exact_match_accuracy"]
        - b0_verification["metrics"]["overall_exact_match_accuracy"]
    )
    eval_loss_delta = a3_verification["eval_losses"][-1] - b0_verification["eval_losses"][-1]
    stable_metric_degradation, outcome, decision = confirmation_decision(
        b0_reliability_failure=b0["reliability_failure"],
        a3_reliability_failure=a3["reliability_failure"],
        overall_em_delta=overall_em_delta,
        eval_loss_delta=eval_loss_delta,
    )

    result = {
        "status": "SUCCESS",
        "stage": "2F-C",
        "evaluation_role": "single_predefined_confirmation_diagnosis",
        "anomaly_effectiveness_evaluated": False,
        "official_test_downloaded_or_used": False,
        "implementation_source_fingerprint": a3_verification[
            "implementation_source_fingerprint"
        ],
        "execution_order": [
            "confirmation-seed123-repeat1-a3",
            "confirmation-seed123-repeat1-b0",
        ],
        "thresholds": {
            "overall_em_observed_repeat_discrepancy": OVERALL_EM_DISCREPANCY,
            "eval_loss_observed_repeat_discrepancy": EVAL_LOSS_DISCREPANCY,
            "not_statistical_thresholds": True,
        },
        "paired_deltas": {
            "a3_minus_b0_overall_em": overall_em_delta,
            "a3_minus_b0_eval_loss_step32": eval_loss_delta,
        },
        "stable_metric_degradation": stable_metric_degradation,
        "a3": a3,
        "b0": b0,
        "outcome": outcome,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
