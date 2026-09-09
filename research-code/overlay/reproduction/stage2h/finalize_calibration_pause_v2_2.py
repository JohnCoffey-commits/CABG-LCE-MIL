#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preflight = load_json(args.root / "preflight.json")
    calibration = load_json(args.root / "calibration.json")
    measurements = load_json(args.root / "candidate-measurements.json")
    tests = {path.stem: load_json(path) for path in sorted((args.root / "tests").glob("*.json"))}
    if preflight.get("status") != "SUCCESS" or preflight.get("protocol_version") != "2.2":
        raise RuntimeError("Cannot finalize v2.2 PAUSE after failed preflight.")
    if any(value.get("status") != "SUCCESS" for value in tests.values()):
        raise RuntimeError("Cannot finalize v2.2 PAUSE after failed tests.")
    if calibration.get("status") != "PAUSE" or measurements.get("decision") != "PAUSE":
        raise RuntimeError("v2.2 terminal result is not the preregistered no-candidate PAUSE.")
    if measurements.get("eligible_candidate_count") != 0 or measurements.get("selected_lambda") is not None:
        raise RuntimeError("v2.2 PAUSE contains an eligible or selected candidate.")
    result = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "protocol_version": "2.2",
        "decision": "PAUSE",
        "terminal_gate": "lambda_recalibration",
        "reason": "No preregistered v2.2 lambda candidate lies inside [0.05,0.20].",
        "promotion_failures": ["No eligible preregistered lambda candidate."],
        "tests": {name: value["status"] for name, value in tests.items()},
        "preflight": preflight,
        "candidate_measurements": measurements,
        "lambda_calibration": calibration["calibration"],
        "auxiliary_gradient_table": calibration["auxiliary_gradient_table"],
        "rng_isolation_passed": calibration["rng_isolation_passed"],
        "gate_and_downstream_isolation_passed": calibration["gate_and_downstream_isolation_passed"],
        "target_gradient_gate_passed": calibration["target_gradient_gate_passed"],
        "matched_training_runs_executed": 0,
        "internal_test_generation_runs_executed": 0,
        "strict_reload_executed": False,
        "stage2h_s_executed": False,
        "claim_scope": "image_level_internal_engineering_only",
        "patient_independence_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
