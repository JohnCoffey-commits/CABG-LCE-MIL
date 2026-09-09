#!/usr/bin/env python3

import argparse
import json
import math
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
    tests = {
        path.stem: load_json(path)
        for path in sorted((args.root / "tests").glob("*.json"))
    }
    peak = float((args.root / "calibration-external-peak-mib.txt").read_text().strip())
    if preflight.get("status") != "SUCCESS":
        raise RuntimeError("Cannot finalize calibration PAUSE after failed preflight.")
    if any(value.get("status") != "SUCCESS" for value in tests.values()):
        raise RuntimeError("Cannot finalize calibration PAUSE after failed tests.")
    if calibration.get("status") != "PAUSE" or calibration.get("calibration", {}).get(
        "status"
    ) != "NO_ELIGIBLE_CANDIDATE":
        raise RuntimeError("Calibration terminal result is not the registered no-candidate PAUSE.")
    if not calibration.get("target_gradient_gate_passed") or not calibration.get(
        "gate_and_downstream_isolation_passed"
    ):
        raise RuntimeError("Calibration PAUSE cannot hide a gradient/isolation implementation failure.")
    if not math.isfinite(peak):
        raise RuntimeError("Calibration external VRAM peak is non-finite.")
    rows = calibration["calibration"]["candidates"]
    if any(row.get("eligible") or row.get("selected") for row in rows):
        raise RuntimeError("No-candidate PAUSE contains an eligible/selected lambda.")
    result = {
        "status": "SUCCESS",
        "stage": "2H-E",
        "decision": "PAUSE",
        "terminal_gate": "lambda_calibration",
        "reason": calibration["pause_reason"],
        "promotion_failures": ["No fixed lambda candidate lies inside the registered gradient-ratio band."],
        "tests": {name: value["status"] for name, value in tests.items()},
        "preflight": preflight,
        "lambda_calibration": calibration["calibration"],
        "auxiliary_gradient_table": calibration["auxiliary_gradient_table"],
        "gate_and_downstream_isolation_passed": True,
        "target_gradient_gate_passed": True,
        "evidence_saturation": calibration["evidence_saturation"],
        "gate_scale": calibration["gate_scale"],
        "calibration_internal_peak_allocated_mib": calibration[
            "peak_cuda_memory_allocated_mib"
        ],
        "calibration_internal_peak_reserved_mib": calibration[
            "peak_cuda_memory_reserved_mib"
        ],
        "calibration_external_peak_mib": peak,
        "external_peak_guardrail_mib": 22500,
        "external_peak_guardrail_passed": peak <= 22500,
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
