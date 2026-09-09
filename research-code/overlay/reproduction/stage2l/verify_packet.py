#!/usr/bin/env python3
"""Independent static/data/test verification for D4 Packet materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2l.constants import (
    EXPECTED_D3R_SHA256, EXPECTED_TRAINING_DEVELOPMENT_SHA256, S9_FINGERPRINT,
    S9_NAMES, SOURCE_FILES, TRAINABLE_NAMES,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--unit-tests", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tests = json.loads(args.unit_tests.read_text())
    dataset = json.loads(args.dataset_audit.read_text())
    run_source = (args.repo_root / "reproduction/stage2l/run_pilot.py").read_text()
    evaluator_source = (args.repo_root / "reproduction/stage2l/verify_pilot.py").read_text()
    wrapper = (args.repo_root / "reproduction/stage2l/run_d4_pilot_v1_l4.sh").read_text()
    checks = {
        "unit_negative_regression_16_of_16": tests == {"errors": 0, "failures": 0, "skipped": 0, "status": "SUCCESS", "tests_run": 16},
        "eligible_difference_108_76_32": dataset.get("eligible_count") == 108 and dataset.get("eligible_class_counts") == {"abnormal": 76, "normal": 32},
        "pilot_12_8_4_four_blocks": dataset.get("selected_count") == 12 and dataset.get("selected_class_counts") == {"abnormal": 8, "normal": 4} and dataset.get("blocks") == 4,
        "d3r_overlap_zero": dataset.get("d3r_overlap") == 0 and dataset.get("d3r_exclusion_sha256") == EXPECTED_D3R_SHA256,
        "training_source_locked": dataset.get("training_development_sha256") == EXPECTED_TRAINING_DEVELOPMENT_SHA256,
        "manifest_byte_identity": dataset.get("manifest_sha256") == sha256_file(args.manifest),
        "protected_boundary_zero": dataset.get("protected_internal_test_image_files_opened") == 0 and dataset.get("protected_internal_test_outputs_read") == 0,
        "exact_s9_fingerprint": canonical_json_sha256(sorted(S9_NAMES)) == S9_FINGERPRINT,
        "t21_s9_cardinality": len(TRAINABLE_NAMES) == 21 and len(S9_NAMES) == 9 and set(S9_NAMES) < set(TRAINABLE_NAMES),
        "two_isolated_gradient_extractions": run_source.count("torch.autograd.grad(") == 2,
        "optimizer_scheduler_after_complete_block": "compute_block_gradients(" in run_source and "optimizer.step()" in run_source and "scheduler.step()" in run_source,
        "no_deepspeed_quantization_offload": all(term not in run_source.lower() for term in ("deepspeed", "quantization", "device_map", "offload")),
        "evaluator_independent_imports": "stage2l.controller" not in evaluator_source and "stage2l.run_pilot" not in evaluator_source,
        "fresh_process_boundary": wrapper.count("-m reproduction.stage2l.run_pilot") == 2 and "phase1_process_exited" in wrapper,
        "fail_closed_wrapper": "INVALID_QUARANTINED" in wrapper and "set -euo pipefail" in wrapper,
        "phase_marker_keyword_collision_absent": 'markers.mark("monitor_started", phase=' not in run_source and 'markers.mark("phase_completed", phase=' not in run_source,
        "source_inventory_complete": all((args.repo_root / path).is_file() for path in SOURCE_FILES),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks),
              "manifest_sha256": sha256_file(args.manifest), "unit_tests_sha256": sha256_file(args.unit_tests)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
