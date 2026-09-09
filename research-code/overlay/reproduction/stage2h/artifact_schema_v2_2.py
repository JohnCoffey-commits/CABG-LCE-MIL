import math


PROTOCOL_VERSION = "2.2"
RECALIBRATION_CANDIDATES = (1e-4, 3e-4, 1e-3, 3e-3)
RATIO_BAND = (0.05, 0.20)
TARGET_RATIO = 0.10


def require_exact_keys(record, required, *, context):
    missing = sorted(set(required) - set(record))
    if missing:
        raise RuntimeError(f"{context} is missing required keys: {missing}")


def validate_dataset_audit(audit):
    require_exact_keys(
        audit,
        {
            "status",
            "protocol_version",
            "counts",
            "class_counts",
            "manifest_sha256",
            "annotation_sha256",
            "recalibration_is_strict_training_subset",
            "recalibration_previous_calibration_overlap",
            "calibration_partition_equals_training",
            "train_internal_test_overlap",
            "recalibration_internal_test_overlap",
            "internal_test_used_for_selection",
        },
        context="Stage 2H v2.2 dataset audit",
    )
    expected_counts = {"train": 16, "previous-calibration": 4, "calibration": 12, "internal-test": 46}
    if audit["status"] != "SUCCESS" or audit["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("Stage 2H v2.2 dataset audit status/version mismatch.")
    if audit["counts"] != expected_counts:
        raise RuntimeError("Stage 2H v2.2 dataset counts changed.")
    if audit["class_counts"]["calibration"] != {"good": 6, "ungood": 6}:
        raise RuntimeError("Stage 2H v2.2 recalibration class balance changed.")
    boolean_contract = (
        audit["recalibration_is_strict_training_subset"],
        audit["calibration_partition_equals_training"],
        audit["recalibration_previous_calibration_overlap"] == 0,
        audit["train_internal_test_overlap"] == 0,
        audit["recalibration_internal_test_overlap"] == 0,
        audit["internal_test_used_for_selection"] is False,
    )
    if not all(boolean_contract):
        raise RuntimeError("Stage 2H v2.2 dataset isolation contract failed.")


def validate_calibration(calibration):
    require_exact_keys(
        calibration,
        {
            "status",
            "protocol_version",
            "candidate_set",
            "rng_isolation_passed",
            "target_gradient_gate_passed",
            "gate_and_downstream_isolation_passed",
            "microbatch_observations",
            "calibration",
            "peak_cuda_memory_allocated_mib",
            "peak_cuda_memory_reserved_mib",
        },
        context="Stage 2H v2.2 calibration artifact",
    )
    if calibration["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("Stage 2H v2.2 calibration protocol mismatch.")
    if tuple(float(value) for value in calibration["candidate_set"]) != RECALIBRATION_CANDIDATES:
        raise RuntimeError("Stage 2H v2.2 candidate set changed.")
    if len(calibration["microbatch_observations"]) != 12:
        raise RuntimeError("Stage 2H v2.2 calibration must contain 12 observations.")
    if not all(
        calibration[key]
        for key in (
            "rng_isolation_passed",
            "target_gradient_gate_passed",
            "gate_and_downstream_isolation_passed",
        )
    ):
        raise RuntimeError("Stage 2H v2.2 calibration isolation/gradient gate failed.")
    core = calibration["calibration"]
    rows = core.get("candidates", [])
    if [float(row.get("lambda")) for row in rows] != list(RECALIBRATION_CANDIDATES):
        raise RuntimeError("Stage 2H v2.2 calibration candidate rows changed.")
    if core.get("logical_batch_size") != 12 or core.get("ratio_band") != list(RATIO_BAND):
        raise RuntimeError("Stage 2H v2.2 calibration batch/band changed.")
    if not math.isclose(float(core.get("target_ratio")), TARGET_RATIO, rel_tol=0, abs_tol=0):
        raise RuntimeError("Stage 2H v2.2 calibration target changed.")
    for row in rows:
        if not math.isfinite(float(row["gradient_ratio"])):
            raise RuntimeError("Stage 2H v2.2 candidate ratio is non-finite.")
    selected = [row for row in rows if row.get("selected")]
    eligible = [row for row in rows if row.get("eligible")]
    if eligible:
        expected = min(eligible, key=lambda row: (row["distance_to_target"], row["lambda"]))
        if len(selected) != 1 or selected[0] is not expected:
            raise RuntimeError("Stage 2H v2.2 unique selection rule failed.")
    elif selected:
        raise RuntimeError("Stage 2H v2.2 selected an ineligible candidate.")


def build_candidate_measurements(calibration, external_peak_mib):
    validate_calibration(calibration)
    peak = float(external_peak_mib)
    if not math.isfinite(peak) or peak < 0:
        raise RuntimeError("Stage 2H v2.2 external Peak VRAM is invalid.")
    core = calibration["calibration"]
    loss_finite = all(
        math.isfinite(float(row[loss_name]))
        for row in calibration["microbatch_observations"]
        for loss_name in ("lm_loss", "evidence_loss")
    )
    effective = all(
        row["lm_mean_gradient_finite"]
        and row["evidence_mean_gradient_finite"]
        and float(row["lm_mean_gradient_norm_fp32"]) > 1e-12
        and float(row["evidence_mean_gradient_norm_fp32"]) > 1e-12
        for row in core["parameter_gradients"]
    )
    rows = []
    for row in core["candidates"]:
        rows.append(
            {
                **row,
                "ratio_finite": math.isfinite(float(row["gradient_ratio"])),
                "losses_finite": loss_finite,
                "effective_target_gradients": effective,
                "nan_or_inf_detected": not (loss_finite and effective),
                "shared_step0_peak_cuda_memory_allocated_mib": calibration[
                    "peak_cuda_memory_allocated_mib"
                ],
                "shared_step0_peak_cuda_memory_reserved_mib": calibration[
                    "peak_cuda_memory_reserved_mib"
                ],
                "shared_step0_external_peak_mib": peak,
                "shared_measurement_id": "locked-12-image-step0-gradient-run",
            }
        )
    return {
        "status": "SUCCESS",
        "stage": "2H-E",
        "protocol_version": PROTOCOL_VERSION,
        "measurement_scope": "one shared exact step-0 gradient run; lambda is an analytic scale",
        "candidate_set": list(RECALIBRATION_CANDIDATES),
        "candidates": rows,
        "eligible_candidate_count": sum(row["eligible"] for row in rows),
        "selected_lambda": core["selected_lambda"],
        "decision": "PROCEED_TO_MATCHED_ENGINEERING_GATE" if core["selected_lambda"] is not None else "PAUSE",
        "external_peak_guardrail_mib": 22500,
        "external_peak_guardrail_passed": peak <= 22500,
    }
