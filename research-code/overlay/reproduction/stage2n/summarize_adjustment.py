#!/usr/bin/env python3
"""Independent verification for the one allowed S9 directional-moment reset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reproduction.stage2i.fingerprint import sha256_file
from reproduction.stage2l.constants import S9_NAMES
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2n import MOMENT_RESET_ARM
from reproduction.stage2n.summarize_repair import (
    _close,
    _prediction_summary,
    _rows,
    _training_summary,
    _verify_file_audits,
    _verify_resume,
    _verify_trace,
    classify,
)


def _load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--registered-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = _load(args.split_audit)
    if any(value for family in split["overlap"].values() for value in family.values()):
        raise RuntimeError("Moment-reset split overlap is nonzero.")
    if split.get("protected_internal_test_image_files_opened") != 0 or split.get("protected_internal_test_outputs_read") != 0:
        raise RuntimeError("Moment-reset protected boundary failed.")
    registered = _load(args.registered_comparison)
    if registered.get("decision") != "C" or set(registered.get("variant_decisions", {})) != {"cabg_cosine_taper", "cabg_spatial_guard"}:
        raise RuntimeError("Controlled adjustment lacks the registered v1 C prerequisite.")
    original = _load(args.original_root / "comparison.json")
    trace = _rows(args.run_root / MOMENT_RESET_ARM / "trace.jsonl")
    _verify_trace(trace, MOMENT_RESET_ARM)
    _verify_resume(args.run_root, MOMENT_RESET_ARM)
    _verify_file_audits(args.run_root, MOMENT_RESET_ARM)
    resets = [(row["block_id"] + 1, row["repair"].get("optimizer_moment_reset")) for row in trace if row["repair"].get("optimizer_moment_reset", {}).get("applied")]
    if len(resets) != 1:
        raise RuntimeError("Moment reset was not applied exactly once.")
    reset_block, reset = resets[0]
    reset_row = trace[reset_block - 1]
    if not (
        reset_row["repair"]["state_before"]["latched"] is False
        and reset_row["repair"]["state_after"]["latched"] is True
        and float(reset_row["repair"]["multiplier"]) == 0.0
        and reset["parameter_count"] == 9
        and reset["state_tensor_count"] == 9
        and 0 < reset["pre_reset_nonzero_parameter_count"] <= 9
        and reset["reset_names"] == list(S9_NAMES)
        and reset["state_key"] == "exp_avg"
        and reset["exp_avg_sq_preserved"] is True
        and reset["step_preserved"] is True
        and reset["post_reset_all_zero"] is True
    ):
        raise RuntimeError("Moment-reset record differs from the preregistered adjustment.")
    if any(row["repair"].get("optimizer_moment_reset", {}).get("applied") for row in trace if row is not reset_row):
        raise RuntimeError("Moment reset repeated after the latch transition.")
    original_trace = _rows(args.original_root / "cabg_lce" / "trace.jsonl")
    if [row["ordered_samples"] for row in trace] != [row["ordered_samples"] for row in original_trace]:
        raise RuntimeError("Moment-reset exposure order differs from original CABG.")
    for key in ("base_checkpoint_fingerprint", "train_manifest_sha256", "eval_manifest_sha256", "matched_configuration_sha256"):
        if {row["provenance"][key] for row in trace} != {original_trace[0]["provenance"][key]}:
            raise RuntimeError(f"Moment-reset matched identity differs: {key}")
    if {row["initial_state_sha256"] for row in trace} != {original_trace[0]["initial_state_sha256"]}:
        raise RuntimeError("Moment-reset initialization differs.")
    if [row["optimizer"]["learning_rate_used"] for row in trace] != [row["optimizer"]["learning_rate_used"] for row in original_trace]:
        raise RuntimeError("Moment-reset learning-rate schedule differs.")
    predictions_path = args.run_root / "evaluation" / MOMENT_RESET_ARM / "predictions.jsonl"
    predictions = _rows(predictions_path)
    original_predictions = _rows(args.original_root / "evaluation" / "cabg_lce" / "predictions.jsonl")
    identity = lambda rows: [(row["sample_id"], row["sha256"], row["label"]) for row in rows]
    if identity(predictions) != identity(original_predictions):
        raise RuntimeError("Moment-reset held-out identity/order differs.")
    metrics = _prediction_summary(predictions)
    reported = _load(args.run_root / "evaluation" / MOMENT_RESET_ARM / "metrics.json")["metrics"]
    recomputed = summarize_metrics(predictions)
    for key in ("auroc", "average_precision"):
        if not _close(recomputed[key], reported[key]) or not _close(metrics[key], reported[key]):
            raise RuntimeError(f"Moment-reset metric recomputation failed: {key}")
    training = _training_summary(trace, args.run_root, MOMENT_RESET_ARM)
    if training["external_peak_mib"] > 22500 or training["trust_ratio_max"] > 0.20001:
        raise RuntimeError("Moment-reset hard resource/trust gate failed.")
    decision = classify(
        metrics=metrics,
        training=training,
        lm_metrics=original["metrics"]["lm_only"],
        original_metrics=original["metrics"]["cabg_lce"],
    )
    result = {
        "status": "SUCCESS",
        "decision": decision["decision"],
        "claim_scope": "development_only_stability_repair_controlled_adjustment",
        "variant": MOMENT_RESET_ARM,
        "variant_decision": decision,
        "metrics": {
            "initial": original["metrics"]["initial"],
            "lm_only": original["metrics"]["lm_only"],
            "original_cabg": original["metrics"]["cabg_lce"],
            "registered_cosine_taper": registered["metrics"]["cabg_cosine_taper"],
            "registered_spatial_guard": registered["metrics"]["cabg_spatial_guard"],
            MOMENT_RESET_ARM: metrics,
        },
        "training_dynamics": training,
        "optimizer_moment_reset": {"block": reset_block, **reset},
        "fairness": {
            "same_initialization": True,
            "same_72_exposures_and_order": True,
            "same_train_eval_manifests": True,
            "same_t21_optimizer_scheduler_steps_clip": True,
            "only_registered_guard_and_one_time_s9_exp_avg_reset_differ": True,
            "heldout_used_for_training_control": False,
        },
        "data_isolation": {
            "all_registered_overlaps_zero": True,
            "protected_internal_test_image_files_opened": 0,
            "protected_internal_test_outputs_read": 0,
        },
        "source_commit": trace[0]["provenance"]["source_commit"],
        "implementation_fingerprint": trace[0]["provenance"]["implementation_fingerprint"],
        "prediction_sha256": sha256_file(predictions_path),
        "registered_comparison_sha256": sha256_file(args.registered_comparison),
        "scientific_claims_not_authorized": ["formal_effectiveness", "generalization", "patient_independence", "clinical_validity", "protected_internal_test_performance", "optimal_stopping"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
