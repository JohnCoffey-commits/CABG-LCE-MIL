#!/usr/bin/env python3
"""Independent verification and decision for the shared-checkpoint causal continuation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2l.constants import S9_NAMES, TRAINABLE_NAMES
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2o import (
    BENEFIT_AP_FLOOR,
    BENEFIT_AUROC_FLOOR,
    BENEFIT_GAIN_FRACTION_FLOOR,
    CAUSAL_ARMS,
    CONTROL_ARM,
    LCE_OFF_ARM,
    PARENT_CHECKPOINT_SHA256,
    PARENT_EVAL_MANIFEST_SHA256,
    PARENT_RNG_FINGERPRINT,
    PARENT_TRACE_ANCHOR,
    PARENT_TRAIN_MANIFEST_SHA256,
    REPRO_MIN_LAST_FOUR_FAILED_BLOCKS,
    REPRO_MIN_SUPPORT_CROSSINGS,
    REPRO_MIN_TOP11_CROSSINGS,
    REPRO_SUPPORT_MEDIAN_CEILING,
    RESET_ARM,
    STABLE_MAX_LAST_FOUR_FAILED_BLOCKS,
    STABLE_MAX_SUPPORT_CROSSINGS,
    STABLE_MAX_TOP11_CROSSINGS,
    STABLE_SUPPORT_MEDIAN_FLOOR,
    TERMINAL_DECISIONS,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _registered_optimizer_delta(arm: str) -> list[str]:
    if arm != RESET_ARM:
        return []
    s9_name_set = set(S9_NAMES)
    return [f"{name}.exp_avg" for name in TRAINABLE_NAMES if name in s9_name_set]


def _write_exclusive(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prediction_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    if len(rows) != 32:
        raise RuntimeError("Causal held-out prediction count is not 32.")
    supports = [float(row["effective_support"]) for row in rows]
    top11 = [float(row["top11_mass"]) for row in rows]
    result = summarize_metrics(rows)
    result["spatial"] = {
        "effective_support_min": min(supports),
        "effective_support_median": median(supports),
        "support_below_128_count": sum(value < 128.0 for value in supports),
        "top11_mass_max": max(top11),
        "top11_mass_median": median(top11),
        "top11_above_0_35_count": sum(value > 0.35 for value in top11),
    }
    return result


def _training_summary(rows: list[dict[str, Any]], branch_root: Path, arm: str) -> dict[str, object]:
    supports = [float(image["effective_support"]) for row in rows for image in row["images"]]
    top11 = [float(image["top11_mass"]) for row in rows for image in row["images"]]
    last_four = []
    for row in rows[-4:]:
        support = median(float(image["effective_support"]) for image in row["images"])
        top = median(float(image["top11_mass"]) for image in row["images"])
        last_four.append({"block": row["block_id"] + 1, "support_median": support, "top11_mass_median": top, "failed": support < 128.0 or top > 0.35})
    monitor = _load(branch_root / arm / "phase2" / "monitor-summary.json")
    return {
        "blocks": len(rows),
        "images": len(supports),
        "effective_support_min": min(supports),
        "effective_support_median": median(supports),
        "support_below_128_count": sum(value < 128.0 for value in supports),
        "top11_mass_max": max(top11),
        "top11_mass_median": median(top11),
        "top11_above_0_35_count": sum(value > 0.35 for value in top11),
        "last_four": last_four,
        "last_four_failed_blocks": sum(row["failed"] for row in last_four),
        "lambda_controller_mean": mean(float(row["controller"]["lambda_controller"]) for row in rows),
        "lambda_applied_mean": mean(float(row["controller"]["lambda_final"]) for row in rows),
        "trust_ratio_max": max(float(row["controller"]["trust_ratio"]) for row in rows),
        "cap_activation_count": sum(bool(row["controller"]["cap_active"]) for row in rows),
        "preclip_norm_max": max(float(row["gradient"]["preclip_full_norm"]) for row in rows),
        "clip_scale_min": min(float(row["gradient"]["clip_scale"]) for row in rows),
        "clip_count": sum(float(row["gradient"]["clip_scale"]) < 1.0 - 1e-12 for row in rows),
        "lm_rms_mean": mean(float(row["class_rms"]["lm_balanced"]) for row in rows),
        "lce_rms_mean": mean(float(row["class_rms"]["lce_balanced"]) for row in rows),
        "block_seconds_total": sum(float(row["runtime"]["block_seconds"]) for row in rows),
        "block_images_per_second": 36.0 / sum(float(row["runtime"]["block_seconds"]) for row in rows),
        "external_peak_mib": int(monitor["peak_memory_used_mib"]),
    }


def _verify_parent(parent: Mapping[str, Any], split: Mapping[str, Any]) -> None:
    expected = {
        "status": "SUCCESS",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "cursor": 12,
        "trace_anchor": PARENT_TRACE_ANCHOR,
        "rng_fingerprint": PARENT_RNG_FINGERPRINT,
        "train_manifest_sha256": PARENT_TRAIN_MANIFEST_SHA256,
        "eval_manifest_sha256": PARENT_EVAL_MANIFEST_SHA256,
        "continuation_blocks": 12,
        "continuation_exposures": 36,
    }
    if any(parent.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Independent causal parent audit identity failed.")
    if split.get("train_manifest_sha256") != PARENT_TRAIN_MANIFEST_SHA256 or split.get("eval_manifest_sha256") != PARENT_EVAL_MANIFEST_SHA256:
        raise RuntimeError("Independent causal split identity failed.")
    if any(value for family in split["overlap"].values() for value in family.values()):
        raise RuntimeError("Independent causal split overlap is nonzero.")
    if split.get("protected_internal_test_image_files_opened") != 0 or split.get("protected_internal_test_outputs_read") != 0:
        raise RuntimeError("Independent causal protected boundary failed.")


def _verify_resume(branch_root: Path, arm: str, parent: Mapping[str, Any]) -> dict[str, Any]:
    audit = _load(branch_root / arm / "phase2" / "resume-audit.json")
    if not all((audit.get("resumed"), audit.get("adapter_exact"), audit.get("master_exact"))):
        raise RuntimeError(f"Causal {arm} adapter/master resume failed.")
    if audit.get("cursor") != 12 or audit.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError(f"Causal {arm} checkpoint/cursor resume failed.")
    if audit.get("rng_fingerprint") != PARENT_RNG_FINGERPRINT or audit.get("checkpoint_rng_fingerprint") != PARENT_RNG_FINGERPRINT or audit.get("post_intervention_rng_fingerprint") != PARENT_RNG_FINGERPRINT:
        raise RuntimeError(f"Causal {arm} RNG resume/intervention failed.")
    if audit.get("optimizer_tensor_inventory_sha256") != audit.get("checkpoint_optimizer_tensor_inventory_sha256") or audit.get("pre_intervention_optimizer_inventory_sha256") != audit.get("checkpoint_optimizer_tensor_inventory_sha256"):
        raise RuntimeError(f"Causal {arm} pre-intervention optimizer restore failed.")
    if audit.get("shared_parent_trace_anchor") != PARENT_TRACE_ANCHOR or audit.get("shared_parent", {}).get("checkpoint_sha256") != parent["checkpoint_sha256"]:
        raise RuntimeError(f"Causal {arm} serialized parent linkage failed.")
    intervention = audit.get("causal_optimizer_intervention")
    if not isinstance(intervention, dict) or intervention.get("branch") != arm:
        raise RuntimeError(f"Causal {arm} intervention record missing.")
    expected_changed = _registered_optimizer_delta(arm)
    if intervention.get("changed_state_tensors") != expected_changed or intervention.get("changed_state_tensor_count") != len(expected_changed):
        raise RuntimeError(f"Causal {arm} optimizer state delta differs from registration.")
    if not all((intervention.get("exp_avg_sq_preserved"), intervention.get("step_preserved"), intervention.get("non_s9_state_preserved"))):
        raise RuntimeError(f"Causal {arm} optimizer preservation failed.")
    if arm == RESET_ARM:
        if intervention.get("applied") is not True or intervention.get("s9_exp_avg_pre_nonzero_names") != list(S9_NAMES) or intervention.get("s9_exp_avg_post_zero_names") != list(S9_NAMES):
            raise RuntimeError("Causal reset branch did not reset exactly nine nonzero S9 first moments.")
    elif intervention.get("applied") is not False or intervention.get("before_inventory_sha256") != intervention.get("after_inventory_sha256") or intervention.get("s9_exp_avg_post_zero_names"):
        raise RuntimeError(f"Causal state-kept branch {arm} altered optimizer state.")
    return audit


def _verify_file_audits(branch_root: Path, arm: str) -> None:
    for path in (
        branch_root / arm / "phase2" / "file-open-audit.json",
        branch_root / "evaluation" / arm / "file-open-audit.json",
    ):
        audit = _load(path)
        if audit.get("status") != "SUCCESS" or audit.get("non_allowlisted_paths") or audit.get("protected_internal_test_outputs_read") != 0:
            raise RuntimeError(f"Causal file audit failed: {path}")


def _verify_trace(branch_root: Path, arm: str, original_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _rows(branch_root / arm / "trace.jsonl")
    if len(rows) != 12 or [row.get("block_id") for row in rows] != list(range(12, 24)):
        raise RuntimeError(f"Causal {arm} trace is not blocks 13-24.")
    previous = PARENT_TRACE_ANCHOR
    expected_multiplier = 1.0 if arm == CONTROL_ARM else 0.0
    for index, row in enumerate(rows):
        if row.get("status") != "SUCCESS" or row.get("arm") != arm or row.get("previous_record_sha256") != previous:
            raise RuntimeError(f"Causal {arm} trace identity/anchor failed.")
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if row.get("record_sha256") != canonical_json_sha256(payload):
            raise RuntimeError(f"Causal {arm} trace record hash failed.")
        if row["ordered_samples"] != original_trace[12 + index]["ordered_samples"]:
            raise RuntimeError(f"Causal {arm} continuation exposure order changed.")
        if not _close(row["optimizer"]["learning_rate_used"], original_trace[12 + index]["optimizer"]["learning_rate_used"], 1e-15):
            raise RuntimeError(f"Causal {arm} learning-rate schedule changed.")
        if row["applied_gradient"]["tensor_count"] != 21 or row["applied_gradient"]["all_finite"] is not True or not 0 < int(row["parameter_update"]["changed_tensor_count"]) <= 21:
            raise RuntimeError(f"Causal {arm} applied/update support failed.")
        if len(row["images"]) != 3 or any(not image["exact_s9"] or not image["structural_zero_outside_s9"] for image in row["images"]):
            raise RuntimeError(f"Causal {arm} image/S9 support failed.")
        if any(not math.isfinite(float(image[key])) for image in row["images"] for key in ("lm_loss", "lce_loss", "lce_score", "effective_support", "top11_mass", "lm_s9_norm", "lce_s9_norm")):
            raise RuntimeError(f"Causal {arm} image diagnostic is non-finite.")
        causal = row.get("causal")
        if not isinstance(causal, dict) or causal.get("branch") != arm or causal.get("heldout_input_used") is not False or not _close(causal.get("applied_lce_multiplier"), expected_multiplier):
            raise RuntimeError(f"Causal {arm} registered intervention record failed.")
        if causal.get("shared_parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256 or causal.get("shared_parent_trace_anchor") != PARENT_TRACE_ANCHOR:
            raise RuntimeError(f"Causal {arm} parent identity changed in trace.")
        controller = row["controller"]
        expected_lambda = float(controller["lambda_controller"]) * expected_multiplier
        if not _close(controller["lambda_final"], expected_lambda) or not _close(controller["lambda_controller"], min(float(controller["lambda_raw"]), float(controller["lambda_cap"]))):
            raise RuntimeError(f"Causal {arm} controller/applied lambda failed recomputation.")
        gradient = row["gradient"]
        trust = float(gradient["scaled_auxiliary_shared_norm"]) / (float(gradient["lm_shared_norm"]) + 1e-12)
        if not _close(controller["trust_ratio"], trust) or trust > 0.20001:
            raise RuntimeError(f"Causal {arm} trust cap failed.")
        expected_clip = min(1.0, 1.0 / (float(gradient["preclip_full_norm"]) + 1e-12))
        if not _close(gradient["clip_scale"], expected_clip) or float(gradient["postclip_full_norm"]) > 1.00001 or gradient.get("cap_checked_before_clip") is not True:
            raise RuntimeError(f"Causal {arm} global clip failed recomputation.")
        previous = row["record_sha256"]
    if rows[0]["rng"]["before_block"] != PARENT_RNG_FINGERPRINT:
        raise RuntimeError(f"Causal {arm} first-block RNG differs from parent.")
    return rows


def _verify_final_checkpoint(branch_root: Path, arm: str, provenance: Mapping[str, object]) -> dict[str, object]:
    checkpoint = branch_root / arm / "checkpoints" / "final-block24.pt"
    payload, manifest = load_strict(checkpoint, provenance)
    if payload["sampler_state"]["cursor"] != 24 or payload["trace_state"]["completed_blocks"] != 24:
        raise RuntimeError(f"Causal {arm} final checkpoint cursor failed.")
    saved = _load(branch_root / arm / "phase2" / "checkpoint-save.json")
    if saved != manifest or saved["checkpoint_sha256"] != sha256_file(checkpoint):
        raise RuntimeError(f"Causal {arm} final checkpoint manifest failed.")
    return {"checkpoint_sha256": manifest["checkpoint_sha256"], "tensor_inventory_sha256": manifest["tensor_inventory_sha256"], "cursor": 24}


def verify_branch(
    branch_root: Path,
    arm: str,
    *,
    parent: Mapping[str, Any],
    split: Mapping[str, Any],
    original_root: Path,
) -> dict[str, object]:
    if arm not in CAUSAL_ARMS:
        raise RuntimeError(f"Unknown causal branch: {arm}")
    original_trace = _rows(original_root / "cabg_lce" / "trace.jsonl")
    if len(original_trace) != 24 or original_trace[11]["record_sha256"] != PARENT_TRACE_ANCHOR:
        raise RuntimeError("Original CABG trace identity failed.")
    rows = _verify_trace(branch_root, arm, original_trace)
    resume = _verify_resume(branch_root, arm, parent)
    _verify_file_audits(branch_root, arm)
    checkpoint = _verify_final_checkpoint(branch_root, arm, rows[0]["provenance"])
    predictions_path = branch_root / "evaluation" / arm / "predictions.jsonl"
    predictions = _rows(predictions_path)
    original_predictions = _rows(original_root / "evaluation" / "cabg_lce" / "predictions.jsonl")
    identity = lambda values: [(row["sample_id"], row["sha256"], row["label"]) for row in values]
    if identity(predictions) != identity(original_predictions):
        raise RuntimeError(f"Causal {arm} held-out identity/order changed.")
    metrics = _prediction_summary(predictions)
    reported = _load(branch_root / "evaluation" / arm / "metrics.json")
    for key in ("auroc", "average_precision"):
        if not _close(metrics[key], reported["metrics"][key]):
            raise RuntimeError(f"Causal {arm} metric recomputation failed: {key}")
    if reported.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"] or reported.get("protected_internal_test_image_files_opened") != 0 or reported.get("protected_internal_test_outputs_read") != 0:
        raise RuntimeError(f"Causal {arm} evaluation identity/boundary failed.")
    training = _training_summary(rows, branch_root, arm)
    if training["external_peak_mib"] > 22500 or training["trust_ratio_max"] > 0.20001:
        raise RuntimeError(f"Causal {arm} resource/trust hard gate failed.")
    reproduced = (
        float(metrics["spatial"]["effective_support_median"]) < REPRO_SUPPORT_MEDIAN_CEILING
        and int(metrics["spatial"]["support_below_128_count"]) >= REPRO_MIN_SUPPORT_CROSSINGS
        and int(metrics["spatial"]["top11_above_0_35_count"]) >= REPRO_MIN_TOP11_CROSSINGS
        and int(training["last_four_failed_blocks"]) >= REPRO_MIN_LAST_FOUR_FAILED_BLOCKS
    )
    return {
        "status": "SUCCESS",
        "arm": arm,
        "metrics": metrics,
        "training": training,
        "reproduction_phenotype": reproduced,
        "first_rng_fingerprint": rows[0]["rng"]["before_block"],
        "ordered_exposures_sha256": canonical_json_sha256([row["ordered_samples"] for row in rows]),
        "provenance": rows[0]["provenance"],
        "resume": {
            "adapter_exact": resume["adapter_exact"],
            "master_exact": resume["master_exact"],
            "optimizer_exact_before_intervention": resume["optimizer_tensor_inventory_sha256"] == resume["checkpoint_optimizer_tensor_inventory_sha256"],
            "rng_exact": resume["post_intervention_rng_fingerprint"] == PARENT_RNG_FINGERPRINT,
            "intervention": resume["causal_optimizer_intervention"],
        },
        "checkpoint": checkpoint,
        "predictions_sha256": sha256_file(predictions_path),
    }


def _stability_and_benefit(branch: Mapping[str, Any], original: Mapping[str, Any]) -> dict[str, object]:
    metrics = branch["metrics"]
    spatial = metrics["spatial"]
    training = branch["training"]
    lm = original["metrics"]["lm_only"]
    cabg = original["metrics"]["cabg_lce"]
    retained_auroc = (float(metrics["auroc"]) - float(lm["auroc"])) / (float(cabg["auroc"]) - float(lm["auroc"]))
    retained_ap = (float(metrics["average_precision"]) - float(lm["average_precision"])) / (float(cabg["average_precision"]) - float(lm["average_precision"]))
    stable = (
        float(spatial["effective_support_median"]) >= STABLE_SUPPORT_MEDIAN_FLOOR
        and int(spatial["support_below_128_count"]) <= STABLE_MAX_SUPPORT_CROSSINGS
        and int(spatial["top11_above_0_35_count"]) <= STABLE_MAX_TOP11_CROSSINGS
        and int(training["last_four_failed_blocks"]) <= STABLE_MAX_LAST_FOUR_FAILED_BLOCKS
    )
    benefit = (
        float(metrics["auroc"]) >= BENEFIT_AUROC_FLOOR
        and float(metrics["average_precision"]) >= BENEFIT_AP_FLOOR
        and retained_auroc >= BENEFIT_GAIN_FRACTION_FLOOR
        and retained_ap >= BENEFIT_GAIN_FRACTION_FLOOR
    )
    return {"spatial_stability_pass": stable, "benefit_retention_pass": benefit, "retained_gain_fraction": {"auroc": retained_auroc, "average_precision": retained_ap}}


def classify_terminal(
    *,
    primary_control_reproduced: bool,
    repeat_control_reproduced: bool | None,
    state_kept: Mapping[str, bool] | None,
    reset: Mapping[str, bool] | None,
) -> str:
    if not primary_control_reproduced:
        if repeat_control_reproduced is not True:
            return "INCONCLUSIVE_REPRODUCIBILITY"
        return "INCONCLUSIVE_REPRODUCIBILITY"
    if state_kept is None or reset is None:
        raise RuntimeError("Reproducing control requires both causal branches.")
    if state_kept["spatial_stability_pass"] and state_kept["benefit_retention_pass"]:
        return "CONTROLLER_REPAIR_ELIGIBLE"
    if not state_kept["spatial_stability_pass"] and reset["spatial_stability_pass"] and reset["benefit_retention_pass"]:
        return "OPTIMIZER_REPAIR_ELIGIBLE"
    return "REDESIGN_REQUIRED"


def control_check(args: argparse.Namespace) -> dict[str, object]:
    parent, split = _load(args.parent_audit), _load(args.split_audit)
    _verify_parent(parent, split)
    branch = verify_branch(args.branch_root, CONTROL_ARM, parent=parent, split=split, original_root=args.original_root)
    return {
        "status": "SUCCESS",
        "decision": "CONTROL_REPRODUCED" if branch["reproduction_phenotype"] else "CONTROL_NOT_REPRODUCED",
        "reproduction_phenotype": branch["reproduction_phenotype"],
        "branch": branch,
        "thresholds": {
            "support_median_below": REPRO_SUPPORT_MEDIAN_CEILING,
            "minimum_support_crossings": REPRO_MIN_SUPPORT_CROSSINGS,
            "minimum_top11_crossings": REPRO_MIN_TOP11_CROSSINGS,
            "minimum_last_four_failed_blocks": REPRO_MIN_LAST_FOUR_FAILED_BLOCKS,
        },
    }


def final_summary(args: argparse.Namespace) -> dict[str, object]:
    parent, split = _load(args.parent_audit), _load(args.split_audit)
    _verify_parent(parent, split)
    original = _load(args.original_root / "comparison.json")
    primary = verify_branch(args.control_primary, CONTROL_ARM, parent=parent, split=split, original_root=args.original_root)
    repeat = None
    if args.control_repeat is not None:
        repeat = verify_branch(args.control_repeat, CONTROL_ARM, parent=parent, split=split, original_root=args.original_root)
    state_kept = reset = None
    state_decision = reset_decision = None
    if primary["reproduction_phenotype"] or (repeat is not None and repeat["reproduction_phenotype"]):
        if args.causal_root is None:
            raise RuntimeError("A reproducing control requires both causal branches.")
        state_kept = verify_branch(args.causal_root, LCE_OFF_ARM, parent=parent, split=split, original_root=args.original_root)
        reset = verify_branch(args.causal_root, RESET_ARM, parent=parent, split=split, original_root=args.original_root)
        identity_fields = ("first_rng_fingerprint", "ordered_exposures_sha256")
        branches = [primary, state_kept, reset]
        if any(len({branch[field] for branch in branches}) != 1 for field in identity_fields):
            raise RuntimeError("Causal branch starting RNG or exposure order differs.")
        state_decision = _stability_and_benefit(state_kept, original)
        reset_decision = _stability_and_benefit(reset, original)
    decision = classify_terminal(
        primary_control_reproduced=bool(primary["reproduction_phenotype"]),
        repeat_control_reproduced=None if repeat is None else bool(repeat["reproduction_phenotype"]),
        state_kept=state_decision,
        reset=reset_decision,
    )
    if decision not in TERMINAL_DECISIONS:
        raise RuntimeError("Unknown causal terminal decision.")
    return {
        "status": "SUCCESS",
        "decision": decision,
        "claim_scope": "development_only_shared_checkpoint_causal_mechanism",
        "parent": parent,
        "control_primary": primary,
        "control_repeat": repeat,
        "lce_off_state_kept": state_kept,
        "lce_off_state_kept_decision": state_decision,
        "lce_off_s9_exp_avg_reset": reset,
        "lce_off_s9_exp_avg_reset_decision": reset_decision,
        "fairness": {
            "same_original_block12_checkpoint": True,
            "same_first_rng": True if state_kept is not None else None,
            "same_blocks13_24_exposure_order": True if state_kept is not None else None,
            "same_t21_optimizer_scheduler_controller_clip": True if state_kept is not None else None,
            "heldout_used_for_training_control": False,
        },
        "data_isolation": {
            "all_registered_overlaps_zero": True,
            "protected_internal_test_image_files_opened": 0,
            "protected_internal_test_outputs_read": 0,
        },
        "thresholds": {
            "reproduction": {"support_median_below": REPRO_SUPPORT_MEDIAN_CEILING, "minimum_support_crossings": REPRO_MIN_SUPPORT_CROSSINGS, "minimum_top11_crossings": REPRO_MIN_TOP11_CROSSINGS, "minimum_last_four_failed_blocks": REPRO_MIN_LAST_FOUR_FAILED_BLOCKS},
            "spatial_stability": {"support_median_at_least": STABLE_SUPPORT_MEDIAN_FLOOR, "maximum_support_crossings": STABLE_MAX_SUPPORT_CROSSINGS, "maximum_top11_crossings": STABLE_MAX_TOP11_CROSSINGS, "maximum_last_four_failed_blocks": STABLE_MAX_LAST_FOUR_FAILED_BLOCKS},
            "benefit": {"auroc_at_least": BENEFIT_AUROC_FLOOR, "average_precision_at_least": BENEFIT_AP_FLOOR, "gain_fraction_at_least": BENEFIT_GAIN_FRACTION_FLOOR},
        },
        "scientific_claims_not_authorized": ["formal_effectiveness", "generalization", "patient_independence", "clinical_validity", "protected_internal_test_performance", "optimal_stopping"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    control = subparsers.add_parser("control-check")
    control.add_argument("--branch-root", type=Path, required=True)
    control.add_argument("--parent-audit", type=Path, required=True)
    control.add_argument("--split-audit", type=Path, required=True)
    control.add_argument("--original-root", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--control-primary", type=Path, required=True)
    final.add_argument("--control-repeat", type=Path)
    final.add_argument("--causal-root", type=Path)
    final.add_argument("--parent-audit", type=Path, required=True)
    final.add_argument("--split-audit", type=Path, required=True)
    final.add_argument("--original-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = control_check(args) if args.mode == "control-check" else final_summary(args)
    _write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
