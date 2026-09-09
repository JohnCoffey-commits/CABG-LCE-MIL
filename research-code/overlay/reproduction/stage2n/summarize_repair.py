#!/usr/bin/env python3
"""Independent comparison for the preregistered D4-Scout stability repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2n import REPAIR_ARMS


AUROC_RETENTION_FLOOR = 0.8966666666666667
AP_RETENTION_FLOOR = 0.9352228131900764
SUPPORT_MEDIAN_FLOOR = 256.0
MAX_SPATIAL_CROSSINGS = 8
MAX_LAST_FOUR_FAILED_BLOCKS = 1


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _expected_taper(block_id: int) -> float:
    block_number = block_id + 1
    if block_number <= 12:
        return 1.0
    if block_number >= 20:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * (block_number - 12) / 8.0))


def visible_update_support_valid(applied_tensor_count: int, changed_tensor_count: int) -> bool:
    return applied_tensor_count == 21 and 0 < changed_tensor_count <= 21


def _verify_trace(rows: list[dict[str, Any]], arm: str) -> None:
    if len(rows) != 24 or [row.get("block_id") for row in rows] != list(range(24)):
        raise RuntimeError(f"Repair {arm} trace is not 24 ordered blocks.")
    previous = "0" * 64
    streak, latched = 0, False
    for row in rows:
        if row.get("status") != "SUCCESS" or row.get("arm") != arm or row.get("previous_record_sha256") != previous:
            raise RuntimeError(f"Repair {arm} trace identity/chain failed.")
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if row.get("record_sha256") != canonical_json_sha256(payload):
            raise RuntimeError(f"Repair {arm} trace record hash failed.")
        if row["applied_gradient"]["tensor_count"] != 21 or not row["applied_gradient"]["all_finite"]:
            raise RuntimeError(f"Repair {arm} applied-gradient support failed.")
        if not visible_update_support_valid(
            int(row["applied_gradient"]["tensor_count"]), int(row["parameter_update"]["changed_tensor_count"])
        ):
            raise RuntimeError(f"Repair {arm} parameter-update support failed.")
        images = row["images"]
        if len(images) != 3 or any(not image["exact_s9"] or not image["structural_zero_outside_s9"] for image in images):
            raise RuntimeError(f"Repair {arm} image/S9 support failed.")
        if any(not math.isfinite(float(image[key])) for image in images for key in ("lm_loss", "lce_loss", "lce_score", "effective_support", "top11_mass", "lm_s9_norm", "lce_s9_norm")):
            raise RuntimeError(f"Repair {arm} image diagnostic is non-finite.")
        repair = row.get("repair")
        if not isinstance(repair, dict) or repair.get("variant") != arm or repair.get("heldout_input_used") is not False:
            raise RuntimeError(f"Repair {arm} decision record failed.")
        minimum_support = min(float(image["effective_support"]) for image in images)
        maximum_top11 = max(float(image["top11_mass"]) for image in images)
        risk = minimum_support < 256.0 or maximum_top11 > 0.25
        spatial = repair["training_spatial_input"]
        if not _close(spatial["minimum_effective_support"], minimum_support) or not _close(spatial["maximum_top11_mass"], maximum_top11) or spatial["early_warning"] is not risk:
            raise RuntimeError(f"Repair {arm} spatial input recomputation failed.")
        if arm == "cabg_cosine_taper":
            expected_multiplier = _expected_taper(row["block_id"])
        else:
            if not latched:
                streak = streak + 1 if risk else 0
                latched = streak >= 3
            expected_multiplier = 0.0 if latched else 0.25 if streak == 2 else 1.0
            expected_state = {"variant": arm, "risk_streak": streak, "latched": latched}
            if repair["state_after"] != expected_state:
                raise RuntimeError("Spatial-guard state transition failed independent recomputation.")
        if not _close(repair["multiplier"], expected_multiplier):
            raise RuntimeError(f"Repair {arm} multiplier differs from preregistration.")
        controller = row["controller"]
        if not _close(controller["lambda_final"], float(controller["lambda_controller"]) * expected_multiplier):
            raise RuntimeError(f"Repair {arm} applied lambda mismatch.")
        trust = float(controller["trust_ratio"])
        if trust > 0.20001 or not math.isfinite(trust):
            raise RuntimeError(f"Repair {arm} trust cap failed.")
        previous = row["record_sha256"]


def _verify_resume(run_root: Path, arm: str) -> None:
    audit = _load(run_root / arm / "phase2" / "resume-audit.json")
    if not all((audit.get("resumed"), audit.get("adapter_exact"), audit.get("master_exact"))):
        raise RuntimeError(f"Repair {arm} state resume failed.")
    if audit.get("cursor") != 12 or audit.get("rng_fingerprint") != audit.get("checkpoint_rng_fingerprint"):
        raise RuntimeError(f"Repair {arm} RNG/cursor resume failed.")
    if audit.get("optimizer_tensor_inventory_sha256") != audit.get("checkpoint_optimizer_tensor_inventory_sha256"):
        raise RuntimeError(f"Repair {arm} optimizer resume failed.")
    if audit.get("repair_state", {}).get("variant") != arm:
        raise RuntimeError(f"Repair {arm} repair-state resume failed.")


def _verify_file_audits(run_root: Path, arm: str) -> None:
    paths = [
        run_root / arm / "phase1" / "file-open-audit.json",
        run_root / arm / "phase2" / "file-open-audit.json",
        run_root / "evaluation" / arm / "file-open-audit.json",
    ]
    for path in paths:
        audit = _load(path)
        if audit.get("status") != "SUCCESS" or audit.get("non_allowlisted_paths") or audit.get("protected_internal_test_outputs_read") != 0:
            raise RuntimeError(f"Repair file audit failed: {path}")


def _prediction_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
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


def _training_summary(rows: list[dict[str, Any]], run_root: Path, arm: str) -> dict[str, object]:
    supports = [float(image["effective_support"]) for row in rows for image in row["images"]]
    top11 = [float(image["top11_mass"]) for row in rows for image in row["images"]]
    last_four_failed = 0
    for row in rows[-4:]:
        block_support = median(float(image["effective_support"]) for image in row["images"])
        block_top11 = median(float(image["top11_mass"]) for image in row["images"])
        last_four_failed += int(block_support < 128.0 or block_top11 > 0.35)
    milestones = []
    for cursor in (4, 8, 12, 16, 20, 24):
        row = rows[cursor - 1]
        milestones.append({
            "cursor": cursor,
            "minimum_effective_support": min(float(image["effective_support"]) for image in row["images"]),
            "median_effective_support": median(float(image["effective_support"]) for image in row["images"]),
            "maximum_top11_mass": max(float(image["top11_mass"]) for image in row["images"]),
            "median_top11_mass": median(float(image["top11_mass"]) for image in row["images"]),
            "lambda_controller": float(row["controller"]["lambda_controller"]),
            "lambda_applied": float(row["controller"]["lambda_final"]),
            "repair_multiplier": float(row["repair"]["multiplier"]),
            "trust_ratio": float(row["controller"]["trust_ratio"]),
            "cap_active": bool(row["controller"]["cap_active"]),
            "clip_scale": float(row["gradient"]["clip_scale"]),
        })
    monitors = [_load(run_root / arm / phase / "monitor-summary.json") for phase in ("phase1", "phase2")]
    first_active = next((row["block_id"] + 1 for row in rows if row["repair"]["active"]), None)
    first_zero = next((row["block_id"] + 1 for row in rows if float(row["repair"]["multiplier"]) == 0.0), None)
    return {
        "blocks": 24,
        "images": 72,
        "clip_count": sum(float(row["gradient"]["clip_scale"]) < 1.0 - 1e-12 for row in rows),
        "clip_frequency": mean(float(row["gradient"]["clip_scale"]) < 1.0 - 1e-12 for row in rows),
        "preclip_norm_max": max(float(row["gradient"]["preclip_full_norm"]) for row in rows),
        "clip_scale_min": min(float(row["gradient"]["clip_scale"]) for row in rows),
        "cap_activation_count": sum(bool(row["controller"]["cap_active"]) for row in rows),
        "trust_ratio_max": max(float(row["controller"]["trust_ratio"]) for row in rows),
        "lambda_controller_mean": mean(float(row["controller"]["lambda_controller"]) for row in rows),
        "lambda_applied_mean": mean(float(row["controller"]["lambda_final"]) for row in rows),
        "lm_rms_mean": mean(float(row["class_rms"]["lm_balanced"]) for row in rows),
        "lce_rms_mean": mean(float(row["class_rms"]["lce_balanced"]) for row in rows),
        "effective_support_min": min(supports),
        "effective_support_median": median(supports),
        "support_below_128_count": sum(value < 128.0 for value in supports),
        "top11_mass_max": max(top11),
        "top11_mass_median": median(top11),
        "top11_above_0_35_count": sum(value > 0.35 for value in top11),
        "last_four_failed_blocks": last_four_failed,
        "repair_first_active_block": first_active,
        "repair_first_zero_block": first_zero,
        "external_peak_mib": max(int(row["peak_memory_used_mib"]) for row in monitors),
        "block_seconds_total": sum(float(row["runtime"]["block_seconds"]) for row in rows),
        "block_images_per_second": 72.0 / sum(float(row["runtime"]["block_seconds"]) for row in rows),
        "milestones": milestones,
    }


def classify(*, metrics: dict[str, object], training: dict[str, object], lm_metrics: dict[str, Any], original_metrics: dict[str, Any]) -> dict[str, object]:
    auroc = float(metrics["auroc"])
    ap = float(metrics["average_precision"])
    spatial = metrics["spatial"]
    retained_auroc = (auroc - float(lm_metrics["auroc"])) / (float(original_metrics["auroc"]) - float(lm_metrics["auroc"]))
    retained_ap = (ap - float(lm_metrics["average_precision"])) / (float(original_metrics["average_precision"]) - float(lm_metrics["average_precision"]))
    benefit_pass = auroc >= AUROC_RETENTION_FLOOR and ap >= AP_RETENTION_FLOOR and retained_auroc >= 0.9 and retained_ap >= 0.9
    spatial_pass = (
        float(spatial["effective_support_median"]) >= SUPPORT_MEDIAN_FLOOR
        and int(spatial["support_below_128_count"]) <= MAX_SPATIAL_CROSSINGS
        and int(spatial["top11_above_0_35_count"]) <= MAX_SPATIAL_CROSSINGS
        and int(training["last_four_failed_blocks"]) <= MAX_LAST_FOUR_FAILED_BLOCKS
    )
    above_baseline = auroc > float(lm_metrics["auroc"]) and ap > float(lm_metrics["average_precision"])
    if benefit_pass and spatial_pass:
        decision = "A"
    elif spatial_pass and above_baseline:
        decision = "B"
    elif benefit_pass and not spatial_pass:
        decision = "C"
    else:
        decision = "D"
    return {
        "decision": decision,
        "benefit_retention_pass": benefit_pass,
        "spatial_stability_pass": spatial_pass,
        "above_lm_only_on_both_metrics": above_baseline,
        "retained_gain_fraction": {"auroc": retained_auroc, "average_precision": retained_ap},
        "thresholds": {
            "auroc": AUROC_RETENTION_FLOOR,
            "average_precision": AP_RETENTION_FLOOR,
            "support_median": SUPPORT_MEDIAN_FLOOR,
            "maximum_spatial_crossings": MAX_SPATIAL_CROSSINGS,
            "maximum_last_four_failed_blocks": MAX_LAST_FOUR_FAILED_BLOCKS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.table.exists():
        raise FileExistsError("Refusing to overwrite repair summary evidence.")
    split = _load(args.split_audit)
    if any(value for family in split["overlap"].values() for value in family.values()):
        raise RuntimeError("Repair split overlap is nonzero.")
    if split.get("protected_internal_test_image_files_opened") != 0 or split.get("protected_internal_test_outputs_read") != 0:
        raise RuntimeError("Repair protected boundary failed.")
    original = _load(args.original_root / "comparison.json")
    expected_original = {"cabg_lce": (0.9708333333333333, 0.9848684210526315), "lm_only": (0.22916666666666666, 0.4884123424270804)}
    for arm, expected in expected_original.items():
        if not _close(original["metrics"][arm]["auroc"], expected[0]) or not _close(original["metrics"][arm]["average_precision"], expected[1]):
            raise RuntimeError("Original Scout reference identity changed.")
    original_trace = _rows(args.original_root / "cabg_lce" / "trace.jsonl")
    original_predictions = _rows(args.original_root / "evaluation" / "cabg_lce" / "predictions.jsonl")
    result_metrics: dict[str, object] = {}
    training: dict[str, object] = {}
    decisions: dict[str, object] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    prediction_hashes: dict[str, str] = {}
    for arm in REPAIR_ARMS:
        trace = _rows(args.run_root / arm / "trace.jsonl")
        traces[arm] = trace
        _verify_trace(trace, arm)
        _verify_resume(args.run_root, arm)
        _verify_file_audits(args.run_root, arm)
        if [row["ordered_samples"] for row in trace] != [row["ordered_samples"] for row in original_trace]:
            raise RuntimeError(f"Repair {arm} exposure order differs from original CABG.")
        for key in ("base_checkpoint_fingerprint", "train_manifest_sha256", "eval_manifest_sha256", "matched_configuration_sha256"):
            if {row["provenance"][key] for row in trace} != {original_trace[0]["provenance"][key]}:
                raise RuntimeError(f"Repair {arm} matched identity differs: {key}")
        if {row["initial_state_sha256"] for row in trace} != {original_trace[0]["initial_state_sha256"]}:
            raise RuntimeError(f"Repair {arm} initialization differs.")
        if [row["optimizer"]["learning_rate_used"] for row in trace] != [row["optimizer"]["learning_rate_used"] for row in original_trace]:
            raise RuntimeError(f"Repair {arm} learning-rate schedule differs.")
        predictions_path = args.run_root / "evaluation" / arm / "predictions.jsonl"
        predictions = _rows(predictions_path)
        if [(row["sample_id"], row["sha256"], row["label"]) for row in predictions] != [(row["sample_id"], row["sha256"], row["label"]) for row in original_predictions]:
            raise RuntimeError(f"Repair {arm} held-out identity/order differs.")
        result_metrics[arm] = _prediction_summary(predictions)
        reported = _load(args.run_root / "evaluation" / arm / "metrics.json")["metrics"]
        for key in ("auroc", "average_precision"):
            if not _close(result_metrics[arm][key], reported[key]):  # type: ignore[index]
                raise RuntimeError(f"Repair {arm} metric recomputation failed: {key}")
        training[arm] = _training_summary(trace, args.run_root, arm)
        if training[arm]["external_peak_mib"] > 22500 or training[arm]["trust_ratio_max"] > 0.20001:  # type: ignore[index]
            raise RuntimeError(f"Repair {arm} hard resource/trust gate failed.")
        decisions[arm] = classify(
            metrics=result_metrics[arm],  # type: ignore[arg-type]
            training=training[arm],  # type: ignore[arg-type]
            lm_metrics=original["metrics"]["lm_only"],
            original_metrics=original["metrics"]["cabg_lce"],
        )
        prediction_hashes[arm] = sha256_file(predictions_path)
    labels = [decisions[arm]["decision"] for arm in REPAIR_ARMS]  # type: ignore[index]
    overall = "A" if "A" in labels else "B" if "B" in labels else "C" if "C" in labels else "D"
    output = {
        "status": "SUCCESS",
        "decision": overall,
        "claim_scope": "development_only_stability_repair",
        "variant_decisions": decisions,
        "metrics": {
            "initial": original["metrics"]["initial"],
            "lm_only": original["metrics"]["lm_only"],
            "original_cabg": original["metrics"]["cabg_lce"],
            **result_metrics,
        },
        "training_dynamics": training,
        "fairness": {
            "same_initialization": True,
            "same_72_exposures_and_order": True,
            "same_train_eval_manifests": True,
            "same_t21_optimizer_scheduler_steps_clip": True,
            "only_registered_lce_repair_differs": True,
            "heldout_used_for_training_control": False,
        },
        "data_isolation": {
            "all_registered_overlaps_zero": True,
            "d3r_used": False,
            "d4_pilot_used": False,
            "protected_internal_test_image_files_opened": 0,
            "protected_internal_test_outputs_read": 0,
        },
        "source_commit": traces[REPAIR_ARMS[0]][0]["provenance"]["source_commit"],
        "implementation_fingerprint": traces[REPAIR_ARMS[0]][0]["provenance"]["implementation_fingerprint"],
        "split_audit_sha256": sha256_file(args.split_audit),
        "original_comparison_sha256": sha256_file(args.original_root / "comparison.json"),
        "prediction_sha256": prediction_hashes,
        "scientific_claims_not_authorized": ["formal_effectiveness", "generalization", "patient_independence", "clinical_validity", "protected_internal_test_performance", "optimal_stopping"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    with args.table.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "auroc", "average_precision", "support_median", "support_below_128", "top11_median", "top11_above_0_35", "last_four_failed_blocks", "retained_auroc_gain", "retained_ap_gain", "decision"])
        for arm in REPAIR_ARMS:
            metric = result_metrics[arm]
            spatial = metric["spatial"]  # type: ignore[index]
            decision = decisions[arm]
            writer.writerow([arm, metric["auroc"], metric["average_precision"], spatial["effective_support_median"], spatial["support_below_128_count"], spatial["top11_mass_median"], spatial["top11_above_0_35_count"], training[arm]["last_four_failed_blocks"], decision["retained_gain_fraction"]["auroc"], decision["retained_gain_fraction"]["average_precision"], decision["decision"]])
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
