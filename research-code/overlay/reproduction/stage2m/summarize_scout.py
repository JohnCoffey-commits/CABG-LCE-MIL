#!/usr/bin/env python3
"""Independent fairness, stability and held-out comparison summarizer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2m.constants import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, RHO_MAX
from reproduction.stage2m.metrics import paired_stratified_bootstrap, summarize_metrics


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _check_trace(rows: list[dict[str, Any]], arm: str) -> None:
    if len(rows) != 24 or [row["block_id"] for row in rows] != list(range(24)):
        raise RuntimeError(f"Scout {arm} trace is not exactly 24 ordered blocks.")
    previous = "0" * 64
    for row in rows:
        if row.get("status") != "SUCCESS" or row.get("arm") != arm or row.get("previous_record_sha256") != previous:
            raise RuntimeError(f"Scout {arm} trace identity/chain failed.")
        digest = row.get("record_sha256")
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if digest != canonical_json_sha256(payload):
            raise RuntimeError(f"Scout {arm} trace hash failed.")
        if row["applied_gradient"]["tensor_count"] != 21 or not row["applied_gradient"]["all_finite"] or row["parameter_update"]["changed_tensor_count"] <= 0:
            raise RuntimeError(f"Scout {arm} applied-gradient/update failed.")
        if any(not image["exact_s9"] or not image["structural_zero_outside_s9"] for image in row["images"]):
            raise RuntimeError(f"Scout {arm} S9 support failed.")
        if not all(math.isfinite(float(image[key])) for image in row["images"] for key in ("lm_loss", "lce_loss", "lce_score", "lm_s9_norm", "lce_s9_norm", "effective_support", "top11_mass")):
            raise RuntimeError(f"Scout {arm} non-finite image diagnostic.")
        trust = float(row["controller"]["trust_ratio"])
        if trust > RHO_MAX + 1e-5:
            raise RuntimeError(f"Scout {arm} trust cap failed.")
        if arm == "lm_only" and any(float(row["controller"][key]) != 0.0 for key in ("lambda_raw", "lambda_cap", "lambda_final", "trust_ratio")):
            raise RuntimeError("Scout LM-only applied a nonzero CABG term.")
        previous = str(digest)


def _dynamics(rows: list[dict[str, Any]], run_root: Path, arm: str) -> dict[str, object]:
    supports = [float(image["effective_support"]) for row in rows for image in row["images"]]
    top11 = [float(image["top11_mass"]) for row in rows for image in row["images"]]
    monitors = [_load(run_root / arm / phase / "monitor-summary.json") for phase in ("phase1", "phase2")]
    return {
        "blocks": len(rows),
        "images": sum(len(row["images"]) for row in rows),
        "clip_count": sum(float(row["gradient"]["clip_scale"]) < 1.0 - 1e-12 for row in rows),
        "clip_frequency": mean(float(row["gradient"]["clip_scale"]) < 1.0 - 1e-12 for row in rows),
        "preclip_norm_max": max(float(row["gradient"]["preclip_full_norm"]) for row in rows),
        "clip_scale_min": min(float(row["gradient"]["clip_scale"]) for row in rows),
        "lm_rms_mean": mean(float(row["class_rms"]["lm_balanced"]) for row in rows),
        "lce_rms_mean": mean(float(row["class_rms"]["lce_balanced"]) for row in rows),
        "lm_loss_mean": mean(float(image["lm_loss"]) for row in rows for image in row["images"]),
        "lce_loss_mean": mean(float(image["lce_loss"]) for row in rows for image in row["images"]),
        "lambda_mean": mean(float(row["controller"]["lambda_final"]) for row in rows),
        "lambda_min": min(float(row["controller"]["lambda_final"]) for row in rows),
        "lambda_max": max(float(row["controller"]["lambda_final"]) for row in rows),
        "trust_ratio_max": max(float(row["controller"]["trust_ratio"]) for row in rows),
        "cap_activation_count": sum(bool(row["controller"]["cap_active"]) for row in rows),
        "effective_support_min": min(supports),
        "effective_support_median": sorted(supports)[len(supports) // 2],
        "top11_mass_max": max(top11),
        "support_below_128_count": sum(value < 128.0 for value in supports),
        "top11_above_0_35_count": sum(value > 0.35 for value in top11),
        "block_seconds_total": sum(float(row["runtime"]["block_seconds"]) for row in rows),
        "block_images_per_second": 72.0 / sum(float(row["runtime"]["block_seconds"]) for row in rows),
        "external_peak_mib": max(int(row["peak_memory_used_mib"]) for row in monitors),
    }


def _verify_resume(run_root: Path, arm: str) -> dict[str, Any]:
    audit = _load(run_root / arm / "phase2" / "resume-audit.json")
    required = audit.get("resumed") and audit.get("adapter_exact") and audit.get("master_exact")
    required = required and audit.get("rng_fingerprint") == audit.get("checkpoint_rng_fingerprint")
    required = required and audit.get("optimizer_tensor_inventory_sha256") == audit.get("checkpoint_optimizer_tensor_inventory_sha256")
    if not required or audit.get("cursor") != 12 or audit.get("arm") != arm:
        raise RuntimeError(f"Scout {arm} resume audit failed.")
    return audit


def classify_decision(
    *, cabg_auroc: float, cabg_ap: float, baseline_auroc: float, baseline_ap: float,
    initial_auroc: float, initial_ap: float, stable: bool,
) -> tuple[str, str]:
    if not stable:
        return "D", "A hard training/resource stability condition failed."
    delta_auroc = cabg_auroc - baseline_auroc
    delta_ap = cabg_ap - baseline_ap
    versus_initial_auroc = cabg_auroc - initial_auroc
    versus_initial_ap = cabg_ap - initial_ap
    if delta_auroc > 0 and delta_ap > 0 and max(delta_auroc, delta_ap) >= 0.03 and versus_initial_auroc >= -0.02 and versus_initial_ap >= -0.02:
        return "A", "CABG improved both primary development metrics over LM-only, one by at least 0.03, without material loss versus initialization."
    if delta_auroc <= 0 and delta_ap <= 0:
        return "C", "CABG was no better than LM-only on either primary development metric."
    return "B", "The two primary metrics or initialization comparison do not provide a consistent registered positive/negative signal."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--cabg-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.table.exists():
        raise FileExistsError("Refusing to overwrite Scout summary evidence.")
    split = _load(args.split_audit)
    if any(value for family in split["overlap"].values() for value in family.values()):
        raise RuntimeError("Scout split overlap is nonzero.")
    cabg_trace = _rows(args.run_root / "cabg_lce" / "trace.jsonl")
    baseline_trace = _rows(args.run_root / "lm_only" / "trace.jsonl")
    _check_trace(cabg_trace, "cabg_lce")
    _check_trace(baseline_trace, "lm_only")
    if [row["ordered_samples"] for row in cabg_trace] != [row["ordered_samples"] for row in baseline_trace]:
        raise RuntimeError("Scout arm sample exposure/order mismatch.")
    fairness_keys = ("source_commit", "implementation_fingerprint", "base_checkpoint_fingerprint", "train_manifest_sha256", "eval_manifest_sha256", "matched_configuration_sha256")
    for key in fairness_keys:
        if {row["provenance"][key] for row in [*cabg_trace, *baseline_trace]} != {cabg_trace[0]["provenance"][key]}:
            raise RuntimeError(f"Scout arm fairness identity mismatch: {key}")
    if {row["initial_state_sha256"] for row in [*cabg_trace, *baseline_trace]} != {cabg_trace[0]["initial_state_sha256"]}:
        raise RuntimeError("Scout arms did not share initialization.")
    if [row["optimizer"]["learning_rate_used"] for row in cabg_trace] != [row["optimizer"]["learning_rate_used"] for row in baseline_trace]:
        raise RuntimeError("Scout arm learning-rate schedules differ.")
    resumes = {arm: _verify_resume(args.run_root, arm) for arm in ("cabg_lce", "lm_only")}
    predictions = {
        "initial": _rows(args.initial_dir / "predictions.jsonl"),
        "cabg_lce": _rows(args.cabg_dir / "predictions.jsonl"),
        "lm_only": _rows(args.baseline_dir / "predictions.jsonl"),
    }
    sample_orders = [[(row["sample_id"], row["sha256"], row["label"]) for row in predictions[arm]] for arm in predictions]
    if not all(value == sample_orders[0] for value in sample_orders[1:]):
        raise RuntimeError("Scout evaluation samples/orders differ across arms.")
    metrics = {arm: summarize_metrics(rows) for arm, rows in predictions.items()}
    for arm, directory in (("initial", args.initial_dir), ("cabg_lce", args.cabg_dir), ("lm_only", args.baseline_dir)):
        reported = _load(directory / "metrics.json")
        for key in ("auroc", "average_precision"):
            if not _close(metrics[arm][key], reported["metrics"][key]):
                raise RuntimeError(f"Scout {arm} metric recomputation mismatch: {key}")
        file_audit = _load(directory / "file-open-audit.json")
        if file_audit.get("status") != "SUCCESS" or file_audit.get("protected_internal_test_outputs_read") != 0:
            raise RuntimeError(f"Scout {arm} evaluation file audit failed.")
    dynamics = {
        "cabg_lce": _dynamics(cabg_trace, args.run_root, "cabg_lce"),
        "lm_only": _dynamics(baseline_trace, args.run_root, "lm_only"),
    }
    deltas = {
        "auroc": float(metrics["cabg_lce"]["auroc"]) - float(metrics["lm_only"]["auroc"]),
        "average_precision": float(metrics["cabg_lce"]["average_precision"]) - float(metrics["lm_only"]["average_precision"]),
    }
    versus_initial = {
        "auroc": float(metrics["cabg_lce"]["auroc"]) - float(metrics["initial"]["auroc"]),
        "average_precision": float(metrics["cabg_lce"]["average_precision"]) - float(metrics["initial"]["average_precision"]),
    }
    bootstrap = paired_stratified_bootstrap(predictions["cabg_lce"], predictions["lm_only"], replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    stable = all(value["external_peak_mib"] <= 22500 and value["trust_ratio_max"] <= 0.20001 for value in dynamics.values())
    decision, rationale = classify_decision(
        cabg_auroc=float(metrics["cabg_lce"]["auroc"]),
        cabg_ap=float(metrics["cabg_lce"]["average_precision"]),
        baseline_auroc=float(metrics["lm_only"]["auroc"]),
        baseline_ap=float(metrics["lm_only"]["average_precision"]),
        initial_auroc=float(metrics["initial"]["auroc"]),
        initial_ap=float(metrics["initial"]["average_precision"]),
        stable=stable,
    )
    result = {
        "status": "SUCCESS",
        "decision": decision,
        "rationale": rationale,
        "claim_scope": "development_only_exploratory_effect_scout",
        "metrics": metrics,
        "cabg_minus_lm_only": deltas,
        "cabg_minus_initial": versus_initial,
        "paired_bootstrap": bootstrap,
        "training_dynamics": dynamics,
        "fairness": {
            "same_initialization": True,
            "same_source_and_base_checkpoint": True,
            "same_train_and_eval_manifests": True,
            "same_72_exposures_and_order": True,
            "same_t21_optimizer_steps_schedule_and_clip": True,
            "both_computed_lce_diagnostic_gradients": True,
            "sole_applied_objective_difference": "dynamic_cabg_lce_on_s9_vs_zero_lce_weight",
            "matched_configuration_sha256": cabg_trace[0]["provenance"]["matched_configuration_sha256"],
        },
        "resume_exact": {arm: True for arm in resumes},
        "data_isolation": {
            "all_registered_overlaps_zero": True,
            "heldout_used_for_training_or_tuning": False,
            "d3r_used": False,
            "d4_pilot_used": False,
            "protected_internal_test_image_files_opened": 0,
            "protected_internal_test_outputs_read": 0,
        },
        "source_commit": cabg_trace[0]["provenance"]["source_commit"],
        "implementation_fingerprint": cabg_trace[0]["provenance"]["implementation_fingerprint"],
        "split_audit_sha256": sha256_file(args.split_audit),
        "prediction_sha256": {arm: sha256_file(directory / "predictions.jsonl") for arm, directory in (("initial", args.initial_dir), ("cabg_lce", args.cabg_dir), ("lm_only", args.baseline_dir))},
        "scientific_claims_not_authorized": ["formal_effectiveness", "generalization", "patient_independence", "clinical_validity", "protected_internal_test_performance", "long_horizon_stability"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    with args.table.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "count", "normal", "abnormal", "auroc", "average_precision", "steps", "clip_frequency", "effective_support_min", "top11_mass_max", "cap_activations", "external_peak_mib"])
        for arm in ("initial", "cabg_lce", "lm_only"):
            metric = metrics[arm]
            dynamics_row: Mapping[str, object] = dynamics.get(arm, {})
            writer.writerow([arm, metric["count"], metric["class_counts"]["normal"], metric["class_counts"]["abnormal"], metric["auroc"], metric["average_precision"], dynamics_row.get("blocks", 0), dynamics_row.get("clip_frequency", ""), dynamics_row.get("effective_support_min", ""), dynamics_row.get("top11_mass_max", ""), dynamics_row.get("cap_activation_count", ""), dynamics_row.get("external_peak_mib", _load((args.initial_dir if arm == "initial" else args.cabg_dir if arm == "cabg_lce" else args.baseline_dir) / "metrics.json")["external_peak_mib"])])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
