#!/usr/bin/env python3
"""Summarize the one allowed read-only midpoint stability supplement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

from reproduction.stage2i.fingerprint import git_identity, implementation_source_record, sha256_file
from reproduction.stage2m.constants import SOURCE_FILES
from reproduction.stage2m.metrics import paired_stratified_bootstrap, summarize_metrics


def _load(path: Path):
    return json.loads(path.read_text())


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _spatial(rows):
    supports = [float(row["effective_support"]) for row in rows]
    top11 = [float(row["top11_mass"]) for row in rows]
    return {
        "effective_support_min": min(supports),
        "effective_support_median": median(supports),
        "support_below_128_count": sum(value < 128 for value in supports),
        "top11_mass_max": max(top11),
        "top11_mass_median": median(top11),
        "top11_above_0_35_count": sum(value > 0.35 for value in top11),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--cabg-midpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-midpoint-dir", type=Path, required=True)
    parser.add_argument("--final-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    directories = {
        "initial": args.initial_dir,
        "cabg_midpoint": args.cabg_midpoint_dir,
        "lm_only_midpoint": args.baseline_midpoint_dir,
    }
    predictions = {name: _rows(path / "predictions.jsonl") for name, path in directories.items()}
    identities = [[(row["sample_id"], row["sha256"], row["label"]) for row in values] for values in predictions.values()]
    if not all(value == identities[0] for value in identities[1:]):
        raise RuntimeError("Midpoint supplement evaluation identities differ.")
    for name in ("cabg_midpoint", "lm_only_midpoint"):
        report = _load(directories[name] / "metrics.json")
        if report.get("checkpoint_cursor") != 12 or report.get("status") != "SUCCESS":
            raise RuntimeError(f"Midpoint supplement checkpoint identity failed: {name}")
    metrics = {name: summarize_metrics(rows) for name, rows in predictions.items()}
    delta = {
        key: float(metrics["cabg_midpoint"][key]) - float(metrics["lm_only_midpoint"][key])
        for key in ("auroc", "average_precision")
    }
    versus_initial = {
        key: float(metrics["cabg_midpoint"][key]) - float(metrics["initial"][key])
        for key in ("auroc", "average_precision")
    }
    result = {
        "status": "SUCCESS",
        "supplement": "read_only_block12_checkpoint_evaluation",
        "reason": "Clarify whether the block-24 CABG effect signal appears before the observed late spatial concentration.",
        "claim_scope": "development_only_exploratory_stability_followup",
        "metrics": metrics,
        "cabg_midpoint_minus_lm_only_midpoint": delta,
        "cabg_midpoint_minus_initial": versus_initial,
        "spatial": {name: _spatial(rows) for name, rows in predictions.items()},
        "paired_bootstrap": paired_stratified_bootstrap(predictions["cabg_midpoint"], predictions["lm_only_midpoint"], replicates=2000, seed=20260903),
        "final_block24_reference": _load(args.final_comparison),
        "evaluator_identity": {
            "repository": git_identity(args.repo_root),
            "source": implementation_source_record(args.repo_root, SOURCE_FILES),
        },
        "prediction_sha256": {name: sha256_file(path / "predictions.jsonl") for name, path in directories.items()},
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
        "scientific_claims_not_authorized": ["formal_effectiveness", "generalization", "patient_independence", "clinical_validity", "training_length_selection"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({key: result[key] for key in ("status", "metrics", "cabg_midpoint_minus_lm_only_midpoint", "cabg_midpoint_minus_initial", "spatial")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
