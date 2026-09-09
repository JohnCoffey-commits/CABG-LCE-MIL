#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from reproduction.stage2g.evaluation_core import load_jsonl, sha256_file


DECISION_MARGIN = 0.02


def classify_decision(pair_records):
    deltas = [record["balanced_accuracy_delta_a3_minus_b0"] for record in pair_records]
    mean_delta = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)
    positive_pairs = sum(delta > 0 for delta in deltas)
    negative_pairs = sum(delta < 0 for delta in deltas)
    invalid_regression_pairs = [
        record["pair_id"]
        for record in pair_records
        if record["a3_invalid_responses"] > record["b0_invalid_responses"]
    ]
    collapse_pairs = [
        record["pair_id"]
        for record in pair_records
        if record["a3_invalid_responses"] >= 57
        and record["a3_invalid_responses"] > record["b0_invalid_responses"]
    ]
    if (
        invalid_regression_pairs
        or collapse_pairs
        or mean_delta < -DECISION_MARGIN
        or (negative_pairs >= 4 and median_delta < 0)
    ):
        decision = "PAUSE"
        next_action = "stop_scaling_fb_maq"
    elif mean_delta > DECISION_MARGIN and positive_pairs >= 4:
        decision = "PROCEED"
        next_action = "draft_anomaly_specific_training_protocol"
    else:
        decision = "INCONCLUSIVE"
        next_action = "at_most_one_preregistered_second_dataset_replication"
    return {
        "decision": decision,
        "required_next_action": next_action,
        "decision_margin": DECISION_MARGIN,
        "mean_balanced_accuracy_delta_a3_minus_b0": mean_delta,
        "median_balanced_accuracy_delta_a3_minus_b0": median_delta,
        "positive_pair_count": positive_pairs,
        "negative_pair_count": negative_pairs,
        "invalid_regression_pairs": invalid_regression_pairs,
        "collapse_pairs": collapse_pairs,
    }


def build_aggregate(registry_path: Path, runs_root: Path):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    run_data = {}
    manifest_hashes = set()
    process_ids = []
    for registered in registry["runs"]:
        run_id = registered["run_id"]
        root = runs_root / run_id
        if (root / "run.status").read_text(encoding="utf-8").strip() != "SUCCESS":
            raise RuntimeError(f"Run status is not successful: {run_id}")
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "run-verification.json").read_text(encoding="utf-8"))
        predictions = load_jsonl(root / "predictions.jsonl")
        if result.get("status") != "SUCCESS" or verification.get("status") != "SUCCESS":
            raise RuntimeError(f"Run result/verification failed: {run_id}")
        if verification.get("result_sha256") != sha256_file(root / "result.json"):
            raise RuntimeError(f"Run result changed after verification: {run_id}")
        if verification.get("predictions_sha256") != sha256_file(root / "predictions.jsonl"):
            raise RuntimeError(f"Run predictions changed after verification: {run_id}")
        manifest_hashes.add(result["evaluation_manifest_sha256"])
        process_ids.append(result["process_id"])
        run_data[run_id] = {
            "registered": registered,
            "result": result,
            "verification": verification,
            "predictions": predictions,
        }
    if len(run_data) != 13 or len(manifest_hashes) != 1:
        raise RuntimeError("Expected 13 complete runs using one evaluation manifest.")
    if len(set(process_ids)) != 13:
        raise RuntimeError("Fresh-process invariant failed across Stage 2G runs.")

    r0 = run_data["stage2g-r0"]
    pair_records = []
    transitions = []
    transition_counts = Counter()
    for seed in (42, 123, 2026):
        for repeat in (1, 2):
            pair_id = f"seed-{seed}-repeat-{repeat}"
            b0 = run_data[f"stage2g-{pair_id}-b0"]
            a3 = run_data[f"stage2g-{pair_id}-a3"]
            b0_metrics = b0["result"]["metrics"]
            a3_metrics = a3["result"]["metrics"]
            pair_records.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "repeat": repeat,
                    "b0_balanced_accuracy": b0_metrics["balanced_accuracy"],
                    "a3_balanced_accuracy": a3_metrics["balanced_accuracy"],
                    "balanced_accuracy_delta_a3_minus_b0": (
                        a3_metrics["balanced_accuracy"] - b0_metrics["balanced_accuracy"]
                    ),
                    "b0_invalid_responses": b0_metrics["invalid_responses"],
                    "a3_invalid_responses": a3_metrics["invalid_responses"],
                    "b0_empty_responses": b0_metrics["empty_responses"],
                    "a3_empty_responses": a3_metrics["empty_responses"],
                    "b0_external_peak_vram_used_mib": b0["verification"][
                        "external_peak_vram_used_mib"
                    ],
                    "a3_external_peak_vram_used_mib": a3["verification"][
                        "external_peak_vram_used_mib"
                    ],
                    "b0_wall_elapsed_seconds": b0["verification"]["wall_elapsed_seconds"],
                    "a3_wall_elapsed_seconds": a3["verification"]["wall_elapsed_seconds"],
                }
            )
            for b0_prediction, a3_prediction in zip(b0["predictions"], a3["predictions"]):
                if b0_prediction["sample_id"] != a3_prediction["sample_id"]:
                    raise RuntimeError(f"Pair sample order mismatch: {pair_id}")
                transition = (
                    f"{'correct' if b0_prediction['correct'] else 'incorrect'}_to_"
                    f"{'correct' if a3_prediction['correct'] else 'incorrect'}"
                )
                transition_counts[transition] += 1
                transitions.append(
                    {
                        "pair_id": pair_id,
                        "seed": seed,
                        "repeat": repeat,
                        "sample_id": b0_prediction["sample_id"],
                        "image_sha256": b0_prediction["image_sha256"],
                        "label": b0_prediction["label"],
                        "ground_truth": b0_prediction["ground_truth"],
                        "b0_prediction": b0_prediction["prediction"],
                        "a3_prediction": a3_prediction["prediction"],
                        "b0_valid": b0_prediction["valid_response"],
                        "a3_valid": a3_prediction["valid_response"],
                        "b0_correct": b0_prediction["correct"],
                        "a3_correct": a3_prediction["correct"],
                        "transition": transition,
                        "b0_raw_response": b0_prediction["raw_response"],
                        "a3_raw_response": a3_prediction["raw_response"],
                    }
                )
    if len(pair_records) != 6 or len(transitions) != 1368:
        raise RuntimeError("Stage 2G pair/transition counts are incomplete.")
    decision = classify_decision(pair_records)
    aggregate = {
        "status": "SUCCESS",
        "stage": "2G",
        "evaluation_role": "anomaly_specific_internal_pilot",
        "claim_scope": "image_level_internal_pilot_only",
        "patient_independence_verified": False,
        "clinical_effectiveness_evaluated": False,
        "paper_level_effectiveness_evaluated": False,
        "run_count": 13,
        "paired_method_run_count": 12,
        "pair_count": 6,
        "evaluation_records_per_run": 228,
        "evaluation_manifest_sha256": next(iter(manifest_hashes)),
        "registry_sha256": sha256_file(registry_path),
        "primary_endpoint": "balanced_accuracy",
        "r0_contextual_reference": {
            "metrics": r0["result"]["metrics"],
            "excluded_from_fb_maq_decision": True,
            "external_peak_vram_used_mib": r0["verification"]["external_peak_vram_used_mib"],
            "wall_elapsed_seconds": r0["verification"]["wall_elapsed_seconds"],
        },
        "pairs": pair_records,
        "transition_counts": dict(sorted(transition_counts.items())),
        **decision,
        "allowed_claim": (
            f"The frozen-adapter image-level internal pilot returned {decision['decision']}; "
            "it does not establish patient-independent, clinical, or paper-level effectiveness."
        ),
    }
    return aggregate, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.transitions):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 2G aggregate evidence: {path}")
    aggregate, transitions = build_aggregate(args.registry, args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.transitions.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in transitions
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
