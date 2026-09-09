#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path


METRIC_KEYS = (
    "overall_exact_match_accuracy",
    "closed_accuracy",
    "open_exact_match",
    "open_token_f1",
    "invalid_closed_responses",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate evidence: {args.output}")

    expected = (("seed-42", 42), ("seed-123", 123), ("seed-2026", 2026))
    runs = []
    references = []
    for run_id, seed in expected:
        path = args.generation_dir / f"{run_id}-generation.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "SUCCESS" or result.get("seed") != seed:
            raise RuntimeError(f"Invalid generation result: {path}")
        if result.get("official_test_downloaded_or_used") is not False:
            raise RuntimeError("Official test data must remain unused.")
        if result.get("metrics", {}).get("total") != 16:
            raise RuntimeError(f"Expected 16 predictions: {path}")
        if result.get("mode") != "trained_adapter" or not result.get("adapter_sha256"):
            raise RuntimeError(f"Expected a verified trained adapter result: {path}")
        runs.append(result)

        reference_path = args.reference_dir / f"reference-seed-{seed}-generation.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if (
            reference.get("status") != "SUCCESS"
            or reference.get("mode") != "untrained_reference"
            or reference.get("seed") != seed
        ):
            raise RuntimeError(f"Reference generation result is invalid: {reference_path}")
        if reference.get("metrics", {}).get("total") != 16:
            raise RuntimeError(f"Reference generation must contain 16 predictions: {reference_path}")
        references.append(reference)

    prediction_ids = [[record["id"] for record in run["predictions"]] for run in runs]
    if any(ids != prediction_ids[0] for ids in prediction_ids[1:]):
        raise RuntimeError("Multi-seed runs evaluated different sample orders.")
    for reference in references:
        reference_ids = [record["id"] for record in reference["predictions"]]
        if reference_ids != prediction_ids[0]:
            raise RuntimeError("Reference and trained runs evaluated different sample orders.")

    trained_aggregate = {}
    reference_aggregate = {}
    paired_delta_aggregate = {}
    for key in METRIC_KEYS:
        trained_values = [float(run["metrics"][key]) for run in runs]
        reference_values = [float(reference["metrics"][key]) for reference in references]
        paired_deltas = [trained - untrained for trained, untrained in zip(trained_values, reference_values)]
        for target, values in (
            (trained_aggregate, trained_values),
            (reference_aggregate, reference_values),
            (paired_delta_aggregate, paired_deltas),
        ):
            target[key] = {
                "values": values,
                "mean": statistics.fmean(values),
                "population_std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
    result = {
        "status": "SUCCESS",
        "seeds": [seed for _, seed in expected],
        "run_ids": [run_id for run_id, _ in expected],
        "validation_records": 16,
        "official_test_downloaded_or_used": False,
        "reference_seeds": [reference["seed"] for reference in references],
        "reference_metrics_by_seed": {
            str(reference["seed"]): reference["metrics"] for reference in references
        },
        "reference_aggregate": reference_aggregate,
        "trained_aggregate": trained_aggregate,
        "paired_delta_aggregate": paired_delta_aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
