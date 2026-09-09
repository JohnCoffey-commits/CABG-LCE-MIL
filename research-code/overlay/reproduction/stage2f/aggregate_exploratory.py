#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


SEEDS = (42, 123, 2026)
REPEATS = (1, 2)
METHODS = ("B0", "A3")
HIGHER_IS_BETTER = (
    "overall_exact_match_accuracy",
    "closed_accuracy",
    "open_exact_match",
    "open_token_f1",
)
LOWER_IS_BETTER = (
    "invalid_closed_responses",
    "empty_responses",
    "eval_loss_step32",
)


def expected_run_id(seed: int, repeat: int, method: str) -> str:
    return f"seed-{seed}-repeat-{repeat}-{method.lower()}"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(run, key: str) -> float:
    if key == "eval_loss_step32":
        return float(run["verification"]["eval_losses"][-1])
    return float(run["verification"]["metrics"][key])


def aggregate_metric(runs, key: str, higher_is_better: bool):
    b0_by_seed = {
        str(seed): [metric_value(runs[(seed, repeat, "B0")], key) for repeat in REPEATS]
        for seed in SEEDS
    }
    a3_by_seed = {
        str(seed): [metric_value(runs[(seed, repeat, "A3")], key) for repeat in REPEATS]
        for seed in SEEDS
    }
    discrepancy_by_seed = {
        seed: abs(values[0] - values[1]) for seed, values in b0_by_seed.items()
    }
    discrepancy_conservative = max(discrepancy_by_seed.values())
    paired_improvement_deltas = {}
    for seed in SEEDS:
        values = []
        for repeat in REPEATS:
            b0 = metric_value(runs[(seed, repeat, "B0")], key)
            a3 = metric_value(runs[(seed, repeat, "A3")], key)
            values.append(a3 - b0 if higher_is_better else b0 - a3)
        paired_improvement_deltas[str(seed)] = values
    seed_mean_improvement = {
        seed: statistics.fmean(values) for seed, values in paired_improvement_deltas.items()
    }
    all_deltas = [value for values in paired_improvement_deltas.values() for value in values]
    return {
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "b0_values_by_seed": b0_by_seed,
        "a3_values_by_seed": a3_by_seed,
        "observed_repeat_discrepancy_by_seed": discrepancy_by_seed,
        "observed_repeat_discrepancy_conservative": discrepancy_conservative,
        "paired_improvement_deltas_by_seed": paired_improvement_deltas,
        "seed_mean_improvement": seed_mean_improvement,
        "mean_paired_improvement": statistics.fmean(all_deltas),
        "population_std_paired_improvement": statistics.pstdev(all_deltas),
        "b0_mean": statistics.fmean(value for values in b0_by_seed.values() for value in values),
        "a3_mean": statistics.fmean(value for values in a3_by_seed.values() for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.transitions):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 2F aggregate evidence: {path}")

    order_lines = [line for line in args.run_order.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(order_lines) != 13:
        raise RuntimeError("Counterbalanced run-order evidence must contain 12 runs plus its header.")

    runs = {}
    source_fingerprints = set()
    prediction_order = None
    for seed in SEEDS:
        for repeat in REPEATS:
            for method in METHODS:
                run_id = expected_run_id(seed, repeat, method)
                run_root = args.log_root / run_id
                status = (run_root / "run.status").read_text(encoding="utf-8").strip()
                verification = load_json(run_root / "run-verification.json")
                generation = load_json(run_root / "generation.json")
                if status != "SUCCESS" or verification.get("status") != "SUCCESS":
                    raise RuntimeError(f"Exploratory run is not verified: {run_id}")
                if verification.get("run_id") != run_id or generation.get("run_id") != run_id:
                    raise RuntimeError(f"Exploratory run identity mismatch: {run_id}")
                if verification.get("seed") != seed or verification.get("repeat") != repeat:
                    raise RuntimeError(f"Exploratory seed/repeat mismatch: {run_id}")
                if verification.get("method") != method:
                    raise RuntimeError(f"Exploratory method mismatch: {run_id}")
                if generation.get("official_test_downloaded_or_used") is not False:
                    raise RuntimeError("Official VQA-RAD test data must remain unused.")
                ids = verification.get("prediction_ids")
                if prediction_order is None:
                    prediction_order = ids
                elif ids != prediction_order:
                    raise RuntimeError("Exploratory runs used different validation sample orders.")
                source_fingerprints.add(verification["implementation_source_fingerprint"])
                runs[(seed, repeat, method)] = {
                    "verification": verification,
                    "generation": generation,
                }
    if len(runs) != 12 or len(source_fingerprints) != 1:
        raise RuntimeError("Expected 12 runs with one implementation source fingerprint.")

    metric_aggregates = {}
    for key in HIGHER_IS_BETTER:
        metric_aggregates[key] = aggregate_metric(runs, key, higher_is_better=True)
    for key in LOWER_IS_BETTER:
        metric_aggregates[key] = aggregate_metric(runs, key, higher_is_better=False)

    transitions = []
    transition_counts = Counter()
    for seed in SEEDS:
        for repeat in REPEATS:
            b0_predictions = {
                record["id"]: record
                for record in runs[(seed, repeat, "B0")]["generation"]["predictions"]
            }
            a3_predictions = {
                record["id"]: record
                for record in runs[(seed, repeat, "A3")]["generation"]["predictions"]
            }
            if list(b0_predictions) != list(a3_predictions):
                raise RuntimeError("Paired B0/A3 prediction IDs or order differ.")
            for sample_id in b0_predictions:
                b0 = b0_predictions[sample_id]
                a3 = a3_predictions[sample_id]
                correctness_key = "closed_correct" if b0["answer_type"] == "closed" else "open_exact_match"
                b0_correct = bool(b0[correctness_key])
                a3_correct = bool(a3[correctness_key])
                transition = f"{'correct' if b0_correct else 'incorrect'}_to_{'correct' if a3_correct else 'incorrect'}"
                transition_counts[transition] += 1
                transitions.append(
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "sample_id": sample_id,
                        "answer_type": b0["answer_type"],
                        "b0_correct": b0_correct,
                        "a3_correct": a3_correct,
                        "transition": transition,
                        "b0_response": b0["response"],
                        "a3_response": a3["response"],
                    }
                )

    overall = metric_aggregates["overall_exact_match_accuracy"]
    discrepancy = overall["observed_repeat_discrepancy_conservative"]
    mean_delta = overall["mean_paired_improvement"]
    seed_means = overall["seed_mean_improvement"]
    positive_seed_count = sum(value > 0 for value in seed_means.values())
    materially_negative_seed_count = sum(value < -discrepancy for value in seed_means.values())
    b0_empty = sum(
        metric_value(runs[(seed, repeat, "B0")], "empty_responses")
        for seed in SEEDS
        for repeat in REPEATS
    )
    a3_empty = sum(
        metric_value(runs[(seed, repeat, "A3")], "empty_responses")
        for seed in SEEDS
        for repeat in REPEATS
    )
    b0_invalid = sum(
        metric_value(runs[(seed, repeat, "B0")], "invalid_closed_responses")
        for seed in SEEDS
        for repeat in REPEATS
    )
    a3_invalid = sum(
        metric_value(runs[(seed, repeat, "A3")], "invalid_closed_responses")
        for seed in SEEDS
        for repeat in REPEATS
    )
    reliability_regression = a3_empty > b0_empty or a3_invalid > b0_invalid
    if mean_delta > discrepancy and positive_seed_count >= 2 and not reliability_regression:
        classification = "positive"
    elif mean_delta < -discrepancy or materially_negative_seed_count >= 2 or reliability_regression:
        classification = "negative"
    else:
        classification = "neutral"
    if classification == "negative":
        required_next_stage = (
            "one predefined confirmation/diagnosis; pause FB-MAQ if stable "
            "method-specific degradation remains"
        )
    else:
        required_next_stage = "separately approved anomaly-specific pilot"

    paired_runtime_ratios = []
    for seed in SEEDS:
        for repeat in REPEATS:
            b0_elapsed = runs[(seed, repeat, "B0")]["verification"]["total_elapsed_seconds"]
            a3_elapsed = runs[(seed, repeat, "A3")]["verification"]["total_elapsed_seconds"]
            paired_runtime_ratios.append(a3_elapsed / b0_elapsed)
    maximum_external_vram = max(
        max(run["verification"]["training_peak_vram_used_mib"], run["verification"]["generation_peak_vram_used_mib"])
        for run in runs.values()
    )

    result = {
        "status": "SUCCESS",
        "stage": "2F-B",
        "evaluation_role": "exploratory_vqa_guardrail",
        "anomaly_effectiveness_evaluated": False,
        "official_test_downloaded_or_used": False,
        "run_count": 12,
        "seeds": list(SEEDS),
        "repeats": list(REPEATS),
        "methods": list(METHODS),
        "validation_records_per_run": 16,
        "implementation_source_fingerprint": next(iter(source_fingerprints)),
        "terminology": {
            "repeat_quantity": "observed repeat discrepancy",
            "not_allowed": [
                "noise estimate",
                "variance estimate",
                "error margin",
                "confidence interval",
                "significance threshold",
            ],
        },
        "metrics": metric_aggregates,
        "overall_vqa_classification": classification,
        "classification_inputs": {
            "mean_overall_em_paired_improvement": mean_delta,
            "overall_em_observed_repeat_discrepancy_conservative": discrepancy,
            "positive_seed_count": positive_seed_count,
            "materially_negative_seed_count": materially_negative_seed_count,
            "b0_empty_responses": b0_empty,
            "a3_empty_responses": a3_empty,
            "b0_invalid_closed_responses": b0_invalid,
            "a3_invalid_closed_responses": a3_invalid,
            "generation_reliability_regression": reliability_regression,
        },
        "transition_counts": dict(sorted(transition_counts.items())),
        "paired_runtime_ratios_a3_over_b0": paired_runtime_ratios,
        "mean_runtime_ratio_a3_over_b0": statistics.fmean(paired_runtime_ratios),
        "maximum_external_vram_used_mib": maximum_external_vram,
        "allowed_claim": (
            f"FB-MAQ passed the Engineering Gate and showed a {classification} "
            "exploratory VQA signal under the compact harness."
        ),
        "required_next_stage": required_next_stage,
    }
    args.transitions.parent.mkdir(parents=True, exist_ok=True)
    args.transitions.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in transitions),
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
