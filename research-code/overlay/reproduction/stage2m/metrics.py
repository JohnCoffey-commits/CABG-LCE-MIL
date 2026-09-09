"""Threshold-free image-level metrics for D4-Scout."""

from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Mapping, Sequence


def _validated(records: Sequence[Mapping[str, object]]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for row in records:
        label = str(row["label"])
        score = float(row["score"])
        if label not in {"normal", "abnormal"} or not math.isfinite(score):
            raise ValueError("Scout metrics require finite scores and normalized binary labels.")
        rows.append((label, score))
    if not rows or {label for label, _ in rows} != {"normal", "abnormal"}:
        raise ValueError("Scout metrics require both classes.")
    return rows


def auroc(records: Sequence[Mapping[str, object]]) -> float:
    rows = _validated(records)
    positive = [score for label, score in rows if label == "abnormal"]
    negative = [score for label, score in rows if label == "normal"]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def average_precision(records: Sequence[Mapping[str, object]]) -> float:
    rows = _validated(records)
    positives = sum(label == "abnormal" for label, _ in rows)
    groups: dict[float, list[str]] = {}
    for label, score in rows:
        groups.setdefault(score, []).append(label)
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        tp += sum(label == "abnormal" for label in labels)
        fp += sum(label == "normal" for label in labels)
        recall = tp / positives
        precision = tp / (tp + fp)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def summarize_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = _validated(records)
    by_class = {
        label: [score for member, score in rows if member == label]
        for label in ("normal", "abnormal")
    }
    return {
        "count": len(rows),
        "class_counts": {label: len(values) for label, values in by_class.items()},
        "auroc": auroc(records),
        "average_precision": average_precision(records),
        "score_summary": {
            label: {"mean": mean(values), "median": median(values), "min": min(values), "max": max(values)}
            for label, values in by_class.items()
        },
    }


def paired_stratified_bootstrap(
    cabg: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive.")
    cabg_by_id = {str(row["sample_id"]): row for row in cabg}
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    if set(cabg_by_id) != set(baseline_by_id):
        raise ValueError("Paired bootstrap sample identities differ.")
    labels = {sample_id: str(row["label"]) for sample_id, row in cabg_by_id.items()}
    if any(str(baseline_by_id[sample_id]["label"]) != label for sample_id, label in labels.items()):
        raise ValueError("Paired bootstrap labels differ.")
    strata = {
        label: sorted(sample_id for sample_id, member in labels.items() if member == label)
        for label in ("normal", "abnormal")
    }
    rng = random.Random(seed)
    deltas = {"auroc": [], "average_precision": []}
    for _ in range(replicates):
        sampled = [rng.choice(strata[label]) for label in ("normal", "abnormal") for _ in strata[label]]
        cabg_rows = [cabg_by_id[sample_id] for sample_id in sampled]
        baseline_rows = [baseline_by_id[sample_id] for sample_id in sampled]
        deltas["auroc"].append(auroc(cabg_rows) - auroc(baseline_rows))
        deltas["average_precision"].append(average_precision(cabg_rows) - average_precision(baseline_rows))

    def percentile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "replicates": replicates,
        "seed": seed,
        "interval": "descriptive_paired_stratified_percentile_90",
        "deltas": {
            name: {"mean": mean(values), "p05": percentile(values, 0.05), "p95": percentile(values, 0.95)}
            for name, values in deltas.items()
        },
    }
