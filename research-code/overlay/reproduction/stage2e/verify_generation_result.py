#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=("untrained_reference", "trained_adapter"), required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=16)
    parser.add_argument("--adapter-manifest", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "SUCCESS":
        raise RuntimeError("Generation result is not successful.")
    if result.get("mode") != args.expected_mode or result.get("seed") != args.expected_seed:
        raise RuntimeError("Generation mode or seed mismatch.")
    if result.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official test data must remain unused.")
    predictions = result.get("predictions", [])
    if len(predictions) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} predictions, got {len(predictions)}")
    if len({record.get("id") for record in predictions}) != len(predictions):
        raise RuntimeError("Duplicate prediction IDs detected.")
    metrics = result.get("metrics", {})
    if metrics.get("total") != args.expected_samples:
        raise RuntimeError("Metric sample count mismatch.")
    for key in ("overall_exact_match_accuracy", "closed_accuracy", "open_exact_match", "open_token_f1"):
        value = float(metrics[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Invalid metric {key}: {value}")

    if args.expected_mode == "trained_adapter":
        if args.adapter_manifest is None:
            raise RuntimeError("A trained result requires --adapter-manifest.")
        manifest = json.loads(args.adapter_manifest.read_text(encoding="utf-8"))
        if result.get("adapter_sha256") != manifest.get("checkpoint_sha256"):
            raise RuntimeError("Generation adapter SHA does not match the training manifest.")
    elif result.get("adapter") is not None or result.get("adapter_sha256") is not None:
        raise RuntimeError("Untrained reference unexpectedly records an adapter.")

    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "mode": result["mode"],
                "seed": result["seed"],
                "samples": len(predictions),
                "metrics": metrics,
                "peak_cuda_memory_allocated_mib": result["peak_cuda_memory_allocated_mib"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
