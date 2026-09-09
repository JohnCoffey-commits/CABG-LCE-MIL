#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


MAX_RUNTIME_RATIO = 1.10
MAX_EXTERNAL_VRAM_MIB = 22_835


def load_verified(path: Path, expected_mode: str):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "SUCCESS" or result.get("anomaly_query_mode") != expected_mode:
        raise RuntimeError(f"Invalid Engineering Gate run verification: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--a3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Engineering Gate summary: {args.output}")

    b0 = load_verified(args.b0, "single")
    a3 = load_verified(args.a3, "multiscale")
    if b0["implementation_source_fingerprint"] != a3["implementation_source_fingerprint"]:
        raise RuntimeError("B0 and A3 used different implementation source fingerprints.")
    runtime_ratio = a3["total_elapsed_seconds"] / b0["total_elapsed_seconds"]
    if runtime_ratio > MAX_RUNTIME_RATIO:
        raise RuntimeError(
            f"A3/B0 total runtime ratio exceeds {MAX_RUNTIME_RATIO}: {runtime_ratio}"
        )
    observed_peak = max(
        b0["training_peak_vram_used_mib"],
        b0["generation_peak_vram_used_mib"],
        a3["training_peak_vram_used_mib"],
        a3["generation_peak_vram_used_mib"],
    )
    if observed_peak > MAX_EXTERNAL_VRAM_MIB:
        raise RuntimeError("Engineering Gate pair exceeds the L4 external VRAM guardrail.")

    result = {
        "status": "SUCCESS",
        "engineering_gate": "PASS",
        "b0_run_id": b0["run_id"],
        "a3_run_id": a3["run_id"],
        "implementation_source_fingerprint": b0["implementation_source_fingerprint"],
        "runtime_ratio_a3_over_b0": runtime_ratio,
        "runtime_ratio_limit": MAX_RUNTIME_RATIO,
        "maximum_external_vram_used_mib": observed_peak,
        "external_vram_limit_mib": MAX_EXTERNAL_VRAM_MIB,
        "b0": b0,
        "a3": a3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
