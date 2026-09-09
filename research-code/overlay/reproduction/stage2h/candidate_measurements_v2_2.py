#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from reproduction.stage2h.artifact_schema_v2_2 import build_candidate_measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--external-peak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    external_peak = float(args.external_peak.read_text(encoding="utf-8").strip())
    result = build_candidate_measurements(calibration, external_peak)
    if not result["external_peak_guardrail_passed"]:
        raise RuntimeError("Stage 2H v2.2 calibration exceeded the external VRAM guardrail.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
