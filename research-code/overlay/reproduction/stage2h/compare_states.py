#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from reproduction.stage2h.state_audit import compare_initial_and_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_initial_and_final(args.initial, args.final)
    if not result["optimizer_changed_at_least_one_target"]:
        raise RuntimeError("Stage 2H optimizer did not change any evidence target parameter.")
    if not result["gate_scale_change"]["initial_finite_positive"] or not result["gate_scale_change"]["final_finite_positive"]:
        raise RuntimeError("Stage 2H gate_scale is not finite and positive.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
