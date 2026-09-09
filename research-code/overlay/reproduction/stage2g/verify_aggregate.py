#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from reproduction.stage2g.aggregate_pilot import build_aggregate
from reproduction.stage2g.evaluation_core import load_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate verification: {args.output}")
    observed_aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    observed_transitions = load_jsonl(args.transitions)
    expected_aggregate, expected_transitions = build_aggregate(args.registry, args.runs_root)
    if observed_aggregate != expected_aggregate:
        raise RuntimeError("Stage 2G aggregate differs from independent recomputation.")
    if observed_transitions != expected_transitions:
        raise RuntimeError("Stage 2G transitions differ from independent recomputation.")
    result = {
        "status": "SUCCESS",
        "stage": "2G",
        "decision": observed_aggregate["decision"],
        "aggregate_sha256": sha256_file(args.aggregate),
        "transitions_sha256": sha256_file(args.transitions),
        "transition_records": len(observed_transitions),
        "run_count": observed_aggregate["run_count"],
        "pair_count": observed_aggregate["pair_count"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
