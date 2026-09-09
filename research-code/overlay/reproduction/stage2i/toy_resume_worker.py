#!/usr/bin/env python3
"""Fresh-process half of the Gate D2 checkpoint/resume parity test."""

import argparse
from pathlib import Path

import torch

from reproduction.stage2i.checkpoint import load_checkpoint_strict
from reproduction.stage2i.toy_engine import TOY_PROVENANCE, ToyCABGEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = load_checkpoint_strict(args.checkpoint, expected_provenance=TOY_PROVENANCE)
    engine = ToyCABGEngine()
    engine.load_payload(payload)
    engine.run_block(engine.completed_blocks)
    torch.save(engine.result_state(), args.output)


if __name__ == "__main__":
    main()
