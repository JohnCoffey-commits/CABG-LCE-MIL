#!/usr/bin/env python3
"""Run targeted D4-Scout tests and write a machine-readable result."""

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = unittest.defaultTestLoader.loadTestsFromName("reproduction.stage2m.tests.test_scout")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "status": "SUCCESS" if result.wasSuccessful() else "FAILED",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
