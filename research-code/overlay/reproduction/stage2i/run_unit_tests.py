#!/usr/bin/env python3
"""Run Gate D2 tests with only the Python standard unittest runner."""

import argparse
import io
import json
import sys
import unittest
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = unittest.defaultTestLoader.discover(
        str(args.repo_root / "reproduction/stage2i/tests"), pattern="test_*.py"
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = stream.getvalue()
    sys.stdout.write(report)
    payload = {
        "status": "SUCCESS" if result.wasSuccessful() else "FAILED",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
