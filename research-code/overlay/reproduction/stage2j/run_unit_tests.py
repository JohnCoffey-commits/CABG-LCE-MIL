#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
        str(args.repo_root / "reproduction/stage2j/tests"), pattern="test_*.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "status": "SUCCESS" if result.wasSuccessful() else "FAILED",
        "schema_version": "cabg-mil-v1.2-redesign-r0-unit-tests-1",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
