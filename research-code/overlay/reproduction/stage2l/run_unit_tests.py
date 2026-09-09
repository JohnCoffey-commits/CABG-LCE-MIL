#!/usr/bin/env python3
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
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).with_name("tests")), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"status": "SUCCESS" if result.wasSuccessful() else "FAILED", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
