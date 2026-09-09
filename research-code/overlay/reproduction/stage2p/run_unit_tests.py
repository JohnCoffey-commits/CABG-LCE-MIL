"""All existing Scout regressions plus fixed-cutoff tests, without model loads."""
import argparse
import json
import unittest
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ("reproduction.stage2m.tests.test_scout", "reproduction.stage2n.tests.test_repair", "reproduction.stage2o.tests.test_causal", "reproduction.stage2p.tests.test_fixed_cutoff"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    with args.output.open("x") as stream:
        json.dump({"status": "SUCCESS" if result.wasSuccessful() else "FAILED", "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped)}, stream, indent=2)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
