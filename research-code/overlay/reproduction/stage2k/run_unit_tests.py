#!/usr/bin/env python3
"""Run the isolated D3R unit, negative, and static contract tests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "reproduction/stage2k/tests", "-v"]
    completed = subprocess.run(command, cwd=args.repo_root, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    print(output, end="")
    tests_run = sum(1 for line in output.splitlines() if line.startswith("test_"))
    result = {
        "status": "SUCCESS" if completed.returncode == 0 else "FAILED",
        "schema_version": "cabg-lce-mil-v1.2-d3r-tests-1",
        "command": command,
        "returncode": completed.returncode,
        "tests_run": tests_run,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_stderr": output,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
