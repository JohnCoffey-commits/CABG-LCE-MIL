#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

from reproduction.stage2g.evaluation_core import sha256_file


SOURCE_FILES = (
    "reproduction/stage2g/evaluation_core.py",
    "reproduction/stage2g/build_brainmri_manifest.py",
    "reproduction/stage2g/build_run_registry.py",
    "reproduction/stage2g/evaluate_adapter.py",
    "reproduction/stage2g/evaluate_r0.py",
    "reproduction/stage2g/preflight.py",
    "reproduction/stage2g/verify_run.py",
    "reproduction/stage2g/aggregate_pilot.py",
    "reproduction/stage2g/verify_aggregate.py",
    "reproduction/stage2g/run_stage2g_l4.sh",
)


def stage2g_source_fingerprint(repo_root: Path) -> str:
    records = []
    for relative_path in SOURCE_FILES:
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(f"{relative_path}:{sha256_file(path)}")
    payload = ("\n".join(records) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(stage2g_source_fingerprint(args.repo_root.resolve()))


if __name__ == "__main__":
    main()
