#!/usr/bin/env python3
"""Compact no-model-load preflight for D4-Scout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import git_identity, implementation_source_record, sha256_file
from reproduction.stage2i.gate_d3_preflight import checkpoint_shard_audit
from reproduction.stage2m.constants import SOURCE_FILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = json.loads(args.split_audit.read_text())
    repository = git_identity(args.repo_root)
    if repository["dirty"]:
        raise CABGContractError("D4-Scout execution source must be committed and clean.")
    if split.get("status") != "SUCCESS" or split.get("decision") != "D4_SCOUT_SPLIT_LOCKED":
        raise CABGContractError("D4-Scout split is not locked.")
    if split.get("blocks") != 24 or split.get("eval_count") != 32 or split.get("train_exposure_count") != 72:
        raise CABGContractError("D4-Scout split counts changed.")
    if any(value for family in split.get("overlap", {}).values() for value in family.values()):
        raise CABGContractError("D4-Scout split overlap is nonzero.")
    if split.get("protected_internal_test_image_files_opened") != 0 or split.get("protected_internal_test_outputs_read") != 0:
        raise CABGContractError("D4-Scout protected boundary crossed before execution.")
    source = implementation_source_record(args.repo_root, SOURCE_FILES)
    checkpoint = checkpoint_shard_audit(args.base_model)
    free = shutil.disk_usage("/home").free
    if free < 5 * 1024**3:
        raise CABGContractError("D4-Scout requires at least 5 GiB free on /home.")
    result = {
        "status": "SUCCESS",
        "decision": "READY_FOR_D4_SCOUT_MODEL_LOAD",
        "repository": repository,
        "source": source,
        "base_checkpoint": checkpoint,
        "split_audit_sha256": sha256_file(args.split_audit),
        "home_free_bytes": free,
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
