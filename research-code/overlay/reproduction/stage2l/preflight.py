#!/usr/bin/env python3
"""Independent, no-model-load D4 execution preflight."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import git_identity, implementation_source_record, sha256_file
from reproduction.stage2i.gate_d3_preflight import checkpoint_shard_audit
from reproduction.stage2l.constants import SOURCE_FILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--packet-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gate = json.loads(args.packet_gate.read_text())
    dataset = json.loads(args.dataset_audit.read_text())
    repository = git_identity(args.repo_root)
    if repository["dirty"]:
        raise CABGContractError("D4 execution source must be committed and clean.")
    required = {"spec_review", "quality_review", "unit_tests", "data_isolation", "source_model_identity", "independent_verification"}
    if set(gate.get("gates", {})) != required or any(gate["gates"][key] != "PASS" for key in required):
        raise CABGContractError("D4 Packet gate is not all PASS.")
    if dataset.get("status") != "SUCCESS" or dataset.get("d3r_overlap") != 0 or dataset.get("selected_count") != 12:
        raise CABGContractError("D4 dataset lock is invalid.")
    if dataset.get("protected_internal_test_image_files_opened") != 0 or dataset.get("protected_internal_test_outputs_read") != 0:
        raise CABGContractError("D4 protected boundary crossed before execution.")
    source = implementation_source_record(args.repo_root, SOURCE_FILES)
    checkpoint = checkpoint_shard_audit(args.base_model)
    free = shutil.disk_usage("/home").free
    if free < 5 * 1024**3:
        raise CABGContractError("D4 pilot requires at least 5 GiB free on /home.")
    result = {
        "status": "SUCCESS",
        "decision": "READY_FOR_D4_PILOT_MODEL_LOAD",
        "repository": repository,
        "source": source,
        "base_checkpoint": checkpoint,
        "dataset_audit_sha256": sha256_file(args.dataset_audit),
        "packet_gate_sha256": sha256_file(args.packet_gate),
        "home_free_bytes": free,
        "packet_gates": gate["gates"],
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
