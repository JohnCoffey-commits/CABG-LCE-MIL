#!/usr/bin/env python3
"""Independent Gate D2 verifier.  It does not import or load the 7B model."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.constants import (
    DATASET_SCHEMA_VERSION,
    LOCKED_PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    REVIEWED_PROTOCOL_SHA256,
)
from reproduction.stage2i.fingerprint import git_identity, implementation_source_record, sha256_file


SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "qwen-vl-finetune/qwenvl/train/anomaly_evidence_trainer.py",
    "qwen-vl-finetune/qwenvl/train/trainable_checkpoint.py",
    "reproduction/stage2f/adapter_schema.py",
    "reproduction/stage2f/fixtures/pre-change/stage2e-seed42-adapter-manifest.json",
    "reproduction/stage2h/evidence.py",
    "reproduction/stage2i/__init__.py",
    "reproduction/stage2i/README.md",
    "reproduction/stage2i/constants.py",
    "reproduction/stage2i/cabg_math.py",
    "reproduction/stage2i/rng_state.py",
    "reproduction/stage2i/fingerprint.py",
    "reproduction/stage2i/block_sampler.py",
    "reproduction/stage2i/build_dataset_manifest.py",
    "reproduction/stage2i/artifact_schema.py",
    "reproduction/stage2i/checkpoint.py",
    "reproduction/stage2i/toy_engine.py",
    "reproduction/stage2i/toy_resume_worker.py",
    "reproduction/stage2i/run_unit_tests.py",
    "reproduction/stage2i/verify_gate_d2.py",
    "reproduction/stage2i/run_gate_d2.sh",
    "reproduction/stage2i/tests/test_cabg_math.py",
    "reproduction/stage2i/tests/test_sampler_rng_fingerprint.py",
    "reproduction/stage2i/tests/test_artifact_checkpoint.py",
    "reproduction/stage2i/tests/test_static_contracts.py",
    "reproduction/stage2i/tests/__init__.py",
)


def _verify_dataset(audit: dict[str, object]) -> None:
    if audit.get("status") != "SUCCESS" or audit.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise CABGContractError("CABG dataset-lock status/schema mismatch.")
    expected_counts = {
        "stage2g-source": 228,
        "excluded-stage2h": 62,
        "development-pool": 166,
        "training-development": 132,
        "threshold-validation": 34,
        "mechanism-diagnostic": 12,
    }
    if audit.get("counts") != expected_counts:
        raise CABGContractError(f"CABG dataset counts changed: {audit.get('counts')}")
    if any(
        int(audit.get(key, -1)) != 0
        for key in (
            "image_files_opened",
            "internal_test_image_files_opened",
            "internal_test_outputs_read",
        )
    ):
        raise CABGContractError("CABG dataset lock accessed prohibited image/output content.")
    for name, path_text in audit["output_paths"].items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != audit["output_sha256"][name]:
            raise CABGContractError(f"CABG dataset artifact hash mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--unit-test-result", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_hash = sha256_file(args.protocol)
    if protocol_hash != LOCKED_PROTOCOL_SHA256:
        raise CABGContractError("Canonical CABG protocol hash differs from the locked constant.")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if lock.get("canonical_protocol_sha256") != LOCKED_PROTOCOL_SHA256:
        raise CABGContractError("Protocol-lock record differs from the canonical protocol.")
    if lock.get("reviewed_protocol_sha256") != REVIEWED_PROTOCOL_SHA256:
        raise CABGContractError("Protocol-lock record differs from the user-reviewed protocol.")
    unit_tests = json.loads(args.unit_test_result.read_text(encoding="utf-8"))
    if (
        unit_tests.get("status") != "SUCCESS"
        or int(unit_tests.get("tests_run", 0)) < 20
        or any(int(unit_tests.get(key, -1)) != 0 for key in ("failures", "errors", "unexpected_successes"))
    ):
        raise CABGContractError(f"Gate D2 unit-test result is insufficient: {unit_tests}")
    dataset = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    _verify_dataset(dataset)
    source = implementation_source_record(args.repo_root, SOURCE_FILES)
    payload = {
        "status": "SUCCESS",
        "decision": "PASS_GATE_D2",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_hash,
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "unit_tests": unit_tests,
        "dataset": {
            "audit_sha256": sha256_file(args.dataset_audit),
            "development_fingerprint": dataset["development_fingerprint"],
            "split_fingerprint": dataset["split_fingerprint"],
            "counts": dataset["counts"],
            "patient_independence_verified": dataset["patient_independence_verified"],
        },
        "source": source,
        "git": git_identity(args.repo_root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "real_model_loaded": False,
        "optimizer_training_run": False,
        "gpu_experiment_run": False,
        "next_eligible_gate": "D3_REAL_MODEL_READ_ONLY_MECHANISM_DIAGNOSTIC",
        "real_model_execution_authorized_by_this_run": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
