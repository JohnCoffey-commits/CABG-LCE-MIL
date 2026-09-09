#!/usr/bin/env python3
"""Fail-closed pre-model-load checks for CABG-MIL v1.1 Gate D3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from reproduction.stage2f.adapter_schema import (
    BASE_CHECKPOINT_FINGERPRINT,
    BASE_MODEL_REVISION,
)
from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.constants import (
    EXPECTED_SHARED_ELEMENTS,
    EXPECTED_SHARED_TENSORS,
    EXPECTED_TRAINABLE_ELEMENTS,
    EXPECTED_TRAINABLE_TENSORS,
    LOCKED_PROTOCOL_SHA256,
    SHARED_SUPPORT_NAMES,
)
from reproduction.stage2i.fingerprint import (
    git_identity,
    implementation_source_record,
    sha256_file,
)


EXPECTED_PARENT_COMMIT = "94681810c818cc02e82dba8c8450ee604372f684"
EXPECTED_D2_DECISION_SHA256 = "02327aa791de3319335b6544368736186200c019a2972710ce0abff32363f16e"
EXPECTED_PACKET_SHA256 = "56db147af5d53157d88ba76850cb412fac4eb4debafdb9674f9cec32d9f11bd0"
EXPECTED_PACKET_LOCK_SCHEMA = "cabg-v1.1-gate-d3-packet-lock-1"
EXPECTED_MANIFEST_SHA256 = "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c"
MIN_FREE_BYTES = 10 * 1024**3

D3_SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "qwen-vl-finetune/qwenvl/data/data_qwen.py",
    "reproduction/stage2f/adapter_schema.py",
    "reproduction/stage2h/adapter_schema.py",
    "reproduction/stage2h/runtime.py",
    "reproduction/stage2h/state_audit.py",
    "reproduction/stage2i/__init__.py",
    "reproduction/stage2i/README.md",
    "reproduction/stage2i/constants.py",
    "reproduction/stage2i/cabg_math.py",
    "reproduction/stage2i/fingerprint.py",
    "reproduction/stage2i/rng_state.py",
    "reproduction/stage2i/build_gate_d3_dataset.py",
    "reproduction/stage2i/d3_artifact_schema.py",
    "reproduction/stage2i/d3_mechanisms.py",
    "reproduction/stage2i/d3_observer.py",
    "reproduction/stage2i/gate_d3_preflight.py",
    "reproduction/stage2i/run_gate_d3.py",
    "reproduction/stage2i/verify_gate_d3.py",
    "reproduction/stage2i/run_gate_d3_l4.sh",
    "reproduction/stage2i/run_unit_tests.py",
    "reproduction/stage2i/tests/__init__.py",
    "reproduction/stage2i/tests/test_d3_contracts.py",
    "reproduction/stage2i/tests/test_d3_dataset.py",
    "reproduction/stage2i/tests/test_d3_static_contracts.py",
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CABGContractError(f"Expected a JSON object: {path}")
    return value


def checkpoint_shard_audit(model_dir: Path) -> dict[str, object]:
    index_path = model_dir / "model.safetensors.index.json"
    index = _load_json(index_path)
    shard_names = sorted(set(index["weight_map"].values()))
    if not shard_names:
        raise CABGContractError("Lingshu checkpoint index contains no shards.")
    shards = []
    for name in shard_names:
        path = model_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append({"file": str(name), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = "".join(f"{row['file']}:{row['sha256']}\n" for row in shards).encode("utf-8")
    return {
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "shard_count": len(shards),
        "physical_shard_bytes": sum(int(row["bytes"]) for row in shards),
        "checkpoint_shard_list_fingerprint": hashlib.sha256(payload).hexdigest(),
        "shards": shards,
    }


def _gpu_snapshot() -> dict[str, object]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise CABGContractError(f"Gate D3 requires exactly one visible GPU, got {len(output)}.")
    name, total, used, utilization = [part.strip() for part in output[0].split(",")]
    result = {
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
    }
    if "L4" not in name or result["memory_total_mib"] != 23034:
        raise CABGContractError(f"Gate D3 target GPU changed: {result}")
    if result["memory_used_mib"] > 32 or result["utilization_percent"] != 0:
        raise CABGContractError(f"Gate D3 GPU is not idle: {result}")
    return result


def _static_source_gate(repo_root: Path) -> dict[str, object]:
    execution_paths = (
        repo_root / "reproduction/stage2i/run_gate_d3.py",
        repo_root / "reproduction/stage2i/d3_mechanisms.py",
        repo_root / "reproduction/stage2i/d3_observer.py",
    )
    prohibited = (
        ".backward(",
        "torch.optim",
        "Trainer(",
        "DeepSpeed",
        "load_in_4bit",
        "load_in_8bit",
        "device_map=",
        "cpu_" + "offload",
    )
    findings = []
    for path in execution_paths:
        text = path.read_text(encoding="utf-8")
        for token in prohibited:
            if token in text:
                findings.append({"path": str(path), "token": token})
    if findings:
        raise CABGContractError(f"D3 source contains a prohibited execution path: {findings}")
    run_source = execution_paths[0].read_text(encoding="utf-8")
    required = ("torch.autograd.grad", "return_anomaly_evidence", "AnomalyAttentionObserver")
    missing = [token for token in required if token not in run_source]
    if missing:
        raise CABGContractError(f"D3 execution source lacks required read-only path: {missing}")
    return {"status": "SUCCESS", "files_checked": len(execution_paths), "prohibited_findings": []}


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() or args.source_inventory_output.exists() or args.environment_output.exists():
        raise FileExistsError("Refusing to overwrite Gate D3 preflight evidence.")
    if sha256_file(args.protocol) != LOCKED_PROTOCOL_SHA256:
        raise CABGContractError("Canonical CABG protocol SHA-256 mismatch.")
    if sha256_file(args.packet) != EXPECTED_PACKET_SHA256:
        raise CABGContractError("Gate D3 execution packet SHA-256 mismatch.")
    lock = _load_json(args.packet_lock)
    if (
        lock.get("schema_version") != EXPECTED_PACKET_LOCK_SCHEMA
        or lock.get("status") != "APPROVED_AND_LOCKED_FOR_EXECUTION"
        or lock.get("packet_sha256") != EXPECTED_PACKET_SHA256
        or lock.get("canonical_protocol_sha256") != LOCKED_PROTOCOL_SHA256
        or lock.get("parent_gate_d2_commit") != EXPECTED_PARENT_COMMIT
        or lock.get("execution_authorized") is not True
    ):
        raise CABGContractError("Gate D3 authorization lock is invalid or incomplete.")
    if sha256_file(args.d2_verification) != EXPECTED_D2_DECISION_SHA256:
        raise CABGContractError("Gate D2 decision artifact SHA-256 mismatch.")
    d2 = _load_json(args.d2_verification)
    if d2.get("status") != "SUCCESS" or d2.get("decision") != "PASS_GATE_D2":
        raise CABGContractError("Gate D2 did not pass.")
    unit_tests = _load_json(args.unit_test_result)
    if (
        unit_tests.get("status") != "SUCCESS"
        or int(unit_tests.get("tests_run", 0)) < 35
        or any(int(unit_tests.get(key, -1)) != 0 for key in ("failures", "errors", "unexpected_successes"))
    ):
        raise CABGContractError(f"Gate D3 unit-test result is insufficient: {unit_tests}")
    dataset = _load_json(args.dataset_audit)
    if (
        dataset.get("status") != "SUCCESS"
        or dataset.get("schema_version") != "cabg-v1.1-gate-d3-dataset-1"
        or dataset.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or dataset.get("locked_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or dataset.get("count") != 12
        or dataset.get("class_counts") != {"abnormal": 6, "normal": 6}
        or dataset.get("unique_identities") != 12
        or dataset.get("internal_test_image_files_opened") != 0
        or dataset.get("internal_test_outputs_read") != 0
    ):
        raise CABGContractError(f"Gate D3 dataset lock is invalid: {dataset}")
    for key in ("locked_manifest", "annotation"):
        path = Path(str(dataset[key]))
        expected_key = f"{key}_sha256"
        if not path.is_file() or sha256_file(path) != dataset[expected_key]:
            raise CABGContractError(f"Gate D3 dataset artifact changed: {key}")
    git = git_identity(args.repo_root)
    if git["dirty"]:
        raise CABGContractError("Gate D3 remote Git worktree is dirty.")
    source = implementation_source_record(args.repo_root, D3_SOURCE_FILES)
    args.source_inventory_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_inventory_output.write_text(
        json.dumps(source, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    environment = {
        "status": "SUCCESS",
        "schema_version": "cabg-v1.1-gate-d3-environment-1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "python_executable": sys.executable,
    }
    args.environment_output.write_text(
        json.dumps(environment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    revision_path = args.base_model / ".medic-ad-pinned-revision"
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != BASE_MODEL_REVISION:
        raise CABGContractError(f"Pinned Lingshu revision changed: {revision}")
    checkpoint = checkpoint_shard_audit(args.base_model)
    if checkpoint["checkpoint_shard_list_fingerprint"] != BASE_CHECKPOINT_FINGERPRINT:
        raise CABGContractError("Pinned Lingshu checkpoint fingerprint mismatch.")
    static_gate = _static_source_gate(args.repo_root)
    disk = shutil.disk_usage("/home")
    if disk.free < MIN_FREE_BYTES:
        raise CABGContractError(f"Remote /home free space is below 10GiB: {disk.free}")
    gpu = _gpu_snapshot()
    result = {
        "status": "SUCCESS",
        "decision": "READY_FOR_GATE_D3_MODEL_LOAD",
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "packet_sha256": EXPECTED_PACKET_SHA256,
        "packet_lock_sha256": sha256_file(args.packet_lock),
        "d2_verification_sha256": EXPECTED_D2_DECISION_SHA256,
        "unit_test_result_sha256": sha256_file(args.unit_test_result),
        "dataset_audit_sha256": sha256_file(args.dataset_audit),
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "git": git,
        "source": source,
        "source_inventory_sha256": sha256_file(args.source_inventory_output),
        "environment_sha256": sha256_file(args.environment_output),
        "base_model_revision": revision,
        "base_model_revision_file": str(revision_path.resolve()),
        "base_checkpoint": checkpoint,
        "trainable_scope": {
            "tensors": EXPECTED_TRAINABLE_TENSORS,
            "elements": EXPECTED_TRAINABLE_ELEMENTS,
        },
        "shared_support": {
            "names": list(SHARED_SUPPORT_NAMES),
            "tensors": EXPECTED_SHARED_TENSORS,
            "elements": EXPECTED_SHARED_ELEMENTS,
        },
        "static_source_gate": static_gate,
        "gpu": gpu,
        "home_free_bytes": disk.free,
        "home_free_gib": disk.free / 1024**3,
        "real_model_loaded": False,
        "optimizer_created": False,
        "quantization": False,
        "model_offloading": False,
        "activation_offloading": False,
        "pid": os.getpid(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-lock", type=Path, required=True)
    parser.add_argument("--d2-verification", type=Path, required=True)
    parser.add_argument("--unit-test-result", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--source-inventory-output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
