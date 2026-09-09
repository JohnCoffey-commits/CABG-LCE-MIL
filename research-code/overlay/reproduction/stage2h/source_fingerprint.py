#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "qwen-vl-finetune/qwenvl/data/__init__.py",
    "qwen-vl-finetune/qwenvl/data/data_qwen.py",
    "qwen-vl-finetune/qwenvl/train/argument.py",
    "qwen-vl-finetune/qwenvl/train/anomaly_evidence_trainer.py",
    "qwen-vl-finetune/qwenvl/train/train_qwen.py",
    "qwen-vl-finetune/qwenvl/train/trainable_checkpoint.py",
    "reproduction/stage2h/evidence.py",
    "reproduction/stage2h/artifact_schema_v2_2.py",
    "reproduction/stage2h/adapter_schema.py",
    "reproduction/stage2h/state_audit.py",
    "reproduction/stage2h/runtime.py",
    "reproduction/stage2h/calibrate_lambda.py",
    "reproduction/stage2h/build_recalibration_dataset_v2_2.py",
    "reproduction/stage2h/preflight_v2_2.py",
    "reproduction/stage2h/candidate_measurements_v2_2.py",
    "reproduction/stage2h/evaluate_stage2h.py",
    "reproduction/stage2h/compare_states.py",
    "reproduction/stage2h/verify_engineering_gate.py",
    "reproduction/stage2h/finalize_calibration_pause.py",
    "reproduction/stage2h/finalize_calibration_pause_v2_2.py",
    "reproduction/stage2h/run_engineering_gate_l4.sh",
    "reproduction/stage2h/plan/protocol-amendment-v2.2.md",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_source_record(root: Path):
    files = []
    for relative_path in SOURCE_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative_path, "sha256": file_sha256(path)})
    payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {"files": files, "fingerprint": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(implementation_source_record(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
