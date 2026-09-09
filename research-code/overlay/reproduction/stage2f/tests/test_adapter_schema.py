#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from qwenvl.train.trainable_checkpoint import (
    ADAPTER_SCHEMA_VERSION,
    _parameter_fingerprint,
    load_trainable_checkpoint,
    save_trainable_checkpoint,
)


class ToyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable_a = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.trainable_b = nn.Parameter(torch.tensor([[3.0], [4.0]]))
        self.frozen = nn.Parameter(torch.tensor([5.0]), requires_grad=False)


def expect_failure(callable_, expected_fragment: str) -> str:
    try:
        callable_()
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        if expected_fragment not in message:
            raise AssertionError(
                f"Expected failure containing {expected_fragment!r}, got {message!r}"
            ) from exc
        return message
    raise AssertionError(f"Expected failure containing {expected_fragment!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.work_dir.exists():
        raise FileExistsError("Refusing to overwrite adapter-schema test evidence.")
    args.work_dir.mkdir(parents=True)

    metadata = {
        "anomaly_query_mode": "single",
        "base_model_revision": "test-revision",
        "base_checkpoint_fingerprint": "test-base-fingerprint",
        "dataset_manifest_sha256": "test-dataset-sha",
        "implementation_source_fingerprint": "test-source-fingerprint",
        "source_base_commit": "580abb4",
        "num_pooling_size": 4,
        "output_token_count": 16,
        "method_id": "medic-ad-b0",
        "gate_type": "none",
        "gate_initial_lambda": None,
        "seed": 42,
        "data_seed": 42,
        "global_step": 2,
        "train_type": "default",
        "full_determinism": False,
        "tune_mm_vpt": True,
        "tune_mm_anomaly": True,
    }
    checkpoint_path = args.work_dir / "toy.safetensors"
    source = ToyAdapterModel()
    manifest = save_trainable_checkpoint(
        SimpleNamespace(model=source),
        str(checkpoint_path),
        metadata=metadata,
    )
    expected_names = ["trainable_a", "trainable_b"]
    if manifest.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise AssertionError("Adapter export did not use the current schema version.")
    if not manifest.get("trainable_key_fingerprint"):
        raise AssertionError("Adapter export omitted its trainable-key fingerprint.")

    target = ToyAdapterModel()
    with torch.no_grad():
        target.trainable_a.zero_()
        target.trainable_b.zero_()
    loaded = load_trainable_checkpoint(
        target,
        str(checkpoint_path),
        strict_schema=True,
        expected_metadata=metadata,
        expected_parameter_names=expected_names,
    )
    if loaded["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise AssertionError("Strict loader returned inconsistent manifest data.")
    if not torch.equal(target.trainable_a, source.trainable_a):
        raise AssertionError("Strict loader did not restore trainable_a exactly.")
    if not torch.equal(target.trainable_b, source.trainable_b):
        raise AssertionError("Strict loader did not restore trainable_b exactly.")

    metadata_failure = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyAdapterModel(),
            str(checkpoint_path),
            strict_schema=True,
            expected_metadata={**metadata, "anomaly_query_mode": "multiscale"},
            expected_parameter_names=expected_names,
        ),
        "metadata mismatch",
    )
    parameter_set_failure = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyAdapterModel(),
            str(checkpoint_path),
            strict_schema=True,
            expected_metadata=metadata,
            expected_parameter_names=[*expected_names, "missing_gate.weight"],
        ),
        "parameter-set mismatch",
    )
    unexpected_parameter_failure = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyAdapterModel(),
            str(checkpoint_path),
            strict_schema=True,
            expected_metadata=metadata,
            expected_parameter_names=["trainable_a"],
        ),
        "parameter-set mismatch",
    )
    missing_expected_names_failure = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyAdapterModel(),
            str(checkpoint_path),
            strict_schema=True,
            expected_metadata=metadata,
        ),
        "expected_parameter_names",
    )

    shape_mismatch_path = args.work_dir / "shape-mismatch.safetensors"
    shutil.copyfile(checkpoint_path, shape_mismatch_path)
    shape_mismatch_manifest = json.loads(
        checkpoint_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    shape_mismatch_manifest["checkpoint_path"] = str(shape_mismatch_path)
    shape_mismatch_manifest["parameters"][0]["shape"] = [999]
    shape_mismatch_manifest["trainable_key_fingerprint"] = _parameter_fingerprint(
        shape_mismatch_manifest["parameters"]
    )
    shape_mismatch_path.with_suffix(".manifest.json").write_text(
        json.dumps(shape_mismatch_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shape_mismatch_failure = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyAdapterModel(),
            str(shape_mismatch_path),
            strict_schema=True,
            expected_metadata=metadata,
            expected_parameter_names=expected_names,
        ),
        "Manifest shape mismatch",
    )

    result = {
        "status": "SUCCESS",
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "strict_round_trip_exact": True,
        "trainable_key_fingerprint": manifest["trainable_key_fingerprint"],
        "metadata_mismatch_rejected": metadata_failure,
        "parameter_set_mismatch_rejected": parameter_set_failure,
        "unexpected_parameter_rejected": unexpected_parameter_failure,
        "missing_expected_names_rejected": missing_expected_names_failure,
        "shape_mismatch_rejected": shape_mismatch_failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
