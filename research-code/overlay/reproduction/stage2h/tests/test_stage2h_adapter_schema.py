#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from qwenvl.train.trainable_checkpoint import load_trainable_checkpoint, save_trainable_checkpoint


class ToyModel(nn.Module):
    def __init__(self, include_gate=False):
        super().__init__()
        self.adapter_a = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.adapter_b = nn.Parameter(torch.tensor([3.0]))
        if include_gate:
            self.fusion_gate = nn.Parameter(torch.tensor([0.0]))


def metadata(method_id, weight):
    return {
        "stage": "2H-E",
        "protocol_version": "2.1",
        "method_id": method_id,
        "train_type": "anomaly_evidence",
        "evidence_loss_weight": weight,
        "evidence_definition": "pre_gate_sigmoid_difference",
        "evidence_margin": 0.1,
        "evidence_topk_fraction": 0.01,
        "evidence_topk_k": 11,
        "anomaly_query_mode": "single",
        "num_pooling_size": 4,
        "output_token_count": 16,
        "gate_type": "none",
        "training_manifest_sha256": "train-hash",
        "evaluation_manifest_sha256": "eval-hash",
        "base_model_revision": "base-revision",
        "base_checkpoint_fingerprint": "base-fingerprint",
        "dataset_manifest_sha256": "train-hash",
        "implementation_source_fingerprint": "source-fingerprint",
        "run_id": "run-id",
        "claim_scope": "image_level_internal_engineering_only",
        "tune_mm_vpt": True,
        "tune_mm_anomaly": True,
        "seed": 42,
        "data_seed": 42,
        "global_step": 2,
        "full_determinism": False,
    }


def expect_failure(callable_, fragment):
    try:
        callable_()
    except RuntimeError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"Expected {fragment!r}, got {str(exc)!r}") from exc
        return str(exc)
    raise AssertionError(f"Expected failure containing {fragment!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.work_dir.exists() or args.output.exists():
        raise FileExistsError("Refusing to overwrite Stage 2H adapter test evidence.")
    args.work_dir.mkdir(parents=True)

    source = ToyModel()
    b0_path = args.work_dir / "b0.safetensors"
    b0_metadata = metadata("medic-ad-b0-as", 0.0)
    manifest = save_trainable_checkpoint(SimpleNamespace(model=source), str(b0_path), b0_metadata)
    expected_names = ["adapter_a", "adapter_b"]
    target = ToyModel()
    loaded = load_trainable_checkpoint(
        target,
        str(b0_path),
        strict_schema=True,
        expected_metadata=b0_metadata,
        expected_parameter_names=expected_names,
    )
    if loaded["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise AssertionError("Strict Stage 2H adapter round-trip changed manifest identity.")

    b0_as_lad = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyModel(),
            str(b0_path),
            strict_schema=True,
            expected_metadata=metadata("lad-mil-v2", 0.1),
            expected_parameter_names=expected_names,
        ),
        "metadata mismatch",
    )

    fb_path = args.work_dir / "fb.safetensors"
    fb = ToyModel(include_gate=True)
    save_trainable_checkpoint(
        SimpleNamespace(model=fb),
        str(fb_path),
        {**b0_metadata, "method_id": "fb-maq-stage2f", "anomaly_query_mode": "multiscale"},
    )
    fb_rejected = expect_failure(
        lambda: load_trainable_checkpoint(
            ToyModel(),
            str(fb_path),
            strict_schema=True,
            expected_metadata=b0_metadata,
            expected_parameter_names=expected_names,
        ),
        "parameter-set mismatch",
    )

    payload = {
        "status": "SUCCESS",
        "strict_round_trip_exact": True,
        "b0_as_lad_mil_rejected": b0_as_lad,
        "fb_maq_adapter_rejected": fb_rejected,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
