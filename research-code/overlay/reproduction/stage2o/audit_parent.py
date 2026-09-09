#!/usr/bin/env python3
"""Read-only audit of the immutable original CABG block-12 branch point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2o import (
    PARENT_CHECKPOINT_SHA256,
    PARENT_EVAL_MANIFEST_SHA256,
    PARENT_MANIFEST_SHA256,
    PARENT_RNG_FINGERPRINT,
    PARENT_TRACE_ANCHOR,
    PARENT_TRAIN_MANIFEST_SHA256,
)
from reproduction.stage2o.causal import load_locked_parent


def _load(path: Path):
    return json.loads(path.read_text())


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--midpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload, manifest = load_locked_parent(args.checkpoint)
    split = _load(args.split_audit)
    if split.get("train_manifest_sha256") != PARENT_TRAIN_MANIFEST_SHA256 or split.get("eval_manifest_sha256") != PARENT_EVAL_MANIFEST_SHA256:
        raise RuntimeError("Causal split identity changed.")
    if any(value for family in split["overlap"].values() for value in family.values()):
        raise RuntimeError("Causal split overlap is nonzero.")
    if split.get("protected_internal_test_image_files_opened") != 0 or split.get("protected_internal_test_outputs_read") != 0:
        raise RuntimeError("Causal protected boundary failed.")
    trace = _rows(args.original_root / "cabg_lce" / "trace.jsonl")
    if len(trace) != 24 or trace[11]["record_sha256"] != PARENT_TRACE_ANCHOR or trace[11]["rng"]["after_block"] != PARENT_RNG_FINGERPRINT:
        raise RuntimeError("Original CABG trace does not match the parent checkpoint.")
    continuation = trace[12:]
    exposures = [sample for row in continuation for sample in row["ordered_samples"]]
    train_manifest = _rows(Path(split["train_manifest"]))
    expected = [{"sample_id": row["sample_id"], "exposure_id": row["scout_exposure_id"], "sha256": row["sha256"], "label": row["scout_label"]} for row in train_manifest[36:72]]
    if exposures != expected:
        raise RuntimeError("Original CABG continuation order differs from the locked manifest.")
    predictions_path = args.midpoint_root / "cabg_lce_block12" / "predictions.jsonl"
    predictions = _rows(predictions_path)
    if len(predictions) != 32:
        raise RuntimeError("Original block-12 held-out prediction count changed.")
    metrics = summarize_metrics(predictions)
    supports = [float(row["effective_support"]) for row in predictions]
    top11 = [float(row["top11_mass"]) for row in predictions]
    result = {
        "status": "SUCCESS",
        "claim_scope": "development_only_shared_checkpoint_causal_mechanism",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
        "checkpoint_bytes": manifest["checkpoint_bytes"],
        "tensor_inventory_sha256": manifest["tensor_inventory_sha256"],
        "cursor": payload["sampler_state"]["cursor"],
        "trace_anchor": payload["trace_state"]["last_record_sha256"],
        "rng_fingerprint": PARENT_RNG_FINGERPRINT,
        "train_manifest_sha256": PARENT_TRAIN_MANIFEST_SHA256,
        "eval_manifest_sha256": PARENT_EVAL_MANIFEST_SHA256,
        "continuation_blocks": len(continuation),
        "continuation_exposures": len(exposures),
        "continuation_order_sha256": canonical_json_sha256(exposures),
        "controller_state": payload["controller_state"],
        "scheduler_state": payload["scheduler_state"],
        "adapter_tensor_count": len(payload["adapter_state"]),
        "master_tensor_count": len(payload["master_state"]),
        "optimizer_parameter_state_count": len(payload["optimizer_state"]["state"]),
        "midpoint": {
            "metrics": metrics,
            "effective_support_median": median(supports),
            "support_below_128_count": sum(value < 128.0 for value in supports),
            "top11_mass_median": median(top11),
            "top11_above_0_35_count": sum(value > 0.35 for value in top11),
            "predictions_sha256": sha256_file(predictions_path),
        },
        "all_registered_overlaps_zero": True,
        "protected_internal_test_image_files_opened": 0,
        "protected_internal_test_outputs_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
