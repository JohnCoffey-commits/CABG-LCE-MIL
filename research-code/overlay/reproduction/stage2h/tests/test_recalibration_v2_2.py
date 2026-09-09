#!/usr/bin/env python3

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from reproduction.stage2h.artifact_schema_v2_2 import (
    RECALIBRATION_CANDIDATES,
    build_candidate_measurements,
    validate_dataset_audit,
)
from reproduction.stage2h.build_recalibration_dataset_v2_2 import main as _builder_import_check
from reproduction.stage2h.calibrate_lambda import rng_fingerprint, rng_restore, rng_snapshot
from reproduction.stage2h.evidence import LambdaCalibrationAccumulator


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(2, 2)
        self.key_proj = nn.Linear(2, 2)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        anomaly = nn.Module()
        anomaly.abnormal_prompt = nn.Parameter(torch.tensor([[[0.2, -0.4]]]))
        anomaly.normal_prompt = nn.Parameter(torch.tensor([[[-0.3, 0.5]]]))
        anomaly.anomaly_attention = ToyAttention()
        visual = nn.Module()
        visual.anomaly_qformer = anomaly
        inner = nn.Module()
        inner.visual = visual
        self.model = inner


def common_loss(model, scale, value):
    return scale * sum((parameter * value).square().mean() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if _builder_import_check is None:
        raise AssertionError("v2.2 dataset builder import failed")

    model = ToyModel()
    accumulator = LambdaCalibrationAccumulator(model)
    for index in range(12):
        value = 0.5 + index / 12
        accumulator.add(
            lm_loss=common_loss(model, 1.0, value),
            evidence_loss=common_loss(model, 100.0, value),
        )
    core = accumulator.finalize(candidates=RECALIBRATION_CANDIDATES)
    if core["logical_batch_size"] != 12 or core["selected_lambda"] != 1e-3:
        raise AssertionError(f"v2.2 selection rule failed: {core['selected_lambda']}")
    selected = [row for row in core["candidates"] if row["selected"]]
    if len(selected) != 1 or not selected[0]["eligible"]:
        raise AssertionError("v2.2 selection is not unique and eligible")

    before = rng_snapshot()
    before_fingerprint = rng_fingerprint(before)
    random.random()
    np.random.random()
    torch.rand(2)
    rng_restore(before)
    if rng_fingerprint(rng_snapshot()) != before_fingerprint:
        raise AssertionError("v2.2 RNG isolation test failed")

    calibration = {
        "status": "SUCCESS",
        "protocol_version": "2.2",
        "candidate_set": list(RECALIBRATION_CANDIDATES),
        "rng_isolation_passed": True,
        "target_gradient_gate_passed": True,
        "gate_and_downstream_isolation_passed": True,
        "microbatch_observations": [
            {"lm_loss": 1.0, "evidence_loss": 1.0} for _ in range(12)
        ],
        "calibration": core,
        "peak_cuda_memory_allocated_mib": 10.0,
        "peak_cuda_memory_reserved_mib": 11.0,
    }
    measurements = build_candidate_measurements(calibration, 12.0)
    if measurements["selected_lambda"] != 1e-3 or measurements["decision"] != (
        "PROCEED_TO_MATCHED_ENGINEERING_GATE"
    ):
        raise AssertionError("v2.2 candidate artifact schema failed")

    audit = {
        "status": "SUCCESS",
        "protocol_version": "2.2",
        "counts": {"train": 16, "previous-calibration": 4, "calibration": 12, "internal-test": 46},
        "class_counts": {"calibration": {"good": 6, "ungood": 6}},
        "manifest_sha256": {},
        "annotation_sha256": {},
        "recalibration_is_strict_training_subset": True,
        "recalibration_previous_calibration_overlap": 0,
        "calibration_partition_equals_training": True,
        "train_internal_test_overlap": 0,
        "recalibration_internal_test_overlap": 0,
        "internal_test_used_for_selection": False,
    }
    validate_dataset_audit(audit)
    with tempfile.TemporaryDirectory() as directory:
        if not Path(directory).is_dir():
            raise AssertionError("temporary artifact test directory failed")

    payload = {
        "status": "SUCCESS",
        "protocol_version": "2.2",
        "candidate_set_exact": list(RECALIBRATION_CANDIDATES),
        "logical_batch_size": core["logical_batch_size"],
        "unique_selected_lambda": core["selected_lambda"],
        "rng_restore_exact": True,
        "artifact_schema_validation": True,
        "dataset_schema_validation": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
