#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from reproduction.stage2h.evidence import LambdaCalibrationAccumulator


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(2, 2)
        self.key_proj = nn.Linear(2, 2)


class ToyAnomalyQformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.abnormal_prompt = nn.Parameter(torch.tensor([[[0.2, -0.4]]]))
        self.normal_prompt = nn.Parameter(torch.tensor([[[-0.3, 0.5]]]))
        self.anomaly_attention = ToyAttention()


class ToyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.anomaly_qformer = ToyAnomalyQformer()


class ToyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = ToyVisual()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyInner()


def common_loss(model, scale: float, value: float) -> torch.Tensor:
    parameters = tuple(model.parameters())
    return scale * sum((parameter * value).square().mean() for parameter in parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(17)
    model = ToyModel()
    accumulator = LambdaCalibrationAccumulator(model)
    for value in (0.5, 1.0, 1.5, 2.0):
        lm_loss = common_loss(model, 1.0, value)
        evidence_loss = common_loss(model, 10.0, value)
        accumulator.add(lm_loss=lm_loss, evidence_loss=evidence_loss)
    result = accumulator.finalize()
    if result["logical_batch_size"] != 4 or result["microbatch_size"] != 1:
        raise AssertionError("Calibration did not lock four sequential microbatches.")
    if result["common_finite_parameter_count"] != 6:
        raise AssertionError("Calibration did not include all six target tensors.")
    if result["selected_lambda"] != 0.01:
        raise AssertionError(f"Expected lambda=0.01, got {result['selected_lambda']}")
    selected = next(row for row in result["candidates"] if row["selected"])
    torch.testing.assert_close(
        torch.tensor(selected["gradient_ratio"]),
        torch.tensor(0.1),
        rtol=1e-5,
        atol=1e-6,
    )
    if any(
        row["lm_gradient_missing_count"]
        or row["evidence_gradient_missing_count"]
        or row["lm_gradient_nonfinite_count"]
        or row["evidence_gradient_nonfinite_count"]
        for row in result["parameter_gradients"]
    ):
        raise AssertionError("Synthetic calibration unexpectedly recorded missing/non-finite gradients.")
    no_candidate_model = ToyModel()
    no_candidate = LambdaCalibrationAccumulator(no_candidate_model)
    no_candidate.add(
        lm_loss=common_loss(no_candidate_model, 1.0, 1.0),
        evidence_loss=common_loss(no_candidate_model, 1000.0, 1.0),
    )
    no_candidate_result = no_candidate.finalize()
    if no_candidate_result["status"] != "NO_ELIGIBLE_CANDIDATE":
        raise AssertionError("No-candidate calibration did not return a serializable terminal status.")
    if no_candidate_result["selected_lambda"] is not None or any(
        row["eligible"] or row["selected"] for row in no_candidate_result["candidates"]
    ):
        raise AssertionError("No-candidate calibration incorrectly selected a lambda.")
    result["no_eligible_candidate_serialized"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
