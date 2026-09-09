#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import torch

from reproduction.stage2h.evidence import compute_evidence_margin_loss


def expect_failure(callable_, fragment: str) -> str:
    try:
        callable_()
    except RuntimeError as exc:
        message = str(exc)
        if fragment not in message:
            raise AssertionError(f"Expected {fragment!r}, got {message!r}") from exc
        return message
    raise AssertionError(f"Expected failure containing {fragment!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Keep both samples inside the margin so this mathematical fixture audits
    # the active squared-hinge gradient rather than a correctly zero gradient
    # for already-separated examples.
    raw = torch.linspace(-0.05, 0.05, 2 * 4 * 1024, dtype=torch.float64).reshape(2, 4, 1024)
    evidence = raw.clone().requires_grad_(True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    result = compute_evidence_margin_loss(evidence, labels)
    if result["k"] != 11 or result["positions"] != 1024:
        raise AssertionError("ceil(0.01*1024) did not resolve to k=11.")
    if result["loss"].dtype != torch.float32 or not torch.isfinite(result["loss"]):
        raise AssertionError("Evidence loss is not finite FP32.")
    expected_layer_mean = raw.float().mean(dim=1)
    expected_scores = torch.topk(expected_layer_mean, k=11, dim=1).values.mean(dim=1)
    expected_targets = labels.float().mul(2).sub(1)
    expected_loss = torch.relu(torch.tensor(0.1) - expected_targets * expected_scores).square().mean()
    torch.testing.assert_close(result["image_scores"], expected_scores, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result["loss"], expected_loss, rtol=0.0, atol=0.0)
    result["loss"].backward()
    if evidence.grad is None or not torch.isfinite(evidence.grad).all() or evidence.grad.abs().max() == 0:
        raise AssertionError("Evidence loss did not produce a finite effective gradient.")

    one_position = compute_evidence_margin_loss(
        torch.zeros((1, 1, 1), requires_grad=True),
        torch.tensor([1]),
    )
    if one_position["k"] != 1:
        raise AssertionError("Top-k floor did not preserve k=1.")

    failures = {
        "shape": expect_failure(
            lambda: compute_evidence_margin_loss(torch.zeros(2, 4), labels),
            "shape",
        ),
        "alignment": expect_failure(
            lambda: compute_evidence_margin_loss(torch.zeros(2, 4, 8), torch.tensor([1])),
            "count mismatch",
        ),
        "floating_label": expect_failure(
            lambda: compute_evidence_margin_loss(torch.zeros(2, 4, 8), labels.float()),
            "integer",
        ),
        "invalid_label": expect_failure(
            lambda: compute_evidence_margin_loss(torch.zeros(2, 4, 8), torch.tensor([0, 2])),
            "outside",
        ),
        "nonfinite": expect_failure(
            lambda: compute_evidence_margin_loss(
                torch.tensor([[[math.nan]]]), torch.tensor([1])
            ),
            "NaN or Inf",
        ),
    }
    payload = {
        "status": "SUCCESS",
        "input_shape": list(evidence.shape),
        "layer_reduction": "mean",
        "k": result["k"],
        "margin": 0.1,
        "loss_dtype": str(result["loss"].dtype).removeprefix("torch."),
        "loss_finite": True,
        "gradient_finite_nonzero": True,
        "alignment_failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
