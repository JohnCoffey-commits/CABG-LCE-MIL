"""Locked read-only M0/M1/M2 mechanism definitions for Gate D3."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import ABNORMAL_LABEL, POSITIONS, TAU


MECHANISM_IDS = ("M0", "M1", "M2")
TOPK_FRACTION = 0.01
HINGE_MARGIN = 0.1


def _validated_evidence(
    anomaly_evidence: torch.Tensor, labels: Sequence[object] | torch.Tensor
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if anomaly_evidence.ndim != 3 or min(anomaly_evidence.shape) <= 0:
        raise CABGContractError(
            "D3 evidence must have shape [images,layers,positions] with positive dimensions."
        )
    if not anomaly_evidence.is_floating_point() or not torch.isfinite(anomaly_evidence).all():
        raise CABGContractError("D3 evidence must be finite floating point.")
    raw = labels.detach().cpu().reshape(-1).tolist() if isinstance(labels, torch.Tensor) else list(labels)
    if len(raw) != anomaly_evidence.shape[0]:
        raise CABGContractError("D3 evidence and label counts differ.")
    normalized = tuple(normalize_label(value) for value in raw)
    return anomaly_evidence.float(), normalized


def compute_mechanisms(
    anomaly_evidence: torch.Tensor,
    labels: Sequence[object] | torch.Tensor,
) -> dict[str, dict[str, torch.Tensor | int | str]]:
    """Compute M0/M1/M2 from one shared evidence tensor.

    No forward or parameter state is created here.  Every returned loss remains
    connected to the exact input evidence graph.
    """

    evidence, normalized = _validated_evidence(anomaly_evidence, labels)
    layer_mean = evidence.mean(dim=1)
    positions = int(layer_mean.shape[-1])
    if positions != POSITIONS:
        raise CABGContractError(f"D3 requires {POSITIONS} spatial positions, got {positions}.")
    targets = torch.tensor(
        [1.0 if value == ABNORMAL_LABEL else -1.0 for value in normalized],
        dtype=layer_mean.dtype,
        device=layer_mean.device,
    )
    k = max(1, int(math.ceil(TOPK_FRACTION * positions)))
    if k != 11:
        raise CABGContractError(f"D3 top-k contract changed: expected 11, got {k}.")
    hard_scores = torch.topk(layer_mean, k=k, dim=-1, largest=True, sorted=False).values.mean(-1)
    m0_loss = F.relu(HINGE_MARGIN - targets * hard_scores).square()
    m1_loss = F.softplus(-targets * hard_scores)
    m2_scores = TAU * (torch.logsumexp(layer_mean / TAU, dim=-1) - math.log(positions))
    m2_loss = F.softplus(-targets * m2_scores)
    spatial_weights = torch.softmax(layer_mean / TAU, dim=-1)
    tiny = torch.finfo(spatial_weights.dtype).tiny
    entropy = -(spatial_weights * spatial_weights.clamp_min(tiny).log()).sum(-1)
    top11_mass = torch.topk(spatial_weights, k=11, dim=-1).values.sum(-1)
    positive = torch.where(
        spatial_weights > 0,
        spatial_weights,
        torch.full_like(spatial_weights, float("inf")),
    )
    values = {
        "M0": {
            "mechanism": "hard_top1pct_squared_hinge",
            "score": hard_scores,
            "per_image_loss": m0_loss,
            "k": k,
        },
        "M1": {
            "mechanism": "hard_top1pct_softplus",
            "score": hard_scores,
            "per_image_loss": m1_loss,
            "k": k,
        },
        "M2": {
            "mechanism": "normalized_logsumexp_softplus",
            "score": m2_scores,
            "per_image_loss": m2_loss,
            "tau": TAU,
            "spatial_weights": spatial_weights,
            "spatial_entropy": entropy,
            "effective_support": entropy.exp(),
            "top11_mass": top11_mass,
            "max_pooling_weight": spatial_weights.max(-1).values,
            "min_nonzero_pooling_weight": positive.min(-1).values,
            "nonzero_pooling_weights": (spatial_weights > 0).sum(-1),
        },
    }
    for mechanism_id, result in values.items():
        if not torch.isfinite(result["score"]).all() or not torch.isfinite(result["per_image_loss"]).all():
            raise CABGContractError(f"D3 {mechanism_id} score/loss is NaN or Inf.")
    return values


def gradient_statistics(
    names: Sequence[str], gradients: Sequence[torch.Tensor | None]
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    if len(names) != len(gradients) or len(set(names)) != len(names):
        raise CABGContractError("D3 gradient names/values are misaligned.")
    rows = []
    present = {}
    for name, gradient in zip(names, gradients):
        if gradient is None:
            rows.append(
                {
                    "name": name,
                    "present": False,
                    "finite": True,
                    "max_abs": 0.0,
                    "l2_norm": 0.0,
                }
            )
            continue
        value = gradient.detach().float()
        finite = bool(torch.isfinite(value).all().item())
        rows.append(
            {
                "name": name,
                "present": True,
                "finite": finite,
                "max_abs": float(value.abs().max().cpu().item()) if finite else None,
                "l2_norm": float(torch.linalg.vector_norm(value.reshape(-1)).cpu().item())
                if finite
                else None,
            }
        )
        if finite:
            present[name] = value.cpu().contiguous()
    return rows, present


def global_norm(gradients: Mapping[str, torch.Tensor]) -> float:
    if not gradients:
        return 0.0
    total = torch.zeros((), dtype=torch.float64)
    for name in sorted(gradients):
        value = gradients[name].double().reshape(-1)
        if not torch.isfinite(value).all():
            raise CABGContractError(f"D3 gradient is non-finite: {name}")
        total += value.square().sum()
    return float(total.sqrt().item())


def gradient_cosine(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], *, floor: float = 1e-12
) -> dict[str, object]:
    common = sorted(set(left) & set(right))
    if not common:
        return {"defined": False, "value": None, "left_norm": 0.0, "right_norm": 0.0}
    left_norm = global_norm({name: left[name] for name in common})
    right_norm = global_norm({name: right[name] for name in common})
    if left_norm <= floor or right_norm <= floor:
        return {
            "defined": False,
            "value": None,
            "left_norm": left_norm,
            "right_norm": right_norm,
        }
    dot = sum(float((left[name].double() * right[name].double()).sum().item()) for name in common)
    value = dot / (left_norm * right_norm)
    if not math.isfinite(value):
        raise CABGContractError("D3 gradient cosine is non-finite.")
    return {
        "defined": True,
        "value": max(-1.0, min(1.0, value)),
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def cancellation_ratio(gradients: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, object]:
    if not gradients:
        raise CABGContractError("D3 cancellation ratio received no gradients.")
    names = sorted(gradients[0])
    if any(sorted(value) != names for value in gradients):
        raise CABGContractError("D3 cancellation gradients have inconsistent support.")
    per_image_norms = [global_norm(value) for value in gradients]
    mean_norm = sum(per_image_norms) / len(per_image_norms)
    aggregate = {
        name: torch.stack([value[name].float() for value in gradients], dim=0).mean(0)
        for name in names
    }
    aggregate_norm = global_norm(aggregate)
    global_ratio = aggregate_norm / mean_norm if mean_norm > 0 else None
    per_tensor = []
    for name in names:
        individual = [float(torch.linalg.vector_norm(value[name].float().reshape(-1)).item()) for value in gradients]
        denominator = sum(individual) / len(individual)
        numerator = float(torch.linalg.vector_norm(aggregate[name].reshape(-1)).item())
        per_tensor.append(
            {
                "name": name,
                "mean_individual_norm": denominator,
                "aggregate_norm": numerator,
                "ratio": numerator / denominator if denominator > 0 else None,
            }
        )
    return {
        "mean_individual_norm": mean_norm,
        "aggregate_norm": aggregate_norm,
        "ratio": global_ratio,
        "per_tensor": per_tensor,
    }
