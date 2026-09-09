"""Exploratory candidate mathematics; no training-engine implementation."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import ABNORMAL_LABEL, POSITIONS, TAU


CANDIDATE_ID = "LCE"


def compute_softsign_logit_contrast(
    abnormal_raw: torch.Tensor,
    normal_raw: torch.Tensor,
    labels: Sequence[object] | torch.Tensor,
) -> dict[str, torch.Tensor | str]:
    """Temperature-free logit contrast with the locked M2 pool/loss.

    This function exists only for read-only mechanism diagnostics.  It does
    not modify the model forward or define a training implementation.
    """

    if abnormal_raw.shape != normal_raw.shape or abnormal_raw.ndim != 3:
        raise CABGContractError("LCE requires aligned A/N raw logits [images,layers,positions].")
    if abnormal_raw.shape[-1] != POSITIONS or not abnormal_raw.is_floating_point():
        raise CABGContractError("LCE raw-logit shape or dtype is invalid.")
    if not torch.isfinite(abnormal_raw).all() or not torch.isfinite(normal_raw).all():
        raise CABGContractError("LCE raw logits contain NaN or Inf.")
    raw_labels = labels.detach().cpu().reshape(-1).tolist() if isinstance(labels, torch.Tensor) else list(labels)
    if len(raw_labels) != abnormal_raw.shape[0]:
        raise CABGContractError("LCE labels and images differ in count.")
    normalized = tuple(normalize_label(value) for value in raw_labels)
    working_gap = abnormal_raw.float() - normal_raw.float()
    contrast = working_gap / (1.0 + working_gap.abs())
    layer_mean = contrast.mean(dim=1)
    score = TAU * (torch.logsumexp(layer_mean / TAU, dim=-1) - math.log(POSITIONS))
    targets = torch.tensor(
        [1.0 if value == ABNORMAL_LABEL else -1.0 for value in normalized],
        dtype=score.dtype,
        device=score.device,
    )
    per_image_loss = F.softplus(-targets * score)
    spatial_weights = torch.softmax(layer_mean / TAU, dim=-1)
    tiny = torch.finfo(spatial_weights.dtype).tiny
    entropy = -(spatial_weights * spatial_weights.clamp_min(tiny).log()).sum(-1)
    positive = torch.where(
        spatial_weights > 0,
        spatial_weights,
        torch.full_like(spatial_weights, float("inf")),
    )
    result = {
        "mechanism": "softsign_raw_logit_contrast_normalized_logsumexp_softplus",
        "raw_gap": working_gap,
        "contrast": contrast,
        "score": score,
        "per_image_loss": per_image_loss,
        "spatial_weights": spatial_weights,
        "spatial_entropy": entropy,
        "effective_support": entropy.exp(),
        "top11_mass": torch.topk(spatial_weights, k=11, dim=-1).values.sum(-1),
        "max_pooling_weight": spatial_weights.max(-1).values,
        "min_nonzero_pooling_weight": positive.min(-1).values,
        "nonzero_pooling_weights": (spatial_weights > 0).sum(-1),
    }
    if not torch.isfinite(score).all() or not torch.isfinite(per_image_loss).all():
        raise CABGContractError("LCE score/loss is NaN or Inf.")
    return result
