"""Producer-side CABG-LCE-MIL v1.2 D3R mathematics."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from reproduction.stage2k.constants import ABNORMAL_LABEL, POSITIONS, TAU
from reproduction.stage2k.manifest import ManifestContractError, normalize_label


def compute_lce(
    abnormal_raw: torch.Tensor,
    normal_raw: torch.Tensor,
    labels: Sequence[object],
) -> dict[str, torch.Tensor | str]:
    if abnormal_raw.shape != normal_raw.shape or abnormal_raw.ndim != 3:
        raise ManifestContractError("LCE requires aligned [images,layers,positions] raw logits.")
    if abnormal_raw.shape[-1] != POSITIONS or not abnormal_raw.is_floating_point():
        raise ManifestContractError("LCE raw-logit shape or dtype changed.")
    if len(labels) != abnormal_raw.shape[0]:
        raise ManifestContractError("LCE labels are misaligned.")
    if not torch.isfinite(abnormal_raw).all() or not torch.isfinite(normal_raw).all():
        raise ManifestContractError("LCE raw logits contain non-finite values.")
    working_gap = abnormal_raw.float() - normal_raw.float()
    contrast = working_gap / (1.0 + working_gap.abs())
    derivative = 1.0 / (1.0 + working_gap.abs()).square()
    layer_mean = contrast.mean(dim=1)
    score = TAU * (torch.logsumexp(layer_mean / TAU, dim=-1) - math.log(POSITIONS))
    targets = torch.tensor(
        [1.0 if normalize_label(value) == ABNORMAL_LABEL else -1.0 for value in labels],
        dtype=score.dtype,
        device=score.device,
    )
    loss = F.softplus(-targets * score)
    weights = torch.softmax(layer_mean / TAU, dim=-1)
    tiny = torch.finfo(weights.dtype).tiny
    entropy = -(weights * weights.clamp_min(tiny).log()).sum(-1)
    positive = torch.where(weights > 0, weights, torch.full_like(weights, float("inf")))
    values = (working_gap, contrast, derivative, layer_mean, score, loss, weights, entropy)
    if any(not torch.isfinite(value).all() for value in values):
        raise ManifestContractError("LCE producer formula contains non-finite values.")
    return {
        "mechanism": "fp32_gap_softsign_layer_mean_normalized_lse_symmetric_softplus",
        "raw_gap": working_gap,
        "contrast": contrast,
        "softsign_derivative": derivative,
        "layer_mean": layer_mean,
        "score": score,
        "per_image_loss": loss,
        "spatial_weights": weights,
        "spatial_entropy": entropy,
        "effective_support": entropy.exp(),
        "top11_mass": torch.topk(weights, k=11, dim=-1).values.sum(-1),
        "max_pooling_weight": weights.max(-1).values,
        "min_nonzero_pooling_weight": positive.min(-1).values,
        "nonzero_pooling_weights": (weights > 0).sum(-1),
    }
