"""Matched LM-only composition used by the Scout baseline."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from reproduction.stage2i.cabg_math import (
    CABGContractError,
    aggregate_class_balanced_gradients,
    aggregate_lm_gradients,
    gradient_norm,
    per_image_class_balanced_rms,
    validate_block_labels,
)
from reproduction.stage2l.constants import EPSILON, GLOBAL_CLIP_NORM


def compute_lm_only_block(
    per_image_lm: Sequence[Mapping[str, torch.Tensor]],
    per_image_lce: Sequence[Mapping[str, torch.Tensor]],
    labels: Sequence[object],
    *,
    trainable_names: Sequence[str],
    s9_names: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    normalized = validate_block_labels(labels)
    if len(per_image_lm) != len(normalized) or len(per_image_lce) != len(normalized):
        raise CABGContractError("Scout LM-only gradient/sample counts differ.")
    if any(set(row) != set(trainable_names) for row in per_image_lm):
        raise CABGContractError("Scout LM-only support differs from T21.")
    if any(set(row) != set(s9_names) for row in per_image_lce):
        raise CABGContractError("Scout diagnostic LCE support differs from S9.")
    lm_s9 = [{name: row[name] for name in s9_names} for row in per_image_lm]
    lm_rms, lm_class = per_image_class_balanced_rms(lm_s9, normalized, epsilon=EPSILON)
    lce_rms, lce_class = per_image_class_balanced_rms(per_image_lce, normalized, epsilon=EPSILON)
    aggregate_lm = aggregate_lm_gradients(per_image_lm)
    aggregate_lce = aggregate_class_balanced_gradients(per_image_lce, normalized)
    preclip = float(gradient_norm(aggregate_lm))
    clip_scale = min(1.0, GLOBAL_CLIP_NORM / (preclip + 1e-12))
    applied = {name: value.float() * clip_scale for name, value in aggregate_lm.items()}
    if not all(torch.isfinite(value).all() for value in applied.values()):
        raise CABGContractError("Scout LM-only applied gradient is non-finite.")
    lm_s9_norm = float(gradient_norm({name: aggregate_lm[name] for name in s9_names}))
    return applied, {
        "labels": list(normalized),
        "lm_rms": float(lm_rms),
        "lce_rms": float(lce_rms),
        "lm_class_rms": {key: float(value) for key, value in lm_class.items()},
        "lce_class_rms": {key: float(value) for key, value in lce_class.items()},
        "aggregate_lm": aggregate_lm,
        "aggregate_lce": aggregate_lce,
        "budget": {
            "lm_ema": 0.0,
            "lce_ema": 0.0,
            "corrected_lm_rms": 0.0,
            "corrected_lce_rms": 0.0,
            "lm_s9_norm": lm_s9_norm,
            "lce_s9_norm": float(gradient_norm(aggregate_lce)),
            "lambda_raw": 0.0,
            "lambda_cap": 0.0,
            "lambda_final": 0.0,
            "valid_blocks": 0,
        },
        "gradient": {
            "lm_shared_norm": lm_s9_norm,
            "scaled_auxiliary_shared_norm": 0.0,
            "trust_cap_tolerance": max(1e-8, 1e-5 * lm_s9_norm),
            "preclip_full_norm": preclip,
            "clip_scale": clip_scale,
            "postclip_full_norm": float(gradient_norm(applied)),
            "cap_checked_before_clip": True,
        },
    }
