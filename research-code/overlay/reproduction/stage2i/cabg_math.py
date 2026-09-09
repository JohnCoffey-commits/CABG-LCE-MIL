"""Pure CABG-MIL v1.1 mathematics used by Gate D2 and later engines."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from reproduction.stage2i.constants import (
    ABNORMAL_LABEL,
    EMA_BETA,
    EPSILON,
    GLOBAL_CLIP_NORM,
    LABEL_ALIASES,
    LM_REFERENCE_FLOOR,
    NORMAL_LABEL,
    RHO,
    RHO_MAX,
    TAU,
)


class CABGContractError(RuntimeError):
    """Raised when a locked CABG-MIL mechanism contract is violated."""


def normalize_label(value: object) -> str:
    if isinstance(value, bool):
        return ABNORMAL_LABEL if value else NORMAL_LABEL
    if isinstance(value, int) and value in (0, 1):
        return ABNORMAL_LABEL if value else NORMAL_LABEL
    normalized = str(value).strip().lower()
    if normalized not in LABEL_ALIASES:
        raise CABGContractError(f"Unsupported anomaly label: {value!r}")
    return LABEL_ALIASES[normalized]


def validate_block_labels(labels: Sequence[object]) -> tuple[str, ...]:
    normalized = tuple(normalize_label(value) for value in labels)
    if not normalized:
        raise CABGContractError("A logical block cannot be empty.")
    if NORMAL_LABEL not in normalized or ABNORMAL_LABEL not in normalized:
        raise CABGContractError("A logical block must contain both classes.")
    return normalized


def normalized_logsumexp_score(
    layer_mean: torch.Tensor,
    *,
    tau: float = TAU,
) -> torch.Tensor:
    if layer_mean.ndim < 1 or layer_mean.shape[-1] <= 0:
        raise CABGContractError("Layer-mean evidence must have a non-empty spatial axis.")
    if not layer_mean.is_floating_point() or not torch.isfinite(layer_mean).all():
        raise CABGContractError("Layer-mean evidence must be finite floating point.")
    if not math.isfinite(tau) or tau <= 0:
        raise CABGContractError("Pooling temperature must be finite and positive.")
    working = layer_mean if layer_mean.dtype == torch.float64 else layer_mean.float()
    positions = working.shape[-1]
    return tau * (torch.logsumexp(working / tau, dim=-1) - math.log(positions))


def smooth_evidence_objective(
    anomaly_evidence: torch.Tensor,
    labels: Sequence[object] | torch.Tensor,
    *,
    tau: float = TAU,
) -> dict[str, torch.Tensor]:
    if anomaly_evidence.ndim != 3:
        raise CABGContractError(
            "Anomaly evidence must have shape [images,layers,positions], "
            f"got {tuple(anomaly_evidence.shape)}."
        )
    if min(anomaly_evidence.shape) <= 0:
        raise CABGContractError("All anomaly-evidence dimensions must be positive.")
    if not anomaly_evidence.is_floating_point() or not torch.isfinite(anomaly_evidence).all():
        raise CABGContractError("Anomaly evidence must be finite floating point.")
    if isinstance(labels, torch.Tensor):
        raw_labels = labels.detach().cpu().reshape(-1).tolist()
    else:
        raw_labels = list(labels)
    if len(raw_labels) != anomaly_evidence.shape[0]:
        raise CABGContractError("Evidence and label counts differ.")
    normalized = tuple(normalize_label(value) for value in raw_labels)
    working_evidence = anomaly_evidence if anomaly_evidence.dtype == torch.float64 else anomaly_evidence.float()
    layer_mean = working_evidence.mean(dim=1)
    scores = normalized_logsumexp_score(layer_mean, tau=tau)
    targets = torch.tensor(
        [1.0 if value == ABNORMAL_LABEL else -1.0 for value in normalized],
        dtype=scores.dtype,
        device=scores.device,
    )
    per_image_loss = F.softplus(-targets * scores)
    # The real engine extracts one loss at a time with microbatch_size=1.
    # Class balancing is therefore applied later, across the complete logical
    # block, rather than being required inside this per-image-capable function.
    loss = per_image_loss.mean()
    spatial_weights = torch.softmax(layer_mean / tau, dim=-1)
    entropy = -(spatial_weights * spatial_weights.clamp_min(torch.finfo(spatial_weights.dtype).tiny).log()).sum(-1)
    top_count = min(11, spatial_weights.shape[-1])
    top_mass = torch.topk(spatial_weights, k=top_count, dim=-1).values.sum(-1)
    positive_weights = torch.where(
        spatial_weights > 0,
        spatial_weights,
        torch.full_like(spatial_weights, float("inf")),
    )
    return {
        "loss": loss,
        "per_image_loss": per_image_loss,
        "scores": scores,
        "layer_mean": layer_mean,
        "spatial_weights": spatial_weights,
        "spatial_entropy": entropy,
        "effective_support": entropy.exp(),
        "top11_mass": top_mass,
        "max_pooling_weight": spatial_weights.max(-1).values,
        "min_nonzero_pooling_weight": positive_weights.min(-1).values,
        "evidence_min": working_evidence.amin(dim=(1, 2)),
        "evidence_max": working_evidence.amax(dim=(1, 2)),
        "evidence_mean": working_evidence.mean(dim=(1, 2)),
        "evidence_saturation_fraction": (working_evidence.abs() >= 0.95).float().mean(dim=(1, 2)),
    }


def class_balanced_mean(values: torch.Tensor, labels: Sequence[object]) -> torch.Tensor:
    if values.ndim != 1:
        values = values.reshape(-1)
    normalized = validate_block_labels(labels)
    if len(normalized) != values.numel():
        raise CABGContractError("Value and label counts differ.")
    positive = torch.tensor(
        [value == ABNORMAL_LABEL for value in normalized], device=values.device, dtype=torch.bool
    )
    negative = ~positive
    return 0.5 * (values[positive].mean() + values[negative].mean())


def _finite_gradient_tensor(tensor: torch.Tensor, *, context: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise CABGContractError(f"{context} is not a tensor.")
    value = tensor.detach().float()
    if not torch.isfinite(value).all():
        raise CABGContractError(f"{context} contains NaN or Inf.")
    return value


def gradient_squared_norm(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise CABGContractError("Gradient mapping is empty.")
    total = torch.zeros((), dtype=torch.float32)
    for name in sorted(gradients):
        tensor = _finite_gradient_tensor(gradients[name], context=f"gradient {name}")
        total = total + tensor.square().sum().cpu()
    return total


def gradient_norm(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return gradient_squared_norm(gradients).sqrt()


def per_image_class_balanced_rms(
    per_image_gradients: Sequence[Mapping[str, torch.Tensor]],
    labels: Sequence[object],
    *,
    epsilon: float = EPSILON,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized = validate_block_labels(labels)
    if len(per_image_gradients) != len(normalized):
        raise CABGContractError("Per-image gradient and label counts differ.")
    squared = torch.stack([gradient_squared_norm(value) for value in per_image_gradients])
    class_means = {}
    for label in (NORMAL_LABEL, ABNORMAL_LABEL):
        mask = torch.tensor([value == label for value in normalized], dtype=torch.bool)
        class_means[label] = squared[mask].mean()
        if not torch.isfinite(class_means[label]) or class_means[label] <= 0:
            raise CABGContractError(f"{label} per-image gradient RMS is zero or non-finite.")
    rms = (0.5 * (class_means[NORMAL_LABEL] + class_means[ABNORMAL_LABEL]) + epsilon**2).sqrt()
    return rms, {label: value.sqrt() for label, value in class_means.items()}


def aggregate_lm_gradients(
    gradients: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not gradients:
        raise CABGContractError("LM gradient list is empty.")
    names = set(gradients[0])
    if any(set(value) != names for value in gradients):
        raise CABGContractError("LM gradient name sets differ across images.")
    return {
        name: torch.stack([_finite_gradient_tensor(value[name], context=name) for value in gradients]).mean(0)
        for name in sorted(names)
    }


def aggregate_class_balanced_gradients(
    gradients: Sequence[Mapping[str, torch.Tensor]],
    labels: Sequence[object],
) -> dict[str, torch.Tensor]:
    normalized = validate_block_labels(labels)
    if len(gradients) != len(normalized):
        raise CABGContractError("Auxiliary gradient and label counts differ.")
    names = set(gradients[0])
    if any(set(value) != names for value in gradients):
        raise CABGContractError("Auxiliary gradient name sets differ across images.")
    result = {}
    for name in sorted(names):
        class_means = []
        for label in (NORMAL_LABEL, ABNORMAL_LABEL):
            members = [
                _finite_gradient_tensor(value[name], context=name)
                for value, member_label in zip(gradients, normalized)
                if member_label == label
            ]
            class_means.append(torch.stack(members).mean(0))
        result[name] = 0.5 * (class_means[0] + class_means[1])
    return result


@dataclass
class BiasCorrectedEMA:
    beta: float = EMA_BETA
    lm: float = 0.0
    auxiliary: float = 0.0
    valid_blocks: int = 0

    def update(self, lm_value: float, auxiliary_value: float) -> tuple[float, float]:
        for name, value in (("LM", lm_value), ("auxiliary", auxiliary_value)):
            if not math.isfinite(value) or value <= 0:
                raise CABGContractError(f"{name} RMS must be finite and positive.")
        self.lm = self.beta * self.lm + (1.0 - self.beta) * lm_value
        self.auxiliary = self.beta * self.auxiliary + (1.0 - self.beta) * auxiliary_value
        self.valid_blocks += 1
        correction = 1.0 - self.beta**self.valid_blocks
        return self.lm / correction, self.auxiliary / correction

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"beta", "lm", "auxiliary", "valid_blocks"}
        if set(state) != required:
            raise CABGContractError("EMA state schema mismatch.")
        beta = float(state["beta"])
        lm = float(state["lm"])
        auxiliary = float(state["auxiliary"])
        valid_blocks = int(state["valid_blocks"])
        if beta != self.beta or valid_blocks < 0 or not all(map(math.isfinite, (lm, auxiliary))):
            raise CABGContractError("EMA state values are invalid.")
        self.lm, self.auxiliary, self.valid_blocks = lm, auxiliary, valid_blocks


def compute_budget(
    *,
    lm_rms: float,
    auxiliary_rms: float,
    lm_shared_gradient: Mapping[str, torch.Tensor],
    auxiliary_shared_gradient: Mapping[str, torch.Tensor],
    ema: BiasCorrectedEMA,
    rho: float = RHO,
    rho_max: float = RHO_MAX,
    epsilon: float = EPSILON,
) -> dict[str, float]:
    corrected_lm, corrected_auxiliary = ema.update(float(lm_rms), float(auxiliary_rms))
    lm_norm = float(gradient_norm(lm_shared_gradient))
    auxiliary_norm = float(gradient_norm(auxiliary_shared_gradient))
    if lm_norm <= LM_REFERENCE_FLOOR:
        raise CABGContractError("Aggregate shared LM reference gradient is zero or too small.")
    raw = float(rho * corrected_lm / (corrected_auxiliary + epsilon))
    cap = float(rho_max * lm_norm / (auxiliary_norm + epsilon))
    final = min(raw, cap)
    if not all(map(math.isfinite, (raw, cap, final))) or min(raw, cap, final) < 0:
        raise CABGContractError("Computed CABG budget is invalid.")
    return {
        "corrected_lm_rms": corrected_lm,
        "corrected_auxiliary_rms": corrected_auxiliary,
        "lm_shared_norm": lm_norm,
        "auxiliary_shared_norm": auxiliary_norm,
        "lambda_raw": raw,
        "lambda_cap": cap,
        "lambda_final": final,
    }


def validate_auxiliary_support(
    named_gradients: Mapping[str, torch.Tensor | None],
    shared_names: Sequence[str],
) -> None:
    shared = set(shared_names)
    if not shared or not shared.issubset(named_gradients):
        raise CABGContractError("Shared-support gradient names are missing.")
    for name, gradient in named_gradients.items():
        if name in shared:
            if gradient is None:
                raise CABGContractError(f"Shared auxiliary gradient is disconnected: {name}")
            value = _finite_gradient_tensor(gradient, context=name)
            if not torch.any(value != 0):
                raise CABGContractError(f"Shared auxiliary gradient is zero: {name}")
        elif gradient is not None:
            value = _finite_gradient_tensor(gradient, context=name)
            if torch.any(value != 0):
                raise CABGContractError(f"Auxiliary gradient leaked outside shared support: {name}")


def combine_and_clip_gradients(
    lm_gradients: Mapping[str, torch.Tensor],
    auxiliary_gradients: Mapping[str, torch.Tensor],
    *,
    shared_names: Sequence[str],
    lambda_value: float,
    rho_max: float = RHO_MAX,
    max_norm: float = GLOBAL_CLIP_NORM,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if set(auxiliary_gradients) != set(shared_names):
        raise CABGContractError("Auxiliary aggregate does not match the exact shared support.")
    if not set(shared_names).issubset(lm_gradients):
        raise CABGContractError("LM aggregate is missing shared-support gradients.")
    if not math.isfinite(lambda_value) or lambda_value < 0:
        raise CABGContractError("Lambda must be finite and non-negative.")
    lm_shared = {name: lm_gradients[name] for name in shared_names}
    scaled_auxiliary = {name: auxiliary_gradients[name].float() * lambda_value for name in shared_names}
    lm_shared_norm = float(gradient_norm(lm_shared))
    scaled_auxiliary_norm = float(gradient_norm(scaled_auxiliary))
    tolerance = max(1e-8, 1e-5 * lm_shared_norm)
    if scaled_auxiliary_norm > rho_max * lm_shared_norm + tolerance:
        raise CABGContractError("Preclip shared-support trust cap was violated.")
    combined = {
        name: _finite_gradient_tensor(value, context=name).clone()
        for name, value in lm_gradients.items()
    }
    for name in shared_names:
        combined[name].add_(scaled_auxiliary[name].to(combined[name]))
    preclip_norm = float(gradient_norm(combined))
    clip_scale = min(1.0, max_norm / (preclip_norm + 1e-12))
    clipped = {name: value * clip_scale for name, value in combined.items()}
    postclip_norm = float(gradient_norm(clipped))
    return clipped, {
        "lm_shared_norm": lm_shared_norm,
        "scaled_auxiliary_shared_norm": scaled_auxiliary_norm,
        "trust_cap_tolerance": tolerance,
        "preclip_full_norm": preclip_norm,
        "clip_scale": clip_scale,
        "postclip_full_norm": postclip_norm,
        "cap_checked_before_clip": True,
    }
