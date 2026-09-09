"""Pure, auditable CABG-LCE-MIL v1.2 controller and gradient composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from reproduction.stage2i.cabg_math import (
    CABGContractError,
    aggregate_class_balanced_gradients,
    aggregate_lm_gradients,
    combine_and_clip_gradients,
    per_image_class_balanced_rms,
    validate_auxiliary_support,
    validate_block_labels,
)
from reproduction.stage2l.constants import (
    EMA_BETA,
    EPSILON,
    GLOBAL_CLIP_NORM,
    LM_REFERENCE_FLOOR,
    RHO,
    RHO_MAX,
)


def _norm(values: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name in sorted(values):
        value = values[name].detach().double().cpu()
        if not torch.isfinite(value).all():
            raise CABGContractError(f"Non-finite gradient: {name}")
        total += value.square().sum()
    return float(total.sqrt())


@dataclass
class DynamicController:
    beta: float = EMA_BETA
    lm_ema: float = 0.0
    lce_ema: float = 0.0
    valid_blocks: int = 0

    def state_dict(self) -> dict[str, float | int]:
        return {
            "beta": self.beta,
            "lm_ema": self.lm_ema,
            "lce_ema": self.lce_ema,
            "valid_blocks": self.valid_blocks,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"beta", "lm_ema", "lce_ema", "valid_blocks"}:
            raise CABGContractError("D4 controller state schema mismatch.")
        if float(state["beta"]) != self.beta or int(state["valid_blocks"]) < 0:
            raise CABGContractError("D4 controller state identity mismatch.")
        values = torch.tensor([float(state["lm_ema"]), float(state["lce_ema"])])
        if not torch.isfinite(values).all() or torch.any(values < 0):
            raise CABGContractError("D4 controller state is invalid.")
        self.lm_ema = float(state["lm_ema"])
        self.lce_ema = float(state["lce_ema"])
        self.valid_blocks = int(state["valid_blocks"])

    def update(
        self,
        *,
        lm_rms: float,
        lce_rms: float,
        aggregate_lm_s9: Mapping[str, torch.Tensor],
        aggregate_lce_s9: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        values = torch.tensor([lm_rms, lce_rms], dtype=torch.float64)
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise CABGContractError("D4 class-balanced RMS must be finite and positive.")
        lm_norm, lce_norm = _norm(aggregate_lm_s9), _norm(aggregate_lce_s9)
        if lm_norm <= LM_REFERENCE_FLOOR:
            raise CABGContractError("D4 aggregate LM S9 support is zero or too small.")
        self.lm_ema = self.beta * self.lm_ema + (1 - self.beta) * float(lm_rms)
        self.lce_ema = self.beta * self.lce_ema + (1 - self.beta) * float(lce_rms)
        self.valid_blocks += 1
        correction = 1.0 - self.beta**self.valid_blocks
        corrected_lm = self.lm_ema / correction
        corrected_lce = self.lce_ema / correction
        lambda_raw = RHO * corrected_lm / (corrected_lce + EPSILON)
        lambda_cap = RHO_MAX * lm_norm / (lce_norm + EPSILON)
        lambda_final = min(lambda_raw, lambda_cap)
        result = {
            "lm_ema": self.lm_ema,
            "lce_ema": self.lce_ema,
            "corrected_lm_rms": corrected_lm,
            "corrected_lce_rms": corrected_lce,
            "lm_s9_norm": lm_norm,
            "lce_s9_norm": lce_norm,
            "lambda_raw": lambda_raw,
            "lambda_cap": lambda_cap,
            "lambda_final": lambda_final,
            "valid_blocks": self.valid_blocks,
        }
        if not torch.isfinite(torch.tensor(list(result.values()), dtype=torch.float64)).all():
            raise CABGContractError("D4 controller produced a non-finite value.")
        return result


def compute_block_gradients(
    per_image_lm: Sequence[Mapping[str, torch.Tensor]],
    per_image_lce: Sequence[Mapping[str, torch.Tensor]],
    labels: Sequence[object],
    *,
    trainable_names: Sequence[str],
    s9_names: Sequence[str],
    controller: DynamicController,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    normalized = validate_block_labels(labels)
    if len(per_image_lm) != len(normalized) or len(per_image_lce) != len(normalized):
        raise CABGContractError("D4 logical-block gradient/sample counts differ.")
    if any(set(row) != set(trainable_names) for row in per_image_lm):
        raise CABGContractError("D4 LM gradient support differs from T21.")
    if any(set(row) != set(s9_names) for row in per_image_lce):
        raise CABGContractError("D4 LCE gradient support differs from exact S9.")
    lm_s9 = [{name: row[name] for name in s9_names} for row in per_image_lm]
    lm_rms, lm_class = per_image_class_balanced_rms(lm_s9, normalized, epsilon=EPSILON)
    lce_rms, lce_class = per_image_class_balanced_rms(per_image_lce, normalized, epsilon=EPSILON)
    aggregate_lm = aggregate_lm_gradients(per_image_lm)
    aggregate_lce = aggregate_class_balanced_gradients(per_image_lce, normalized)
    aggregate_lm_s9 = {name: aggregate_lm[name] for name in s9_names}
    budget = controller.update(
        lm_rms=float(lm_rms),
        lce_rms=float(lce_rms),
        aggregate_lm_s9=aggregate_lm_s9,
        aggregate_lce_s9=aggregate_lce,
    )
    applied, diagnostics = combine_and_clip_gradients(
        aggregate_lm,
        aggregate_lce,
        shared_names=s9_names,
        lambda_value=budget["lambda_final"],
        rho_max=RHO_MAX,
        max_norm=GLOBAL_CLIP_NORM,
    )
    return applied, {
        "labels": list(normalized),
        "lm_rms": float(lm_rms),
        "lce_rms": float(lce_rms),
        "lm_class_rms": {key: float(value) for key, value in lm_class.items()},
        "lce_class_rms": {key: float(value) for key, value in lce_class.items()},
        "aggregate_lm": aggregate_lm,
        "aggregate_lce": aggregate_lce,
        "budget": budget,
        "gradient": diagnostics,
    }


def exact_s9_from_full(
    named_gradients: Mapping[str, torch.Tensor | None], s9_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    validate_auxiliary_support(named_gradients, s9_names)
    return {name: named_gradients[name].detach().float().cpu() for name in s9_names}  # type: ignore[union-attr]

