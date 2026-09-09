"""Small preregistered CABG stability repairs with auditable state."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from reproduction.stage2i.cabg_math import CABGContractError, combine_and_clip_gradients
from reproduction.stage2l.constants import GLOBAL_CLIP_NORM, RHO_MAX
from reproduction.stage2n import MOMENT_RESET_ARM, SUPPORTED_REPAIR_ARMS


SUPPORT_EARLY_WARNING = 256.0
TOP11_EARLY_WARNING = 0.25
GUARD_DAMPED_MULTIPLIER = 0.25
TAPER_START_BLOCK = 12
TAPER_ZERO_BLOCK = 20


def initial_repair_state(arm: str) -> dict[str, object] | None:
    if arm == "cabg_cosine_taper":
        return {"variant": arm}
    if arm in {"cabg_spatial_guard", MOMENT_RESET_ARM}:
        return {"variant": arm, "risk_streak": 0, "latched": False}
    if arm in {"cabg_lce", "lm_only"}:
        return None
    raise CABGContractError(f"Unknown repair arm: {arm}")


def validate_repair_state(arm: str, state: object) -> dict[str, object]:
    if not isinstance(state, Mapping) or state.get("variant") != arm:
        raise CABGContractError("Repair checkpoint state identity mismatch.")
    if arm == "cabg_cosine_taper":
        if set(state) != {"variant"}:
            raise CABGContractError("Cosine-taper state schema mismatch.")
        return dict(state)
    if arm in {"cabg_spatial_guard", MOMENT_RESET_ARM}:
        if set(state) != {"variant", "risk_streak", "latched"}:
            raise CABGContractError("Spatial-guard state schema mismatch.")
        streak = int(state["risk_streak"])
        latched = bool(state["latched"])
        if streak < 0 or (latched and streak < 3):
            raise CABGContractError("Spatial-guard state value mismatch.")
        return {"variant": arm, "risk_streak": streak, "latched": latched}
    raise CABGContractError(f"Unknown repair arm: {arm}")


def cosine_taper_multiplier(block_id: int) -> float:
    if block_id < 0 or block_id >= 24:
        raise CABGContractError("Repair block index outside the registered 24-block horizon.")
    block_number = block_id + 1
    if block_number <= TAPER_START_BLOCK:
        return 1.0
    if block_number >= TAPER_ZERO_BLOCK:
        return 0.0
    progress = (block_number - TAPER_START_BLOCK) / (TAPER_ZERO_BLOCK - TAPER_START_BLOCK)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def repair_decision(
    arm: str,
    block_id: int,
    image_records: Sequence[Mapping[str, object]],
    state: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if arm not in SUPPORTED_REPAIR_ARMS or len(image_records) != 3:
        raise CABGContractError("Repair decision requires one registered arm and one three-image block.")
    validated = validate_repair_state(arm, state)
    minimum_support = min(float(row["effective_support"]) for row in image_records)
    maximum_top11 = max(float(row["top11_mass"]) for row in image_records)
    if not torch.isfinite(torch.tensor([minimum_support, maximum_top11], dtype=torch.float64)).all():
        raise CABGContractError("Repair spatial input is non-finite.")
    risk = minimum_support < SUPPORT_EARLY_WARNING or maximum_top11 > TOP11_EARLY_WARNING
    before = dict(validated)
    if arm == "cabg_cosine_taper":
        multiplier = cosine_taper_multiplier(block_id)
        after = dict(validated)
        reason = "fixed_half_cosine_after_block12"
    else:
        streak = int(validated["risk_streak"])
        latched = bool(validated["latched"])
        if not latched:
            streak = streak + 1 if risk else 0
            latched = streak >= 3
        if latched:
            multiplier = 0.0
            reason = "persistent_warning_latched_off"
        elif streak == 2:
            multiplier = GUARD_DAMPED_MULTIPLIER
            reason = "second_consecutive_warning_damped"
        else:
            multiplier = 1.0
            reason = "no_persistent_warning"
        after = {"variant": arm, "risk_streak": streak, "latched": latched}
    record = {
        "variant": arm,
        "multiplier": multiplier,
        "active": multiplier < 1.0,
        "reason": reason,
        "training_spatial_input": {
            "minimum_effective_support": minimum_support,
            "maximum_top11_mass": maximum_top11,
            "early_warning": risk,
            "support_threshold": SUPPORT_EARLY_WARNING,
            "top11_threshold": TOP11_EARLY_WARNING,
        },
        "state_before": before,
        "state_after": after,
        "heldout_input_used": False,
    }
    return record, after


def apply_repair_multiplier(
    details: Mapping[str, object], multiplier: float
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise CABGContractError("Repair multiplier is outside [0, 1].")
    budget = dict(details["budget"])  # type: ignore[arg-type]
    controller_lambda = float(budget["lambda_final"])
    applied_lambda = controller_lambda * multiplier
    aggregate_lce = details["aggregate_lce"]
    applied, diagnostics = combine_and_clip_gradients(
        details["aggregate_lm"],  # type: ignore[arg-type]
        aggregate_lce,  # type: ignore[arg-type]
        shared_names=tuple(aggregate_lce),  # type: ignore[arg-type]
        lambda_value=applied_lambda,
        rho_max=RHO_MAX,
        max_norm=GLOBAL_CLIP_NORM,
    )
    budget["lambda_controller"] = controller_lambda
    budget["repair_multiplier"] = multiplier
    budget["lambda_final"] = applied_lambda
    updated = dict(details)
    updated["budget"] = budget
    updated["gradient"] = diagnostics
    return applied, updated
