"""Shared-checkpoint loading and the two registered causal interventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, tensor_sha256
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2l.checkpoint import tensor_inventory, validate
from reproduction.stage2l.constants import S9_NAMES, TRAINABLE_NAMES
from reproduction.stage2n.repair import apply_repair_multiplier
from reproduction.stage2o import (
    CAUSAL_ARMS,
    CONTROL_ARM,
    LCE_OFF_ARM,
    PARENT_CHECKPOINT_BYTES,
    PARENT_CHECKPOINT_SHA256,
    PARENT_EVAL_MANIFEST_SHA256,
    PARENT_IMPLEMENTATION_FINGERPRINT,
    PARENT_MANIFEST_SHA256,
    PARENT_ORDER_SHA256,
    PARENT_RNG_FINGERPRINT,
    PARENT_SOURCE_COMMIT,
    PARENT_TENSOR_INVENTORY_SHA256,
    PARENT_TRACE_ANCHOR,
    PARENT_TRAIN_MANIFEST_SHA256,
    RESET_ARM,
)


PARENT_PROVENANCE = {
    "source_commit": PARENT_SOURCE_COMMIT,
    "implementation_fingerprint": PARENT_IMPLEMENTATION_FINGERPRINT,
    "base_checkpoint_fingerprint": "f414ff64ce8db0cfc17d82c83c9fcbb0eda22cd68592b83eb635aef202cb3186",
    "train_manifest_sha256": PARENT_TRAIN_MANIFEST_SHA256,
    "eval_manifest_sha256": PARENT_EVAL_MANIFEST_SHA256,
    "matched_configuration_sha256": "eabe158a08e11078d7e5e29f622f28549c4da5de5fce5d7b2590898dfea3f4bd",
    "arm": "cabg_lce",
    "applied_objective": "lm_plus_dynamic_cabg_lce",
}


def load_locked_parent(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.name != "mid-block12.pt" or not manifest_path.is_file():
        raise CABGContractError("Causal continuation requires the locked original block-12 checkpoint and manifest.")
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(manifest_path) != PARENT_MANIFEST_SHA256:
        raise CABGContractError("Causal parent manifest SHA-256 changed.")
    if path.stat().st_size != PARENT_CHECKPOINT_BYTES or sha256_file(path) != PARENT_CHECKPOINT_SHA256:
        raise CABGContractError("Causal parent checkpoint identity changed.")
    if (
        manifest.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or manifest.get("checkpoint_bytes") != PARENT_CHECKPOINT_BYTES
        or manifest.get("tensor_inventory_sha256") != PARENT_TENSOR_INVENTORY_SHA256
        or manifest.get("rng_fingerprint") != PARENT_RNG_FINGERPRINT
    ):
        raise CABGContractError("Causal parent checkpoint manifest contract changed.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate(payload)
    inventory = tensor_inventory(payload)
    if inventory != manifest.get("tensor_inventory") or canonical_json_sha256(inventory) != PARENT_TENSOR_INVENTORY_SHA256:
        raise CABGContractError("Causal parent tensor inventory changed.")
    if payload["sampler_state"] != {
        "cursor": 12,
        "manifest_sha256": PARENT_TRAIN_MANIFEST_SHA256,
        "order_sha256": PARENT_ORDER_SHA256,
    }:
        raise CABGContractError("Causal parent sampler state changed.")
    if payload["trace_state"] != {"completed_blocks": 12, "last_record_sha256": PARENT_TRACE_ANCHOR}:
        raise CABGContractError("Causal parent trace anchor changed.")
    if dict(payload["provenance"]) != PARENT_PROVENANCE:
        raise CABGContractError("Causal parent provenance changed.")
    if rng_state_fingerprint(payload["rng_state"]) != PARENT_RNG_FINGERPRINT:
        raise CABGContractError("Causal parent RNG state changed.")
    return dict(payload), manifest


def _optimizer_tensor_hashes(
    optimizer: torch.optim.Optimizer,
    masters: Mapping[str, torch.nn.Parameter],
    names: Sequence[str] = TRAINABLE_NAMES,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in names:
        state = optimizer.state.get(masters[name])
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise CABGContractError(f"Causal optimizer state schema changed: {name}")
        member_hashes: dict[str, str] = {}
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = state[key]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise CABGContractError(f"Causal optimizer state is invalid: {name}.{key}")
            member_hashes[key] = tensor_sha256(value)
        result[name] = member_hashes
    return result


def apply_optimizer_intervention(
    branch: str,
    optimizer: torch.optim.Optimizer,
    masters: Mapping[str, torch.nn.Parameter],
    *,
    names: Sequence[str] = TRAINABLE_NAMES,
    s9_names: Sequence[str] = S9_NAMES,
) -> dict[str, object]:
    if branch not in CAUSAL_ARMS:
        raise CABGContractError(f"Unknown causal branch: {branch}")
    before = _optimizer_tensor_hashes(optimizer, masters, names)
    pre_nonzero: list[str] = []
    if branch == RESET_ARM:
        for name in s9_names:
            first = optimizer.state[masters[name]]["exp_avg"]
            if torch.any(first != 0):
                pre_nonzero.append(name)
            first.zero_()
    after = _optimizer_tensor_hashes(optimizer, masters, names)
    changed = [f"{name}.{key}" for name in names for key in ("step", "exp_avg", "exp_avg_sq") if before[name][key] != after[name][key]]
    # ``changed`` follows the canonical T21 order above.  S9_NAMES is a
    # semantic grouping and deliberately has a different order, so derive the
    # expectation in the same canonical order rather than comparing two
    # unrelated orderings.
    s9_name_set = set(s9_names)
    expected_changed = [f"{name}.exp_avg" for name in names if name in s9_name_set] if branch == RESET_ARM else []
    if changed != expected_changed:
        raise CABGContractError("Causal optimizer intervention changed an unauthorized state tensor.")
    post_zero = [name for name in s9_names if not torch.any(optimizer.state[masters[name]]["exp_avg"] != 0)]
    if branch == RESET_ARM and (pre_nonzero != list(s9_names) or post_zero != list(s9_names)):
        raise CABGContractError("Causal S9 first-moment reset was incomplete or had a zero pre-state.")
    if branch != RESET_ARM and post_zero:
        raise CABGContractError("State-kept causal branch unexpectedly contains a zero S9 first moment.")
    return {
        "branch": branch,
        "applied": branch == RESET_ARM,
        "timing": "after_parent_restore_before_block13" if branch == RESET_ARM else "none",
        "changed_state_tensors": changed,
        "changed_state_tensor_count": len(changed),
        "s9_exp_avg_pre_nonzero_names": pre_nonzero,
        "s9_exp_avg_post_zero_names": post_zero,
        "exp_avg_sq_preserved": all(before[name]["exp_avg_sq"] == after[name]["exp_avg_sq"] for name in names),
        "step_preserved": all(before[name]["step"] == after[name]["step"] for name in names),
        "non_s9_state_preserved": all(
            before[name] == after[name] for name in names if name not in s9_name_set
        ),
        "before_inventory_sha256": canonical_json_sha256(before),
        "after_inventory_sha256": canonical_json_sha256(after),
    }


def apply_causal_gradient(branch: str, details: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any], float]:
    if branch not in CAUSAL_ARMS:
        raise CABGContractError(f"Unknown causal branch: {branch}")
    multiplier = 1.0 if branch == CONTROL_ARM else 0.0
    applied, updated = apply_repair_multiplier(details, multiplier)
    return applied, updated, multiplier


def validate_parent_audit(audit: Mapping[str, Any]) -> None:
    required = {
        "status": "SUCCESS",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
        "cursor": 12,
        "trace_anchor": PARENT_TRACE_ANCHOR,
        "rng_fingerprint": PARENT_RNG_FINGERPRINT,
        "train_manifest_sha256": PARENT_TRAIN_MANIFEST_SHA256,
        "eval_manifest_sha256": PARENT_EVAL_MANIFEST_SHA256,
        "continuation_blocks": 12,
        "continuation_exposures": 36,
    }
    if any(audit.get(key) != value for key, value in required.items()):
        raise CABGContractError("Serialized causal parent audit changed.")
    if audit.get("all_registered_overlaps_zero") is not True or audit.get("protected_internal_test_image_files_opened") != 0 or audit.get("protected_internal_test_outputs_read") != 0:
        raise CABGContractError("Serialized causal parent data boundary failed.")
