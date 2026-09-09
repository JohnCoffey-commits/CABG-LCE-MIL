"""Only block number controls the cutoff; no held-out or spatial input."""

from pathlib import Path

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2n.repair import apply_repair_multiplier
from reproduction.stage2p import ARMS


def cutoff_multiplier(arm: str, block_id: int) -> float:
    if arm not in ARMS or type(block_id) is not int or not 0 <= block_id < 24:
        raise CABGContractError("Invalid fixed-cutoff arm or block.")
    return 0.0 if arm == "cabg_fixed_cutoff" and block_id >= 12 else 1.0


def apply_fixed_cutoff(arm, block_id, applied, details):
    multiplier = cutoff_multiplier(arm, block_id)
    if multiplier == 0.0:
        applied, details = apply_repair_multiplier(details, multiplier)
    else:
        # Preserve the exact original pre-cutoff numerical path, including clipping.
        details = dict(details)
        details["budget"] = dict(details["budget"])
        details["budget"]["lambda_controller"] = details["budget"]["lambda_final"]
    return applied, details, {
        "block_number": block_id + 1,
        "applied_lce_multiplier": multiplier,
        "controller_diagnostics_computed": True,
        "heldout_input_used": False,
        "optimizer_or_controller_reset": False,
    }


def validate_origin(arm: str, phase: str, run_root: Path, resume: Path | None, parent: Path | None):
    if arm not in ARMS or phase not in ("phase1", "phase2") or parent is not None:
        raise CABGContractError("Fixed-cutoff cannot use a historical/shared parent.")
    own = (run_root / arm / "checkpoints" / "mid-block12.pt").resolve()
    if phase == "phase1" and resume is not None:
        raise CABGContractError("Fixed-cutoff phase1 must start from initialization.")
    if phase == "phase2" and (resume is None or resume.resolve() != own):
        raise CABGContractError("Fixed-cutoff phase2 must resume its own new midpoint.")
