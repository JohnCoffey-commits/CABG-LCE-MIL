"""CABG-MIL v1.1 Gate D2 contracts.

This package contains CPU-testable mathematics, deterministic data/RNG
utilities, strict artifacts, and checkpoint primitives.  Importing it must not
load the MEDIC-AD model or initialize CUDA.
"""

from .constants import PROTOCOL_VERSION, SHARED_SUPPORT_NAMES

__all__ = ["PROTOCOL_VERSION", "SHARED_SUPPORT_NAMES"]
