"""Development-only D4-Scout stability repair."""

REPAIR_RUN_ID = "cabg-lce-mil-v1.2-d4-scout-stability-repair-v1"
REPAIR_SCHEMA_VERSION = "cabg-lce-mil-v1.2-d4-scout-stability-repair-1"
REPAIR_ARMS = ("cabg_cosine_taper", "cabg_spatial_guard")
MOMENT_RESET_ARM = "cabg_spatial_guard_moment_reset"
SUPPORTED_REPAIR_ARMS = (*REPAIR_ARMS, MOMENT_RESET_ARM)
ADJUSTMENT_RUN_ID = "cabg-lce-mil-v1.2-d4-scout-stability-repair-moment-reset-v1"
ADJUSTMENT_SCHEMA_VERSION = "cabg-lce-mil-v1.2-d4-scout-stability-repair-moment-reset-1"
