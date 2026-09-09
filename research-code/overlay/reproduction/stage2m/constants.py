"""Locked D4-Scout design constants."""

from __future__ import annotations

from reproduction.stage2l.constants import (
    ADAM_BETAS,
    ADAM_EPSILON,
    EMA_BETA,
    EPSILON,
    GLOBAL_CLIP_NORM,
    LEARNING_RATE,
    LM_REFERENCE_FLOOR,
    RHO,
    RHO_MAX,
    S9_FINGERPRINT,
    S9_NAMES,
    STRUCTURAL_ZERO_NAME,
    TRAINABLE_NAMES,
    TRAINABLE_SCHEMA,
    WEIGHT_DECAY,
)

SCOUT_BLOCKS = 24
PHASE1_BLOCKS = 12
NORMAL_PER_BLOCK = 1
ABNORMAL_PER_BLOCK = 2
EVAL_NORMAL = 12
EVAL_ABNORMAL = 20
SEED = 42
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_REPLICATES = 2000
SELECTION_SALT = "cabg-lce-mil-v1.2-d4-scout-v1"
EXTERNAL_PEAK_LIMIT_MIB = 22500

EXPECTED_TRAINING_DEVELOPMENT_SHA256 = "c27ea1381d81949c1565412b81df6472be6f08061d11aa140191bbd28dd2810f"
EXPECTED_D3R_SHA256 = "7e581453de5f4abba2eafa87b982eccb751264f66e86ac10f3b92a160ae20db1"
EXPECTED_D4_PILOT_SHA256 = "b3edf80271a37436c8c43583053a2be6d2d42019a43d5e4d50ba767e353ba233"
EXPECTED_POST_EXCLUSION_COUNTS = {"abnormal": 68, "normal": 28}
EXPECTED_TRAIN_UNIQUE_COUNTS = {"abnormal": 48, "normal": 16}
EXPECTED_TRAIN_EXPOSURE_COUNTS = {"abnormal": 48, "normal": 24}
EXPECTED_EVAL_COUNTS = {"abnormal": 20, "normal": 12}

SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "reproduction/stage2h/runtime.py",
    "reproduction/stage2h/state_audit.py",
    "reproduction/stage2i/cabg_math.py",
    "reproduction/stage2i/fingerprint.py",
    "reproduction/stage2i/rng_state.py",
    "reproduction/stage2k/evidence.py",
    "reproduction/stage2k/producer_math.py",
    "reproduction/stage2l/constants.py",
    "reproduction/stage2l/controller.py",
    "reproduction/stage2l/checkpoint.py",
    "reproduction/stage2m/__init__.py",
    "reproduction/stage2m/constants.py",
    "reproduction/stage2m/metrics.py",
    "reproduction/stage2m/training_math.py",
    "reproduction/stage2m/build_split.py",
    "reproduction/stage2m/preflight.py",
    "reproduction/stage2m/train_scout.py",
    "reproduction/stage2m/evaluate_scout.py",
    "reproduction/stage2m/summarize_scout.py",
    "reproduction/stage2m/summarize_midpoint.py",
    "reproduction/stage2m/run_unit_tests.py",
    "reproduction/stage2m/tests/test_scout.py",
    "reproduction/stage2m/run_d4_scout_v1_l4.sh",
    "reproduction/stage2n/__init__.py",
    "reproduction/stage2n/repair.py",
    "reproduction/stage2n/summarize_repair.py",
    "reproduction/stage2n/summarize_adjustment.py",
    "reproduction/stage2n/run_unit_tests.py",
    "reproduction/stage2n/tests/test_repair.py",
    "reproduction/stage2n/run_stability_repair_v1_l4.sh",
    "reproduction/stage2n/run_moment_reset_adjustment_v1_l4.sh",
    "reproduction/stage2o/__init__.py",
    "reproduction/stage2o/causal.py",
    "reproduction/stage2o/audit_parent.py",
    "reproduction/stage2o/summarize_causal.py",
    "reproduction/stage2o/run_unit_tests.py",
    "reproduction/stage2o/tests/__init__.py",
    "reproduction/stage2o/tests/test_causal.py",
    "reproduction/stage2o/run_causal_branch_v1_l4.sh",
    "reproduction/stage2p/__init__.py",
    "reproduction/stage2p/schedule.py",
    "reproduction/stage2p/verify_fixed_cutoff.py",
    "reproduction/stage2p/run_unit_tests.py",
    "reproduction/stage2p/tests/__init__.py",
    "reproduction/stage2p/tests/test_fixed_cutoff.py",
    "reproduction/stage2p/run_fixed_cutoff_v1_l4.sh",
)
