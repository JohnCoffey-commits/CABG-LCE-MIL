"""Locked D4 pilot constants and exact tensor identities."""

from __future__ import annotations

import math

from reproduction.stage2k.constants import S9_NAMES, TRAINABLE_NAMES, TRAINABLE_SCHEMA

POSITIONS = 1024
TAU = 1.0 / math.log(POSITIONS)
EMA_BETA = 0.9
RHO = 0.10
RHO_MAX = 0.20
EPSILON = 1e-8
LM_REFERENCE_FLOOR = 1e-12
GLOBAL_CLIP_NORM = 1.0
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.0
PILOT_BLOCKS = 4
PHASE1_BLOCKS = 2
NORMAL_PER_BLOCK = 1
ABNORMAL_PER_BLOCK = 2
SEED = 42
SELECTION_SALT = "cabg-lce-mil-v1.2-d4-pilot-v1"

EXPECTED_TRAINING_DEVELOPMENT_SHA256 = "c27ea1381d81949c1565412b81df6472be6f08061d11aa140191bbd28dd2810f"
EXPECTED_D3R_SHA256 = "7e581453de5f4abba2eafa87b982eccb751264f66e86ac10f3b92a160ae20db1"
EXPECTED_ELIGIBLE_COUNTS = {"abnormal": 76, "normal": 32}
EXPECTED_PILOT_COUNTS = {"abnormal": 8, "normal": 4}
S9_FINGERPRINT = "fa79bc96523a3d7d3d3e16e714415201516103dffb114950923083719063dc5c"
STRUCTURAL_ZERO_NAME = "model.visual.anomaly_qformer.anomaly_attention.query_proj.bias"
CHECKPOINT_SCHEMA_VERSION = "cabg-lce-mil-v1.2-d4-checkpoint-1"

SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "reproduction/stage2h/runtime.py",
    "reproduction/stage2h/state_audit.py",
    "reproduction/stage2i/fingerprint.py",
    "reproduction/stage2i/rng_state.py",
    "reproduction/stage2k/constants.py",
    "reproduction/stage2k/evidence.py",
    "reproduction/stage2k/producer_math.py",
    "reproduction/stage2l/__init__.py",
    "reproduction/stage2l/constants.py",
    "reproduction/stage2l/controller.py",
    "reproduction/stage2l/build_dataset.py",
    "reproduction/stage2l/checkpoint.py",
    "reproduction/stage2l/run_pilot.py",
    "reproduction/stage2l/verify_pilot.py",
    "reproduction/stage2l/preflight.py",
    "reproduction/stage2l/verify_packet.py",
    "reproduction/stage2l/run_unit_tests.py",
    "reproduction/stage2l/run_d4_pilot_v1_l4.sh",
    "reproduction/stage2l/tests/test_controller.py",
    "reproduction/stage2l/tests/test_checkpoint_static.py",
    "reproduction/stage2l/packet-gate.json",
)
