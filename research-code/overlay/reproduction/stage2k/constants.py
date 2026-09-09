"""Locked D3R identities, schemas, gates, and source inputs."""

from __future__ import annotations

import math


SELECTION_SALT = "cabg-lce-mil-v1.2-d3r-v1"
EXPECTED_PARENT_COMMIT = "ce5d62d421f070770a388b53af151ef251462148"
EXPECTED_FORMAL_R0_COMMIT = "5e49c66150c487f67f4bca1bb5e4cfdcb6615612"
EXPECTED_APPROVED_R0_TREE = "fbe20a2e81304c54423d2ee9e16c35848c3a5b14"
EXPECTED_GATE_D3_COMMIT = "f8bd0130bb9a2ba0324e73148a4ecd0c331ec361"
EXPECTED_TRAINING_DEVELOPMENT_SHA256 = "c27ea1381d81949c1565412b81df6472be6f08061d11aa140191bbd28dd2810f"
EXPECTED_THRESHOLD_VALIDATION_SHA256 = "28ea92d7fe0f957ab4c34d5038bd1cc79db5805bd3ad923c0d1383c5c2e942d5"
EXPECTED_OLD_D3_SHA256 = "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c"
EXPECTED_STAGE2H_EXCLUDED_SHA256 = "98943d6f1f6e333696ad0f38fd3af6d20e82c85239f389a285f9d222de5aeaee"
EXPECTED_STAGE2H_TRAIN_SHA256 = "580af0a78aec247004a6367c831594ad6c4325d0a3ed2ad9be8cce8c37d5ed3c"
EXPECTED_STAGE2H_PREVIOUS_CALIBRATION_SHA256 = "c6ff4dba0e53f8a3a185360c395d6b8f262476bdaf97229a85f96a576f922f7a"
EXPECTED_STAGE2H_CALIBRATION_SHA256 = "69988c768928faf0e1120b453963148d3fad111e7ba51c141a597fa05fce4b5d"
EXPECTED_STAGE2H_INTERNAL_TEST_SHA256 = "4d8c419d0a0571fe2d9ba7462beb65fa0fccd31260f9b6f5d2f62be278b01c03"

NORMAL_LABEL = "normal"
ABNORMAL_LABEL = "abnormal"
POSITIONS = 1024
TAU = 1.0 / math.log(POSITIONS)
GRADIENT_FLOOR = 1e-12
FORMULA_ATOL = 2e-6
FORMULA_RTOL = 2e-6
EXTERNAL_PEAK_LIMIT_MIB = 22500
MEAN_IMAGE_LIMIT_SECONDS = 5.0
MAX_IMAGE_LIMIT_SECONDS = 10.0
POST_LOAD_LIMIT_SECONDS = 900.0

TRAINABLE_SCHEMA = (
    ("model.visual.anomaly_qformer.abnormal_prompt", (1, 1, 1280), 1280),
    ("model.visual.anomaly_qformer.anomaly_attention.key_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.anomaly_attention.key_proj.weight", (512, 1280), 655360),
    ("model.visual.anomaly_qformer.anomaly_attention.query_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.anomaly_attention.query_proj.weight", (512, 1280), 655360),
    ("model.visual.anomaly_qformer.gate_scale", (), 1),
    ("model.visual.anomaly_qformer.normal_prompt", (1, 1, 1280), 1280),
    ("model.visual.anomaly_qformer.post_ffn.0.bias", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.0.weight", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.1.bias", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.1.weight", (3584, 3584), 12845056),
    ("model.visual.anomaly_qformer.post_ffn.3.bias", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.3.weight", (3584, 3584), 12845056),
    ("model.visual.anomaly_qformer.q_former.key_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.q_former.key_proj.weight", (512, 3584), 1835008),
    ("model.visual.anomaly_qformer.q_former.query_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.q_former.query_proj.weight", (512, 1280), 655360),
    ("model.visual.deep_prompt_embeddings.0", (10, 1280), 12800),
    ("model.visual.deep_prompt_embeddings.1", (10, 1280), 12800),
    ("model.visual.deep_prompt_embeddings.2", (10, 1280), 12800),
    ("model.visual.deep_prompt_embeddings.3", (10, 1280), 12800),
)
TRAINABLE_NAMES = tuple(row[0] for row in TRAINABLE_SCHEMA)
EXPECTED_TRAINABLE_TENSORS = 21
EXPECTED_TRAINABLE_ELEMENTS = 29561345

S9_NAMES = (
    "model.visual.anomaly_qformer.abnormal_prompt",
    "model.visual.anomaly_qformer.normal_prompt",
    "model.visual.anomaly_qformer.anomaly_attention.query_proj.weight",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.weight",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.bias",
    "model.visual.deep_prompt_embeddings.0",
    "model.visual.deep_prompt_embeddings.1",
    "model.visual.deep_prompt_embeddings.2",
    "model.visual.deep_prompt_embeddings.3",
)
STRUCTURAL_ZERO_NAME = "model.visual.anomaly_qformer.anomaly_attention.query_proj.bias"

SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "reproduction/stage2h/runtime.py",
    "reproduction/stage2h/state_audit.py",
    "reproduction/stage2i/fingerprint.py",
    "reproduction/stage2i/gate_d3_preflight.py",
    "reproduction/stage2i/rng_state.py",
    "reproduction/stage2i/run_gate_d3.py",
    "reproduction/stage2k/__init__.py",
    "reproduction/stage2k/constants.py",
    "reproduction/stage2k/manifest.py",
    "reproduction/stage2k/verify_manifest.py",
    "reproduction/stage2k/build_dataset.py",
    "reproduction/stage2k/producer_math.py",
    "reproduction/stage2k/evidence.py",
    "reproduction/stage2k/run_d3r.py",
    "reproduction/stage2k/verify_d3r.py",
    "reproduction/stage2k/run_unit_tests.py",
    "reproduction/stage2k/run_d3r_v1_l4.sh",
    "reproduction/stage2k/tests/test_manifest_and_dataset.py",
    "reproduction/stage2k/tests/test_math_and_evidence.py",
    "reproduction/stage2k/tests/test_static_contracts.py",
    "reproduction/stage2k/packet-gate.json",
)
