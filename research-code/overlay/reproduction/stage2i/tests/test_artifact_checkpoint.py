import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from reproduction.stage2i.artifact_schema import (
    GENESIS_HASH,
    append_block_record,
    finalize_block_record,
    validate_block_record,
)
from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.checkpoint import (
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
from reproduction.stage2i.constants import BLOCK_TRACE_SCHEMA_VERSION
from reproduction.stage2i.rng_state import seed_all
from reproduction.stage2i.toy_engine import TOY_PROVENANCE, ToyCABGEngine


def valid_record(previous_hash=GENESIS_HASH, block_id=0):
    record = {
        "schema_version": BLOCK_TRACE_SCHEMA_VERSION,
        "status": "SUCCESS",
        "run_id": "toy-cabg",
        "method": "CABG-MIL-v1.1",
        "seed": 42,
        "epoch": 0,
        "block_id": block_id,
        "ordered_samples": [
            {"sample_id": "n", "sha256": "1" * 64, "label": "normal"},
            {"sample_id": "a1", "sha256": "2" * 64, "label": "abnormal"},
            {"sample_id": "a2", "sha256": "3" * 64, "label": "abnormal"},
        ],
        "class_counts": {"normal": 1, "abnormal": 2},
        "losses": {
            "lm_per_image": [1.0, 2.0, 3.0],
            "auxiliary_per_image": [0.5, 0.6, 0.7],
            "lm_mean": 2.0,
            "auxiliary_class_balanced": 0.575,
        },
        "per_image_shared_norms": [
            {"sample_id": "n", "label": "normal", "lm": 1.0, "auxiliary": 2.0},
            {"sample_id": "a1", "label": "abnormal", "lm": 1.5, "auxiliary": 2.5},
            {"sample_id": "a2", "label": "abnormal", "lm": 2.0, "auxiliary": 3.0},
        ],
        "ema": {
            "raw_lm": 0.1,
            "raw_auxiliary": 0.2,
            "corrected_lm": 1.0,
            "corrected_auxiliary": 2.0,
            "valid_blocks": block_id + 1,
        },
        "budget": {
            "lambda_raw": 0.05,
            "lambda_cap": 0.04,
            "lambda_final": 0.04,
            "rho": 0.1,
            "rho_max": 0.2,
        },
        "aggregate_shared": {
            "lm_norm": 1.0,
            "auxiliary_norm": 5.0,
            "scaled_auxiliary_norm": 0.2,
            "cosine": 0.0,
            "cap_utilization": 1.0,
        },
        "gradient_norms": {
            "full_preclip": 2.0,
            "full_postclip": 1.0,
            "clip_scale": 0.5,
            "cap_checked_before_clip": True,
        },
        "optimizer": {"step": block_id + 1, "type": "AdamW"},
        "scheduler": {"step": block_id + 1, "learning_rate": 1e-4},
        "evidence": {
            "score_min": -0.2,
            "score_max": 0.3,
            "score_mean": 0.1,
            "auxiliary_loss_min": 0.5,
            "auxiliary_loss_max": 0.7,
            "auxiliary_loss_mean": 0.6,
            "evidence_min": -0.8,
            "evidence_max": 0.9,
            "evidence_mean": 0.05,
            "evidence_saturation_fraction": 0.0,
            "abnormal_map_min": -2.0,
            "abnormal_map_max": 2.0,
            "abnormal_map_mean": 0.1,
            "abnormal_map_saturation_fraction": 0.0,
            "normal_map_min": -2.0,
            "normal_map_max": 2.0,
            "normal_map_mean": -0.1,
            "normal_map_saturation_fraction": 0.0,
            "spatial_entropy_mean": 5.0,
            "effective_support_mean": 100.0,
            "top11_mass_mean": 0.2,
            "max_pooling_weight": 0.05,
            "min_nonzero_pooling_weight": 1e-6,
        },
        "memory": {
            "cuda_allocated_mib": 0.0,
            "cuda_reserved_mib": 0.0,
            "external_peak_mib": 0.0,
        },
        "wall_time_seconds": 0.1,
        "rng": {"before_block": "4" * 64, "after_block": "5" * 64},
        "provenance": {
            "source_commit": "0" * 40,
            "implementation_fingerprint": "6" * 64,
            "dataset_manifest_sha256": "7" * 64,
            "support_fingerprint": "8" * 64,
        },
        "checkpoint_sha256": "9" * 64,
        "previous_record_sha256": previous_hash,
    }
    return finalize_block_record(record)


def assert_nested_equal(testcase, first, second, path="root"):
    testcase.assertEqual(type(first), type(second), path)
    if isinstance(first, torch.Tensor):
        testcase.assertTrue(torch.equal(first, second), path)
    elif isinstance(first, dict):
        testcase.assertEqual(set(first), set(second), path)
        for key in first:
            assert_nested_equal(testcase, first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        testcase.assertEqual(len(first), len(second), path)
        for index, (left, right) in enumerate(zip(first, second)):
            assert_nested_equal(testcase, left, right, f"{path}[{index}]")
    else:
        testcase.assertEqual(first, second, path)


class ArtifactSchemaTests(unittest.TestCase):
    def test_strict_schema_and_append_only_hash_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            first = valid_record()
            append_block_record(path, first)
            second = valid_record(first["record_sha256"], block_id=1)
            append_block_record(path, second)
            rows = [json.loads(value) for value in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            validate_block_record(rows[0])
            validate_block_record(rows[1])
            self.assertEqual(rows[1]["previous_record_sha256"], rows[0]["record_sha256"])

    def test_schema_missing_extra_nonfinite_and_tampering_fail_closed(self):
        record = valid_record()
        missing = dict(record)
        missing.pop("memory")
        with self.assertRaises(CABGContractError):
            validate_block_record(missing)
        extra = dict(record)
        extra["unexpected"] = True
        with self.assertRaises(CABGContractError):
            validate_block_record(extra)
        nonfinite = copy.deepcopy(record)
        nonfinite["losses"]["lm_mean"] = float("nan")
        with self.assertRaises(CABGContractError):
            validate_block_record(nonfinite)
        tampered = copy.deepcopy(record)
        tampered["budget"]["lambda_final"] = 0.03
        with self.assertRaises(CABGContractError):
            validate_block_record(tampered)

    def test_wrong_previous_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            append_block_record(path, valid_record())
            with self.assertRaises(CABGContractError):
                append_block_record(path, valid_record("f" * 64, block_id=1))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_schema_rejects_invalid_cabg_and_sampler_state(self):
        seed_all(123)
        engine = ToyCABGEngine()
        engine.run_block(0)
        invalid_budget = engine.checkpoint_payload()
        invalid_budget["cabg_state"]["last_lambda_final"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CABGContractError):
                save_checkpoint_atomic(Path(directory) / "invalid.pt", invalid_budget)
        invalid_sampler = engine.checkpoint_payload()
        invalid_sampler["sampler_state"]["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CABGContractError):
                save_checkpoint_atomic(Path(directory) / "invalid.pt", invalid_sampler)

    def test_strict_checkpoint_detects_corruption_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            seed_all(123)
            engine = ToyCABGEngine()
            engine.run_block(0)
            save_checkpoint_atomic(path, engine.checkpoint_payload())
            payload = load_checkpoint_strict(path, expected_provenance=TOY_PROVENANCE)
            self.assertEqual(payload["trace_state"]["completed_blocks"], 1)
            with self.assertRaises(FileExistsError):
                save_checkpoint_atomic(path, engine.checkpoint_payload())
            with path.open("r+b") as stream:
                stream.seek(0)
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 0x01]))
            with self.assertRaises(CABGContractError):
                load_checkpoint_strict(path, expected_provenance=TOY_PROVENANCE)

    def test_fresh_process_two_block_resume_is_bitwise_equal(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "block-000001.pt"
            resumed_output = root / "resumed-final.pt"

            seed_all(2026)
            uninterrupted = ToyCABGEngine()
            uninterrupted.run_block(0)
            uninterrupted.run_block(1)
            expected = uninterrupted.result_state()

            seed_all(2026)
            interrupted = ToyCABGEngine()
            interrupted.run_block(0)
            save_checkpoint_atomic(checkpoint, interrupted.checkpoint_payload())
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "reproduction.stage2i.toy_resume_worker",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(resumed_output),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            observed = torch.load(resumed_output, map_location="cpu", weights_only=False)
            assert_nested_equal(self, expected, observed)


if __name__ == "__main__":
    unittest.main()
