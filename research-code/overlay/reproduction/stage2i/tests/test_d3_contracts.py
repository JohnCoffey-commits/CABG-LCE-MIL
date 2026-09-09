from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.d3_artifact_schema import D3_RECORD_SCHEMA, validate_mechanism_record
from reproduction.stage2i.d3_mechanisms import compute_mechanisms, gradient_cosine
from reproduction.stage2i.d3_observer import AnomalyAttentionObserver


class ToyAttention(torch.nn.Module):
    def forward(self, scores):
        return scores.mean(dim=-1, keepdim=True), scores


class TestD3Mechanisms(unittest.TestCase):
    def test_same_evidence_mechanisms_and_class_directions(self):
        evidence = torch.linspace(-0.8, 0.8, 2 * 4 * 1024, dtype=torch.float64).reshape(2, 4, 1024)
        evidence.requires_grad_(True)
        result = compute_mechanisms(evidence, ["normal", "abnormal"])
        self.assertEqual(tuple(result), ("M0", "M1", "M2"))
        self.assertEqual(result["M0"]["k"], 11)
        self.assertEqual(result["M1"]["k"], 11)
        self.assertTrue(torch.equal(result["M0"]["score"], result["M1"]["score"]))
        for mechanism in result.values():
            self.assertTrue(torch.isfinite(mechanism["score"]).all())
            self.assertTrue(torch.isfinite(mechanism["per_image_loss"]).all())
        gradient = torch.autograd.grad(result["M2"]["per_image_loss"].sum(), evidence)[0]
        self.assertTrue(torch.all(gradient[0] > 0))
        self.assertTrue(torch.all(gradient[1] < 0))

    def test_positions_are_locked(self):
        with self.assertRaisesRegex(CABGContractError, "requires 1024"):
            compute_mechanisms(torch.zeros(1, 4, 16), [0])

    def test_cosine_null_semantics(self):
        value = gradient_cosine({"a": torch.zeros(2)}, {"a": torch.ones(2)})
        self.assertFalse(value["defined"])
        self.assertIsNone(value["value"])


class TestD3Observer(unittest.TestCase):
    def test_exact_two_call_observation(self):
        module = ToyAttention()
        abnormal_scores = torch.randn(4, 1, 1024, dtype=torch.float32)
        normal_scores = torch.randn(4, 1, 1024, dtype=torch.float32)
        evidence = (
            torch.sigmoid(abnormal_scores).reshape(1, 4, 1024)
            - torch.sigmoid(normal_scores).reshape(1, 4, 1024)
        )
        with AnomalyAttentionObserver(module) as observer:
            module(abnormal_scores)
            module(normal_scores)
        result = observer.finalize(evidence)
        self.assertTrue(result["reconstruction_exact"])
        self.assertEqual(result["reconstruction_max_abs"], 0.0)

    def test_call_count_fail_closed(self):
        module = ToyAttention()
        evidence = torch.zeros(1, 4, 1024)
        with AnomalyAttentionObserver(module) as observer:
            module(torch.zeros(4, 1, 1024))
        with self.assertRaisesRegex(CABGContractError, "exactly two"):
            observer.finalize(evidence)


def valid_record():
    grad = {"name": "p", "present": True, "finite": True, "max_abs": 1.0, "l2_norm": 1.0}
    return {
        "schema_version": D3_RECORD_SCHEMA,
        "status": "SUCCESS",
        "sample_id": "s",
        "image_sha256": "1" * 64,
        "relative_path": "x/y.jpg",
        "label": "normal",
        "mechanism_id": "M2",
        "mechanism": "normalized_logsumexp_softplus",
        "forward_id": "forward-000",
        "evidence_fingerprint": "2" * 64,
        "losses": {"lm": 1.0, "auxiliary": 0.5},
        "score": 0.1,
        "gradients": {
            "lm_shared_global_norm": 1.0,
            "auxiliary_shared_global_norm": 1.0,
            "cosine_defined": True,
            "cosine": 0.5,
            "lm_shared_per_tensor": [dict(grad)],
            "auxiliary_all_trainable_per_tensor": [dict(grad)],
            "non_shared_leakage_count": 0,
        },
        "evidence": {
            "evidence_min": -0.5,
            "evidence_max": 0.5,
            "evidence_mean": 0.0,
            "evidence_saturation_fraction": 0.0,
            "abnormal_probability_min": 0.2,
            "abnormal_probability_max": 0.8,
            "abnormal_probability_mean": 0.5,
            "abnormal_probability_saturation_fraction": 0.0,
            "normal_probability_min": 0.2,
            "normal_probability_max": 0.8,
            "normal_probability_mean": 0.5,
            "normal_probability_saturation_fraction": 0.0,
            "reconstruction_exact": True,
            "reconstruction_max_abs": 0.0,
        },
        "spatial": {
            "entropy": 4.0,
            "effective_support": 54.0,
            "top11_mass": 0.2,
            "max_pooling_weight": 0.03,
            "min_nonzero_pooling_weight": 1e-6,
            "nonzero_pooling_weights": 1024,
            "exact_one_position_collapse": False,
        },
        "memory": {
            "image_peak_allocated_mib": 100.0,
            "image_peak_reserved_mib": 120.0,
            "run_external_peak_mib": 130.0,
        },
        "runtime": {
            "forward_seconds": 1.0,
            "mechanism_loss_seconds": 0.1,
            "mechanism_gradient_seconds": 0.2,
            "image_seconds": 2.0,
        },
        "rng": {"before_image": "3" * 64, "after_image": "4" * 64},
        "provenance": {
            "source_commit": "5" * 40,
            "implementation_fingerprint": "6" * 64,
            "dataset_manifest_sha256": "7" * 64,
            "support_fingerprint": "8" * 64,
        },
    }


class TestD3ArtifactSchema(unittest.TestCase):
    def test_valid_record(self):
        validate_mechanism_record(valid_record())

    def test_undefined_cosine_requires_null(self):
        value = valid_record()
        value["gradients"]["cosine_defined"] = False
        with self.assertRaisesRegex(CABGContractError, "cosine"):
            validate_mechanism_record(value)

    def test_m0_has_no_spatial_metrics(self):
        value = valid_record()
        value["mechanism_id"] = "M0"
        value["mechanism"] = "hard_top1pct_squared_hinge"
        value["spatial"] = None
        validate_mechanism_record(value)


if __name__ == "__main__":
    unittest.main()
