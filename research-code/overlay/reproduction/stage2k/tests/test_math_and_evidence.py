from __future__ import annotations

import unittest

import torch

from reproduction.stage2k.constants import S9_NAMES, TRAINABLE_NAMES
from reproduction.stage2k.evidence import gradient_rows
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2k.verify_d3r import independent_formula, lce_gradient_gate


class TestD3RMathAndEvidence(unittest.TestCase):
    def test_both_classes_have_finite_nonzero_raw_logit_gradient(self):
        abnormal = torch.stack((torch.full((4, 1024), 20.0), torch.full((4, 1024), -20.0))).requires_grad_()
        normal = torch.zeros_like(abnormal, requires_grad=True)
        result = compute_lce(abnormal, normal, ["abnormal", "normal"])
        gradient = torch.autograd.grad(result["per_image_loss"].sum(), abnormal)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient[0] < 0))
        self.assertTrue(torch.all(gradient[1] > 0))
        self.assertEqual(tuple(result["spatial_weights"].shape), (2, 1024))

    def test_independent_formula_matches_and_detects_tamper(self):
        abnormal = torch.linspace(-3, 4, 4096).reshape(1, 4, 1024)
        normal = torch.linspace(2, -2, 4096).reshape(1, 4, 1024)
        result = compute_lce(abnormal, normal, ["abnormal"])
        record = {
            "label": "abnormal",
            "lce": {
                "score": float(result["score"][0]),
                "loss": float(result["per_image_loss"][0]),
                "spatial": {
                    "effective_support": float(result["effective_support"][0]),
                    "top11_mass": float(result["top11_mass"][0]),
                    "max_pooling_weight": float(result["max_pooling_weight"][0]),
                    "min_nonzero_pooling_weight": float(result["min_nonzero_pooling_weight"][0]),
                    "nonzero_pooling_weights": int(result["nonzero_pooling_weights"][0]),
                },
            },
            "formula_payload": {
                "tensor_shape": [1, 4, 1024],
                "raw_logits": {
                    "abnormal_flat": abnormal.reshape(-1).tolist(),
                    "normal_flat": normal.reshape(-1).tolist(),
                },
                "producer_formula": {
                    "gap_flat": result["raw_gap"].reshape(-1).tolist(),
                    "contrast_flat": result["contrast"].reshape(-1).tolist(),
                    "derivative_flat": result["softsign_derivative"].reshape(-1).tolist(),
                    "layer_mean_flat": result["layer_mean"].reshape(-1).tolist(),
                    "weights_flat": result["spatial_weights"].reshape(-1).tolist(),
                },
            },
        }
        self.assertTrue(independent_formula(record)["formula_passed"])
        record["formula_payload"]["producer_formula"]["gap_flat"][0] += 0.1
        self.assertFalse(independent_formula(record)["formula_passed"])

    def test_exact_s9_and_leakage_gate(self):
        gradients = []
        for name in TRAINABLE_NAMES:
            gradients.append(torch.ones(1) if name in S9_NAMES else None)
        rows = gradient_rows(TRAINABLE_NAMES, gradients)
        self.assertTrue(lce_gradient_gate(rows)["passed"])
        rows[TRAINABLE_NAMES.index("model.visual.anomaly_qformer.gate_scale")]["present"] = True
        rows[TRAINABLE_NAMES.index("model.visual.anomaly_qformer.gate_scale")]["l2_norm"] = 1.0
        self.assertFalse(lce_gradient_gate(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
