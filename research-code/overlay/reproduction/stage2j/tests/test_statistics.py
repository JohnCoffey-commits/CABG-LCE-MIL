import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from reproduction.stage2j.candidate_math import compute_softsign_logit_contrast
from reproduction.stage2j.raw_observer import RawAnomalyAttentionObserver
from reproduction.stage2j.statistics import (
    distribution_record,
    gradient_rows,
    pearson,
    spearman,
    support_intersection,
)
from reproduction.stage2j.verify_redesign_r0 import (
    _post_run_gpu_record,
    _reconstruct_bf16_forward_branches,
)


class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.outputs = []

    def forward(self, value):
        score = self.outputs.pop(0)
        return value, score


class TestR0Statistics(unittest.TestCase):
    def test_logit_contrast_is_bounded_finite_and_both_class_active(self):
        abnormal = torch.tensor([[[20.0] * 1024], [[[-20.0] * 1024][0]]], requires_grad=True)
        normal = torch.zeros_like(abnormal, requires_grad=True)
        result = compute_softsign_logit_contrast(abnormal, normal, ["abnormal", "normal"])
        self.assertTrue(torch.isfinite(result["per_image_loss"]).all())
        self.assertLessEqual(float(result["contrast"].abs().max()), 1.0)
        gradient = torch.autograd.grad(result["per_image_loss"].sum(), abnormal)[0]
        self.assertTrue(torch.all(gradient[0] < 0))
        self.assertTrue(torch.all(gradient[1] > 0))

    def test_probability_distribution_records_saturation_and_derivative(self):
        result = distribution_record(torch.tensor([0.0, 0.5, 1.0]), kind="probability")
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["saturation_fraction"], 2 / 3)
        self.assertAlmostEqual(result["sigmoid_derivative_mean"], 1 / 12)

    def test_support_is_actual_intersection(self):
        x = torch.tensor([1.0])
        lm = gradient_rows(["a", "b", "c"], [x, x, None])
        auxiliary = gradient_rows(["a", "b", "c"], [None, x, x])
        result = support_intersection(lm, auxiliary)
        self.assertEqual(result["actual_intersection"], ["b"])

    def test_correlations(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [9, 8, 7]), -1.0)

    def test_raw_observer_reconstructs_evidence(self):
        module = ToyAttention()
        abnormal = torch.tensor([[[[2.0, -1.0]]]])
        normal = torch.tensor([[[[-2.0, 1.0]]]])
        module.outputs = [abnormal, normal]
        with RawAnomalyAttentionObserver(module) as observer:
            module(torch.ones(1))
            module(torch.ones(1))
        evidence = (torch.sigmoid(abnormal) - torch.sigmoid(normal)).reshape(1, 1, 2)
        result = observer.finalize(evidence)
        self.assertTrue(torch.equal(result["evidence"], evidence))
        self.assertTrue(torch.equal(result["raw_gap"], (abnormal - normal).reshape(1, 1, 2)))

    def test_evaluator_reconstructs_bf16_subtraction_before_fp32_conversion(self):
        payload = {
            "dtype": "float32_serialized_from_bfloat16_forward",
            "abnormal_raw_flat": [512.0],
            "normal_raw_flat": [83.0],
            "abnormal_probability_flat": [1.0],
            "normal_probability_flat": [0.5],
        }
        branches = _reconstruct_bf16_forward_branches(payload, (1, 1, 1))
        self.assertEqual(float(branches["raw_gap"].item()), 428.0)
        self.assertEqual(
            float((branches["abnormal_raw"] - branches["normal_raw"]).item()),
            429.0,
        )
        self.assertEqual(float(branches["evidence"].item()), 0.5)

    def test_gpu_idle_uses_memory_and_compute_processes_not_lagging_utilization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = root / "gpu.csv"
            apps = root / "apps.csv"
            gpu.write_text("NVIDIA L4, 23034, 0, 10\n", encoding="utf-8")
            apps.write_text("", encoding="utf-8")
            record = _post_run_gpu_record(gpu, apps)
            self.assertTrue(record["idle_after_run"])
            self.assertEqual(record["utilization_percent_observed"], 10)
            apps.write_text("1234, python, 1024\n", encoding="utf-8")
            self.assertFalse(_post_run_gpu_record(gpu, apps)["idle_after_run"])


if __name__ == "__main__":
    unittest.main()
