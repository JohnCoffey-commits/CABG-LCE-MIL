import math
import unittest

import torch

from reproduction.stage2i.cabg_math import (
    CABGContractError,
    BiasCorrectedEMA,
    class_balanced_mean,
    combine_and_clip_gradients,
    compute_budget,
    normalized_logsumexp_score,
    per_image_class_balanced_rms,
    smooth_evidence_objective,
    validate_auxiliary_support,
    validate_block_labels,
)
from reproduction.stage2i.constants import RHO_MAX, TAU


class NormalizedLogSumExpTests(unittest.TestCase):
    def test_matches_fp64_reference_and_is_finite_at_bounds(self):
        evidence = torch.linspace(-1.0, 1.0, 2048, dtype=torch.float64).reshape(2, 1024)
        observed = normalized_logsumexp_score(evidence)
        reference = TAU * (torch.logsumexp(evidence / TAU, dim=-1) - math.log(1024))
        torch.testing.assert_close(observed, reference, rtol=0, atol=1e-14)
        self.assertTrue(torch.isfinite(observed).all())
        self.assertTrue(torch.all(observed >= evidence.min(dim=-1).values))
        self.assertTrue(torch.all(observed <= evidence.max(dim=-1).values))

    def test_analytical_pooling_derivative_gradcheck(self):
        value = torch.linspace(-0.8, 0.9, 12, dtype=torch.float64, requires_grad=True).reshape(2, 6)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda tensor: normalized_logsumexp_score(tensor, tau=0.2),
                (value,),
                eps=1e-6,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_softplus_class_direction_and_both_class_activity(self):
        score = torch.tensor([0.15, 0.15], dtype=torch.float64, requires_grad=True)
        evidence = score.reshape(2, 1, 1).expand(2, 1, 8)
        result = smooth_evidence_objective(evidence, ["normal", "abnormal"], tau=0.25)
        gradients = torch.autograd.grad(result["per_image_loss"].sum(), score)[0]
        self.assertGreater(float(gradients[0]), 0.0)
        self.assertLess(float(gradients[1]), 0.0)
        self.assertTrue(torch.all(gradients.abs() > 0))

    def test_single_image_microbatch_is_valid_and_bf16_reduces_in_fp32(self):
        evidence = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.bfloat16)
        result = smooth_evidence_objective(evidence, ["abnormal"], tau=0.25)
        explicit_layer_mean = evidence.float().mean(dim=1)
        expected = normalized_logsumexp_score(explicit_layer_mean, tau=0.25)
        self.assertEqual(result["layer_mean"].dtype, torch.float32)
        torch.testing.assert_close(result["scores"], expected, rtol=0, atol=0)
        self.assertGreater(float(result["per_image_loss"][0]), 0.0)


class ReductionAndBudgetTests(unittest.TestCase):
    def test_class_balanced_reduction_is_invariant_to_class_duplication(self):
        base = class_balanced_mean(torch.tensor([2.0, 6.0]), ["normal", "abnormal"])
        duplicated = class_balanced_mean(
            torch.tensor([2.0, 2.0, 2.0, 6.0]),
            ["normal", "normal", "normal", "abnormal"],
        )
        self.assertEqual(float(base), float(duplicated))

    def test_per_image_rms_matches_explicit_concatenation(self):
        gradients = [
            {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0])},
            {"a": torch.tensor([0.0, 2.0]), "b": torch.tensor([0.0])},
            {"a": torch.tensor([0.0, 0.0]), "b": torch.tensor([6.0])},
        ]
        observed, classes = per_image_class_balanced_rms(
            gradients, ["normal", "abnormal", "abnormal"], epsilon=0.0
        )
        expected = math.sqrt(0.5 * (5.0**2 + (2.0**2 + 6.0**2) / 2.0))
        self.assertAlmostEqual(float(observed), expected, places=5)
        self.assertAlmostEqual(float(classes["normal"]), 5.0, places=5)
        self.assertAlmostEqual(float(classes["abnormal"]), math.sqrt(20.0), places=5)

    def test_ema_bias_correction_matches_closed_form_first_five_blocks(self):
        ema = BiasCorrectedEMA(beta=0.9)
        lm_values = [1.0, 2.0, 4.0, 3.0, 5.0]
        auxiliary_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        lm_raw = auxiliary_raw = 0.0
        for index, (lm_value, auxiliary_value) in enumerate(
            zip(lm_values, auxiliary_values), start=1
        ):
            lm_raw = 0.9 * lm_raw + 0.1 * lm_value
            auxiliary_raw = 0.9 * auxiliary_raw + 0.1 * auxiliary_value
            observed_lm, observed_auxiliary = ema.update(lm_value, auxiliary_value)
            correction = 1.0 - 0.9**index
            self.assertAlmostEqual(observed_lm, lm_raw / correction, places=12)
            self.assertAlmostEqual(observed_auxiliary, auxiliary_raw / correction, places=12)

    def test_trust_cap_holds_for_aligned_orthogonal_and_opposed_gradients(self):
        directions = (
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1.0, 0.0]),
        )
        for auxiliary in directions:
            with self.subTest(auxiliary=auxiliary.tolist()):
                lm = {"shared": torch.tensor([1.0, 0.0]), "other": torch.tensor([0.2])}
                aux = {"shared": auxiliary}
                cap = RHO_MAX * 1.0 / (float(auxiliary.norm()) + 1e-8)
                _, diagnostics = combine_and_clip_gradients(
                    lm,
                    aux,
                    shared_names=["shared"],
                    lambda_value=cap,
                    max_norm=100.0,
                )
                self.assertLessEqual(
                    diagnostics["scaled_auxiliary_shared_norm"],
                    RHO_MAX * diagnostics["lm_shared_norm"] + diagnostics["trust_cap_tolerance"],
                )
        with self.assertRaises(CABGContractError):
            combine_and_clip_gradients(
                {"shared": torch.tensor([1.0, 0.0])},
                {"shared": torch.tensor([1.0, 0.0])},
                shared_names=["shared"],
                lambda_value=0.21,
            )

    def test_compute_budget_rejects_zero_lm_reference(self):
        with self.assertRaises(CABGContractError):
            compute_budget(
                lm_rms=1.0,
                auxiliary_rms=1.0,
                lm_shared_gradient={"shared": torch.zeros(2)},
                auxiliary_shared_gradient={"shared": torch.ones(2)},
                ema=BiasCorrectedEMA(),
            )


class GradientContractTests(unittest.TestCase):
    def test_support_isolation_accepts_disconnected_nonshared_and_rejects_leak(self):
        shared_a = torch.tensor(2.0, requires_grad=True)
        shared_b = torch.tensor(3.0, requires_grad=True)
        nonshared = torch.tensor(4.0, requires_grad=True)
        auxiliary = shared_a.square() + shared_b.square()
        gradients = torch.autograd.grad(
            auxiliary, [shared_a, shared_b, nonshared], allow_unused=True
        )
        named = dict(zip(("shared_a", "shared_b", "nonshared"), gradients))
        validate_auxiliary_support(named, ["shared_a", "shared_b"])
        leaked = dict(named)
        leaked["nonshared"] = torch.tensor(1.0)
        with self.assertRaises(CABGContractError):
            validate_auxiliary_support(leaked, ["shared_a", "shared_b"])

    def test_manual_combined_gradient_matches_scalar_backward(self):
        shared = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
        other = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
        lm_loss = (shared.square().sum() + other.square().sum()) * 0.1
        auxiliary_loss = (shared * torch.tensor([0.1, -0.2], dtype=torch.float64)).sum()
        lm_gradients = torch.autograd.grad(lm_loss, [shared, other], retain_graph=True)
        auxiliary_gradient = torch.autograd.grad(auxiliary_loss, shared, retain_graph=True)[0]
        lambda_value = 0.01
        manual, _ = combine_and_clip_gradients(
            {"shared": lm_gradients[0], "other": lm_gradients[1]},
            {"shared": auxiliary_gradient},
            shared_names=["shared"],
            lambda_value=lambda_value,
            max_norm=1e9,
        )
        combined_loss = lm_loss + lambda_value * auxiliary_loss
        reference = torch.autograd.grad(combined_loss, [shared, other])
        torch.testing.assert_close(manual["shared"].double(), reference[0], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(manual["other"].double(), reference[1], rtol=1e-6, atol=1e-8)

    def test_cap_is_checked_before_full_gradient_clipping(self):
        lm = {"shared": torch.tensor([1.0]), "other": torch.tensor([100.0])}
        auxiliary = {"shared": torch.tensor([100.0])}
        _, diagnostics = combine_and_clip_gradients(
            lm,
            auxiliary,
            shared_names=["shared"],
            lambda_value=0.002,
            max_norm=1.0,
        )
        self.assertTrue(diagnostics["cap_checked_before_clip"])
        self.assertGreater(diagnostics["preclip_full_norm"], 1.0)
        self.assertLessEqual(diagnostics["postclip_full_norm"], 1.0 + 1e-6)
        with self.assertRaises(CABGContractError):
            combine_and_clip_gradients(
                lm,
                auxiliary,
                shared_names=["shared"],
                lambda_value=0.003,
                max_norm=1.0,
            )

    def test_invalid_blocks_and_nonfinite_gradients_fail_closed(self):
        with self.assertRaises(CABGContractError):
            validate_block_labels(["normal", "normal"])
        with self.assertRaises(CABGContractError):
            per_image_class_balanced_rms(
                [{"x": torch.tensor([float("nan")])}, {"x": torch.ones(1)}],
                ["normal", "abnormal"],
            )
        with self.assertRaises(CABGContractError):
            validate_auxiliary_support({"shared": None}, ["shared"])


if __name__ == "__main__":
    unittest.main()
