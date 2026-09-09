import copy
import math
import unittest

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2l.constants import RHO_MAX
from reproduction.stage2l.controller import DynamicController, compute_block_gradients, exact_s9_from_full


T = ("s1", "s2", "outside")
S = ("s1", "s2")


def block(scale_lm=1.0, scale_lce=1.0):
    labels = ["normal", "abnormal", "abnormal"]
    lm = [
        {"s1": torch.tensor([1.0, 0.0]) * scale_lm, "s2": torch.tensor([0.2]) * scale_lm, "outside": torch.tensor([0.5]) * scale_lm},
        {"s1": torch.tensor([0.0, 2.0]) * scale_lm, "s2": torch.tensor([0.3]) * scale_lm, "outside": torch.tensor([0.6]) * scale_lm},
        {"s1": torch.tensor([-1.0, 1.0]) * scale_lm, "s2": torch.tensor([0.4]) * scale_lm, "outside": torch.tensor([0.7]) * scale_lm},
    ]
    lce = [
        {"s1": torch.tensor([2.0, 0.0]) * scale_lce, "s2": torch.tensor([0.5]) * scale_lce},
        {"s1": torch.tensor([0.0, -3.0]) * scale_lce, "s2": torch.tensor([0.6]) * scale_lce},
        {"s1": torch.tensor([1.0, 1.0]) * scale_lce, "s2": torch.tensor([0.7]) * scale_lce},
    ]
    return lm, lce, labels


class ControllerTests(unittest.TestCase):
    def test_locked_class_balanced_rms_and_ema(self):
        controller = DynamicController()
        lm, lce, labels = block()
        _, first = compute_block_gradients(lm, lce, labels, trainable_names=T, s9_names=S, controller=controller)
        self.assertEqual(first["budget"]["valid_blocks"], 1)
        self.assertAlmostEqual(first["budget"]["corrected_lm_rms"], first["lm_rms"], places=6)
        self.assertAlmostEqual(first["budget"]["corrected_lce_rms"], first["lce_rms"], places=6)
        _, second = compute_block_gradients(*block(2.0, 0.5), trainable_names=T, s9_names=S, controller=controller)
        expected = (0.9 * first["lm_rms"] + second["lm_rms"]) / 1.9
        self.assertAlmostEqual(second["budget"]["corrected_lm_rms"], expected, places=5)

    def test_trust_cap_always_holds_and_can_activate(self):
        controller = DynamicController()
        applied, details = compute_block_gradients(*block(0.01, 100.0), trainable_names=T, s9_names=S, controller=controller)
        self.assertLessEqual(details["gradient"]["scaled_auxiliary_shared_norm"],
                             RHO_MAX * details["gradient"]["lm_shared_norm"] + details["gradient"]["trust_cap_tolerance"])
        self.assertAlmostEqual(details["budget"]["lambda_final"], min(details["budget"]["lambda_raw"], details["budget"]["lambda_cap"]), places=12)
        self.assertEqual(set(applied), set(T))
        direct = DynamicController().update(
            lm_rms=100.0, lce_rms=1.0,
            aggregate_lm_s9={"s1": torch.ones(1)},
            aggregate_lce_s9={"s1": torch.full((1,), 100.0)},
        )
        self.assertEqual(direct["lambda_final"], direct["lambda_cap"])

    def test_outside_s9_is_lm_only(self):
        controller = DynamicController()
        lm, lce, labels = block()
        applied, details = compute_block_gradients(lm, lce, labels, trainable_names=T, s9_names=S, controller=controller)
        lm_mean = torch.stack([row["outside"] for row in lm]).mean(0)
        expected = lm_mean * details["gradient"]["clip_scale"]
        torch.testing.assert_close(applied["outside"], expected)

    def test_manual_combination_matches_scalar_autograd(self):
        shared = torch.tensor([0.2, -0.1], requires_grad=True)
        outside = torch.tensor([0.4], requires_grad=True)
        lm_loss = 0.1 * (shared.square().sum() + outside.square().sum())
        lce_loss = (shared * torch.tensor([0.1, -0.2])).sum()
        lm_grad = torch.autograd.grad(lm_loss, (shared, outside), retain_graph=True)
        lce_grad = torch.autograd.grad(lce_loss, shared, retain_graph=True)[0]
        value = 0.03
        reference = torch.autograd.grad(lm_loss + value * lce_loss, (shared, outside))
        torch.testing.assert_close(lm_grad[0] + value * lce_grad, reference[0])
        torch.testing.assert_close(lm_grad[1], reference[1])

    def test_partial_block_zero_nonfinite_and_support_leak_fail_closed(self):
        lm, lce, labels = block()
        for bad_labels in (["normal"] * 3, ["abnormal"] * 3):
            with self.assertRaises(CABGContractError):
                compute_block_gradients(lm, lce, bad_labels, trainable_names=T, s9_names=S, controller=DynamicController())
        zero = copy.deepcopy(lce)
        zero[0] = {name: torch.zeros_like(value) for name, value in zero[0].items()}
        with self.assertRaises(CABGContractError):
            compute_block_gradients(lm, zero, labels, trainable_names=T, s9_names=S, controller=DynamicController())
        bad = copy.deepcopy(lm)
        bad[0]["s1"][0] = float("nan")
        with self.assertRaises(CABGContractError):
            compute_block_gradients(bad, lce, labels, trainable_names=T, s9_names=S, controller=DynamicController())
        with self.assertRaises(CABGContractError):
            exact_s9_from_full({"s1": torch.ones(1), "s2": torch.ones(1), "outside": torch.ones(1)}, S)

    def test_controller_state_roundtrip_and_schema_rejection(self):
        first = DynamicController()
        compute_block_gradients(*block(), trainable_names=T, s9_names=S, controller=first)
        second = DynamicController()
        second.load_state_dict(first.state_dict())
        self.assertEqual(first.state_dict(), second.state_dict())
        bad = first.state_dict()
        bad["unexpected"] = 1
        with self.assertRaises(CABGContractError):
            second.load_state_dict(bad)

    def test_gradient_accumulation_is_image_mean_not_class_balanced_lm(self):
        lm, lce, labels = block()
        _, details = compute_block_gradients(lm, lce, labels, trainable_names=T, s9_names=S, controller=DynamicController())
        for name in T:
            torch.testing.assert_close(details["aggregate_lm"][name], torch.stack([row[name] for row in lm]).mean(0))

    def test_softsign_extreme_gradient_ratio_remains_finite(self):
        controller = DynamicController()
        for lm_scale, lce_scale in ((1e-8, 1e8), (1e8, 1e-8), (1.0, 1.0)):
            _, details = compute_block_gradients(*block(lm_scale, lce_scale), trainable_names=T, s9_names=S, controller=controller)
            self.assertTrue(all(math.isfinite(float(value)) for value in details["budget"].values()))


if __name__ == "__main__":
    unittest.main()
