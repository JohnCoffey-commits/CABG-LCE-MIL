from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from reproduction.stage2m.train_scout import provenance, reset_s9_directional_moments
from reproduction.stage2n import MOMENT_RESET_ARM
from reproduction.stage2n.repair import (
    apply_repair_multiplier,
    cosine_taper_multiplier,
    initial_repair_state,
    repair_decision,
    validate_repair_state,
)
from reproduction.stage2n.summarize_repair import classify, visible_update_support_valid


def _images(support: float, top11: float) -> list[dict[str, float]]:
    return [{"effective_support": support + index, "top11_mass": top11} for index in range(3)]


class RepairScheduleTests(unittest.TestCase):
    def test_cosine_taper_boundaries(self):
        self.assertEqual(cosine_taper_multiplier(0), 1.0)
        self.assertEqual(cosine_taper_multiplier(11), 1.0)
        self.assertAlmostEqual(cosine_taper_multiplier(12), 0.5 * (1 + math.cos(math.pi / 8)))
        self.assertAlmostEqual(cosine_taper_multiplier(15), 0.5)
        self.assertEqual(cosine_taper_multiplier(19), 0.0)
        self.assertEqual(cosine_taper_multiplier(23), 0.0)
        with self.assertRaises(RuntimeError):
            cosine_taper_multiplier(24)

    def test_spatial_guard_requires_persistence_and_latches(self):
        arm = "cabg_spatial_guard"
        state = initial_repair_state(arm)
        multipliers = []
        sequence = [(400, 0.1), (200, 0.1), (180, 0.1), (400, 0.1), (240, 0.1), (230, 0.1), (220, 0.1), (500, 0.1)]
        for block_id, (support, top11) in enumerate(sequence):
            record, state = repair_decision(arm, block_id, _images(support, top11), state)
            multipliers.append(record["multiplier"])
        self.assertEqual(multipliers, [1.0, 1.0, 0.25, 1.0, 1.0, 0.25, 0.0, 0.0])
        self.assertEqual(state, {"variant": arm, "risk_streak": 3, "latched": True})

    def test_guard_uses_top11_or_support_and_never_heldout(self):
        arm = "cabg_spatial_guard"
        state = initial_repair_state(arm)
        record, _ = repair_decision(arm, 0, _images(500, 0.3), state)
        self.assertTrue(record["training_spatial_input"]["early_warning"])
        self.assertFalse(record["heldout_input_used"])

    def test_moment_reset_arm_uses_the_same_guard_state_machine(self):
        state = initial_repair_state(MOMENT_RESET_ARM)
        for block_id in range(3):
            record, state = repair_decision(MOMENT_RESET_ARM, block_id, _images(200, 0.1), state)
        self.assertEqual(record["multiplier"], 0.0)
        self.assertTrue(state["latched"])

    def test_repair_state_rejects_wrong_identity(self):
        with self.assertRaises(RuntimeError):
            validate_repair_state("cabg_spatial_guard", {"variant": "cabg_cosine_taper"})


class RepairGradientTests(unittest.TestCase):
    def _details(self):
        return {
            "aggregate_lm": {"p": torch.tensor([3.0]), "q": torch.tensor([4.0])},
            "aggregate_lce": {"p": torch.tensor([2.0])},
            "budget": {"lambda_final": 0.1, "lambda_raw": 0.2, "lambda_cap": 0.1},
            "gradient": {},
        }

    def test_zero_multiplier_is_exact_lm_only_then_global_clip(self):
        applied, details = apply_repair_multiplier(self._details(), 0.0)
        torch.testing.assert_close(applied["p"], torch.tensor([0.6]))
        torch.testing.assert_close(applied["q"], torch.tensor([0.8]))
        self.assertEqual(details["budget"]["lambda_controller"], 0.1)
        self.assertEqual(details["budget"]["lambda_final"], 0.0)
        self.assertEqual(details["gradient"]["scaled_auxiliary_shared_norm"], 0.0)

    def test_multiplier_one_preserves_controller_lambda(self):
        _, details = apply_repair_multiplier(self._details(), 1.0)
        self.assertEqual(details["budget"]["lambda_controller"], 0.1)
        self.assertEqual(details["budget"]["lambda_final"], 0.1)
        self.assertLessEqual(details["gradient"]["scaled_auxiliary_shared_norm"] / details["gradient"]["lm_shared_norm"], 0.20001)

    def test_directional_moment_reset_preserves_second_moment_and_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        state = optimizer.state[parameter]
        second_before = state["exp_avg_sq"].clone()
        step_before = state["step"].clone()
        result = reset_s9_directional_moments(optimizer, {"p": parameter}, names=("p",))
        self.assertEqual(result["pre_reset_nonzero_parameter_count"], 1)
        self.assertTrue(torch.equal(state["exp_avg"], torch.zeros_like(state["exp_avg"])))
        self.assertTrue(torch.equal(state["exp_avg_sq"], second_before))
        self.assertTrue(torch.equal(state["step"], step_before))


class RepairContractTests(unittest.TestCase):
    def test_repair_provenance_differs_only_by_registered_arm_fields(self):
        preflight = {"repository": {"commit": "c"}, "source": {"fingerprint": "s"}, "base_checkpoint": {"fingerprint": "b"}}
        split = {"train_manifest_sha256": "t", "eval_manifest_sha256": "e"}
        taper = provenance(preflight, split, "cabg_cosine_taper")
        guard = provenance(preflight, split, "cabg_spatial_guard")
        self.assertEqual({key for key in taper if taper[key] != guard[key]}, {"arm", "applied_objective"})

    def test_registered_classification(self):
        lm = {"auroc": 0.22916666666666666, "average_precision": 0.4884123424270804}
        original = {"auroc": 0.9708333333333333, "average_precision": 0.9848684210526315}
        stable_spatial = {"effective_support_median": 300.0, "support_below_128_count": 2, "top11_above_0_35_count": 3}
        unstable_spatial = {"effective_support_median": 40.0, "support_below_128_count": 25, "top11_above_0_35_count": 24}
        common_training = {"last_four_failed_blocks": 0}
        a = classify(metrics={"auroc": 0.95, "average_precision": 0.97, "spatial": stable_spatial}, training=common_training, lm_metrics=lm, original_metrics=original)
        b = classify(metrics={"auroc": 0.80, "average_precision": 0.90, "spatial": stable_spatial}, training=common_training, lm_metrics=lm, original_metrics=original)
        c = classify(metrics={"auroc": 0.95, "average_precision": 0.97, "spatial": unstable_spatial}, training=common_training, lm_metrics=lm, original_metrics=original)
        d = classify(metrics={"auroc": 0.40, "average_precision": 0.50, "spatial": unstable_spatial}, training=common_training, lm_metrics=lm, original_metrics=original)
        self.assertEqual([a["decision"], b["decision"], c["decision"], d["decision"]], ["A", "B", "C", "D"])

    def test_visible_update_contract_allows_fp32_resolution_loss(self):
        self.assertTrue(visible_update_support_valid(21, 20))
        self.assertTrue(visible_update_support_valid(21, 1))
        self.assertFalse(visible_update_support_valid(21, 0))
        self.assertFalse(visible_update_support_valid(20, 20))

    def test_wrapper_has_new_roots_and_no_internal_test(self):
        wrapper = Path(__file__).parents[1] / "run_stability_repair_v1_l4.sh"
        text = wrapper.read_text()
        self.assertIn("d4-scout-stability-repair-v1", text)
        self.assertIn("d4-scout-v1/split-audit.json", text)
        self.assertIn("set -euo pipefail", text)
        self.assertNotIn("internal-test", text.lower())

    def test_adjustment_wrapper_is_single_arm_and_new_root(self):
        wrapper = Path(__file__).parents[1] / "run_moment_reset_adjustment_v1_l4.sh"
        text = wrapper.read_text()
        self.assertIn("d4-scout-stability-repair-moment-reset-v1", text)
        self.assertIn("cabg_spatial_guard_moment_reset", text)
        self.assertNotIn("internal-test", text.lower())


if __name__ == "__main__":
    unittest.main()
