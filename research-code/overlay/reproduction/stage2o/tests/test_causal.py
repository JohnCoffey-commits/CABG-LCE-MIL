from __future__ import annotations

import unittest
from pathlib import Path

import torch

from reproduction.stage2m.train_scout import provenance
from reproduction.stage2o import (
    CONTROL_ARM,
    LCE_OFF_ARM,
    PARENT_CHECKPOINT_SHA256,
    PARENT_EVAL_MANIFEST_SHA256,
    PARENT_MANIFEST_SHA256,
    PARENT_RNG_FINGERPRINT,
    PARENT_TRACE_ANCHOR,
    PARENT_TRAIN_MANIFEST_SHA256,
    RESET_ARM,
)
from reproduction.stage2o.causal import apply_causal_gradient, apply_optimizer_intervention, validate_parent_audit
from reproduction.stage2l.constants import S9_NAMES, TRAINABLE_NAMES
from reproduction.stage2o.summarize_causal import _registered_optimizer_delta, _stability_and_benefit, classify_terminal


def _optimizer():
    names = ("s1", "s2", "other")
    parameters = {name: torch.nn.Parameter(torch.tensor([float(index + 1)])) for index, name in enumerate(names)}
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=0.1)
    for parameter in parameters.values():
        parameter.grad = torch.tensor([2.0])
    optimizer.step()
    return names, parameters, optimizer


def _details():
    return {
        "aggregate_lm": {"p": torch.tensor([3.0]), "q": torch.tensor([4.0])},
        "aggregate_lce": {"p": torch.tensor([2.0])},
        "budget": {"lambda_final": 0.1, "lambda_raw": 0.2, "lambda_cap": 0.1},
        "gradient": {},
    }


class CausalInterventionTests(unittest.TestCase):
    def test_control_keeps_dynamic_lambda(self):
        applied, details, multiplier = apply_causal_gradient(CONTROL_ARM, _details())
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(details["budget"]["lambda_controller"], 0.1)
        self.assertEqual(details["budget"]["lambda_final"], 0.1)
        self.assertGreater(float(applied["p"]), 0.6)

    def test_both_lce_off_branches_are_exact_lm_only(self):
        for arm in (LCE_OFF_ARM, RESET_ARM):
            applied, details, multiplier = apply_causal_gradient(arm, _details())
            self.assertEqual(multiplier, 0.0)
            self.assertEqual(details["budget"]["lambda_final"], 0.0)
            torch.testing.assert_close(applied["p"], torch.tensor([0.6]))
            torch.testing.assert_close(applied["q"], torch.tensor([0.8]))

    def test_state_kept_branch_changes_no_optimizer_tensor(self):
        names, parameters, optimizer = _optimizer()
        audit = apply_optimizer_intervention(LCE_OFF_ARM, optimizer, parameters, names=names, s9_names=("s1", "s2"))
        self.assertFalse(audit["applied"])
        self.assertEqual(audit["changed_state_tensors"], [])
        self.assertEqual(audit["before_inventory_sha256"], audit["after_inventory_sha256"])

    def test_reset_changes_only_two_test_s9_first_moments(self):
        names, parameters, optimizer = _optimizer()
        second = {name: optimizer.state[parameters[name]]["exp_avg_sq"].clone() for name in names}
        steps = {name: optimizer.state[parameters[name]]["step"].clone() for name in names}
        audit = apply_optimizer_intervention(RESET_ARM, optimizer, parameters, names=names, s9_names=("s1", "s2"))
        self.assertEqual(audit["changed_state_tensors"], ["s1.exp_avg", "s2.exp_avg"])
        self.assertTrue(audit["exp_avg_sq_preserved"])
        self.assertTrue(audit["step_preserved"])
        self.assertTrue(audit["non_s9_state_preserved"])
        for name in names:
            torch.testing.assert_close(optimizer.state[parameters[name]]["exp_avg_sq"], second[name])
            torch.testing.assert_close(optimizer.state[parameters[name]]["step"], steps[name])

    def test_reset_accepts_semantic_s9_order_different_from_canonical_t21_order(self):
        names, parameters, optimizer = _optimizer()
        audit = apply_optimizer_intervention(
            RESET_ARM,
            optimizer,
            parameters,
            names=names,
            s9_names=("s2", "s1"),
        )
        self.assertEqual(audit["changed_state_tensors"], ["s1.exp_avg", "s2.exp_avg"])
        self.assertEqual(audit["changed_state_tensor_count"], 2)
        self.assertTrue(audit["exp_avg_sq_preserved"])
        self.assertTrue(audit["step_preserved"])
        self.assertTrue(audit["non_s9_state_preserved"])


class CausalContractTests(unittest.TestCase):
    def test_independent_evaluator_expects_reset_delta_in_canonical_t21_order(self):
        expected = [f"{name}.exp_avg" for name in TRAINABLE_NAMES if name in set(S9_NAMES)]
        self.assertEqual(_registered_optimizer_delta(RESET_ARM), expected)
        self.assertNotEqual(expected, [f"{name}.exp_avg" for name in S9_NAMES])
        self.assertEqual(_registered_optimizer_delta(LCE_OFF_ARM), [])

    def test_three_provenances_differ_only_by_registered_arm_fields(self):
        preflight = {"repository": {"commit": "c"}, "source": {"fingerprint": "s"}, "base_checkpoint": {"fingerprint": "b"}}
        split = {"train_manifest_sha256": "t", "eval_manifest_sha256": "e"}
        values = [provenance(preflight, split, arm) for arm in (CONTROL_ARM, LCE_OFF_ARM, RESET_ARM)]
        for left, right in zip(values, values[1:]):
            self.assertEqual({key for key in left if left[key] != right[key]}, {"arm", "applied_objective"})

    def test_parent_audit_is_exact_and_fail_closed(self):
        audit = {
            "status": "SUCCESS", "checkpoint_sha256": PARENT_CHECKPOINT_SHA256, "manifest_sha256": PARENT_MANIFEST_SHA256,
            "cursor": 12, "trace_anchor": PARENT_TRACE_ANCHOR, "rng_fingerprint": PARENT_RNG_FINGERPRINT,
            "train_manifest_sha256": PARENT_TRAIN_MANIFEST_SHA256, "eval_manifest_sha256": PARENT_EVAL_MANIFEST_SHA256,
            "continuation_blocks": 12, "continuation_exposures": 36, "all_registered_overlaps_zero": True,
            "protected_internal_test_image_files_opened": 0, "protected_internal_test_outputs_read": 0,
        }
        validate_parent_audit(audit)
        with self.assertRaises(RuntimeError):
            validate_parent_audit({**audit, "cursor": 13})

    def test_independent_evaluator_does_not_import_producer_intervention(self):
        source = (Path(__file__).parents[1] / "summarize_causal.py").read_text()
        self.assertNotIn("from reproduction.stage2m.train_scout", source)
        self.assertNotIn("from reproduction.stage2o.causal", source)

    def test_wrapper_is_control_first_new_root_and_fail_closed(self):
        source = (Path(__file__).parents[1] / "run_causal_branch_v1_l4.sh").read_text()
        self.assertIn('CAUSAL_RUN_VERSION:-v1', source)
        self.assertLess(source.index("control_primary"), source.index("lce_off_state_kept"))
        self.assertIn("refusing to overwrite", source)
        self.assertNotIn("internal-test", source.lower())

    def test_v2_wrapper_selects_a_fresh_formal_root(self):
        source = (Path(__file__).parents[1] / "run_causal_branch_v2_l4.sh").read_text()
        self.assertIn("CAUSAL_RUN_VERSION=v2", source)
        self.assertIn("run_causal_branch_v1_l4.sh", source)


class CausalDecisionTests(unittest.TestCase):
    def test_mutually_exclusive_terminal_mapping(self):
        stable = {"spatial_stability_pass": True, "benefit_retention_pass": True}
        unstable = {"spatial_stability_pass": False, "benefit_retention_pass": True}
        lost = {"spatial_stability_pass": True, "benefit_retention_pass": False}
        self.assertEqual(classify_terminal(primary_control_reproduced=True, repeat_control_reproduced=None, state_kept=stable, reset=stable), "CONTROLLER_REPAIR_ELIGIBLE")
        self.assertEqual(classify_terminal(primary_control_reproduced=True, repeat_control_reproduced=None, state_kept=unstable, reset=stable), "OPTIMIZER_REPAIR_ELIGIBLE")
        self.assertEqual(classify_terminal(primary_control_reproduced=True, repeat_control_reproduced=None, state_kept=unstable, reset=unstable), "REDESIGN_REQUIRED")
        self.assertEqual(classify_terminal(primary_control_reproduced=True, repeat_control_reproduced=None, state_kept=lost, reset=lost), "REDESIGN_REQUIRED")
        self.assertEqual(classify_terminal(primary_control_reproduced=False, repeat_control_reproduced=False, state_kept=None, reset=None), "INCONCLUSIVE_REPRODUCIBILITY")
        self.assertEqual(classify_terminal(primary_control_reproduced=False, repeat_control_reproduced=True, state_kept=stable, reset=stable), "INCONCLUSIVE_REPRODUCIBILITY")

    def test_stability_and_gain_rules_are_locked(self):
        original = {"metrics": {"lm_only": {"auroc": 0.22916666666666666, "average_precision": 0.4884123424270804}, "cabg_lce": {"auroc": 0.9708333333333333, "average_precision": 0.9848684210526315}}}
        branch = {"metrics": {"auroc": 0.95, "average_precision": 0.97, "spatial": {"effective_support_median": 300.0, "support_below_128_count": 8, "top11_above_0_35_count": 8}}, "training": {"last_four_failed_blocks": 1}}
        result = _stability_and_benefit(branch, original)
        self.assertTrue(result["spatial_stability_pass"])
        self.assertTrue(result["benefit_retention_pass"])


if __name__ == "__main__":
    unittest.main()
