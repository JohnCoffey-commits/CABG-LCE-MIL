import copy
import math
import unittest
from pathlib import Path

import torch

from reproduction.stage2l.controller import DynamicController, compute_block_gradients
from reproduction.stage2m.train_scout import matched_configuration, provenance
from reproduction.stage2i.fingerprint import canonical_json_sha256
from reproduction.stage2p import ARMS, CONFIG_SHA256
from reproduction.stage2p.schedule import cutoff_multiplier, apply_fixed_cutoff, validate_origin
from reproduction.stage2p.verify_fixed_cutoff import verify_scalars, prefix_distance, terminal, endpoint


def fixture(arm):
    controller, trace = DynamicController(), []
    lm = [{"p": torch.tensor([3.0+i]), "q": torch.tensor([4.0-i])} for i in range(3)]
    lce = [{"p": torch.tensor([1.0+i])} for i in range(3)]
    for i in range(24):
        applied, details = compute_block_gradients(lm, lce, ["normal", "abnormal", "abnormal"], trainable_names=("p", "q"), s9_names=("p",), controller=controller)
        applied, details, record = apply_fixed_cutoff(arm, i, applied, details)
        budget, gradient = dict(details["budget"]), details["gradient"]
        budget.update(cap_active=budget["lambda_controller"] + 1e-12 < budget["lambda_raw"], trust_ratio=gradient["scaled_auxiliary_shared_norm"]/(gradient["lm_shared_norm"]+1e-12))
        trace.append({"images": [{"label": label, "lm_s9_norm": float(lm[j]["p"].norm()), "lce_s9_norm": float(lce[j]["p"].norm())} for j, label in enumerate(("normal", "abnormal", "abnormal"))], "class_rms": {"lm": details["lm_class_rms"], "lce": details["lce_class_rms"], "lm_balanced": details["lm_rms"], "lce_balanced": details["lce_rms"]}, "controller": budget, "gradient": gradient, "fixed_cutoff": record, "scheduler": {"step": i+1}, "optimizer": {"step": i+1, "learning_rate_used": 1e-4*.5*(1+math.cos(math.pi*i/24))}, "parameter_update": {"changed_tensor_count": 2}, "rng": {"before_block": str(i), "after_block": str(i+1)}})
    return trace


class FixedCutoffTests(unittest.TestCase):
    def test_locked_boundary(self):
        self.assertEqual([cutoff_multiplier(ARMS[1], i) for i in range(24)], [1.0]*12+[0.0]*12)
        self.assertEqual([cutoff_multiplier(ARMS[0], i) for i in range(24)], [1.0]*24)

    def test_invalid_schedule(self):
        for arm, index in (("other", 1), (ARMS[0], -1), (ARMS[1], 24), (ARMS[1], True)):
            with self.assertRaises(RuntimeError):
                cutoff_multiplier(arm, index)

    def test_prefix_preserves_gradient_objects(self):
        applied = {"p": torch.tensor([1.0])}
        details = {"budget": {"lambda_final": .1}, "gradient": {"clip_scale": .2}}
        for arm in ARMS:
            output, modified, record = apply_fixed_cutoff(arm, 11, applied, details)
            self.assertIs(output, applied)
            self.assertIs(modified["gradient"], details["gradient"])
            self.assertNotIn("lambda_controller", details["budget"])

    def test_zero_is_exact_lm_then_clip(self):
        lm = [{"p": torch.tensor([3.]), "q": torch.tensor([4.])} for _ in range(3)]
        lce = [{"p": torch.tensor([2.])} for _ in range(3)]
        original, details = compute_block_gradients(lm, lce, ["normal", "abnormal", "abnormal"], trainable_names=("p", "q"), s9_names=("p",), controller=DynamicController())
        applied, output, _ = apply_fixed_cutoff(ARMS[1], 12, original, details)
        torch.testing.assert_close(applied["p"], torch.tensor([.6]))
        torch.testing.assert_close(applied["q"], torch.tensor([.8]))
        self.assertEqual(output["gradient"]["scaled_auxiliary_shared_norm"], 0.)
        self.assertGreater(output["budget"]["lambda_controller"], 0.)

    def test_own_resume_only(self):
        root = Path("/new-run")
        for arm in ARMS:
            validate_origin(arm, "phase1", root, None, None)
            validate_origin(arm, "phase2", root, root/arm/"checkpoints/mid-block12.pt", None)
            for phase, resume, parent in (("phase1", Path("/old.pt"), None), ("phase2", Path("/old.pt"), None), ("phase1", None, Path("/audit.json")), ("phase2", root/ARMS[1 if arm == ARMS[0] else 0]/"checkpoints/mid-block12.pt", None)):
                with self.assertRaises(RuntimeError):
                    validate_origin(arm, phase, root, resume, parent)

    def test_independent_24_block_scalar_recomputation(self):
        for arm in ARMS:
            verify_scalars(fixture(arm), arm)

    def test_evaluator_rejects_controller_schedule_rng_clip_mutations(self):
        original = fixture(ARMS[1])
        changes = [("controller", "lm_ema", 0.), ("controller", "lambda_final", .5), ("controller", "valid_blocks", 1), ("controller", "trust_ratio", .3), ("fixed_cutoff", "optimizer_or_controller_reset", True), ("fixed_cutoff", "heldout_input_used", True), ("fixed_cutoff", "controller_diagnostics_computed", False), ("gradient", "clip_scale", 1.), ("rng", "before_block", "wrong"), ("optimizer", "step", 1), ("controller", "lambda_raw", float("nan"))]
        for section, key, value in changes:
            changed = copy.deepcopy(original)
            changed[12][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(RuntimeError):
                verify_scalars(changed, ARMS[1])

    def test_configuration_and_provenance_match(self):
        self.assertEqual(canonical_json_sha256(matched_configuration()), CONFIG_SHA256)
        preflight = {"repository": {"commit": "same"}, "source": {"fingerprint": "same"}, "base_checkpoint": {}}
        split = {"train_manifest_sha256": "same", "eval_manifest_sha256": "same"}
        a, b = [provenance(preflight, split, arm) for arm in ARMS]
        self.assertEqual({k for k in a if a[k] != b[k]}, {"arm", "applied_objective"})

    def test_terminal_precedence(self):
        self.assertEqual(terminal(True, True, False), "PASS_FIXED_CUTOFF")
        self.assertEqual(terminal(False, True, False), "NOT_VALIDATED")
        self.assertEqual(terminal(True, False, False), "INCONCLUSIVE")
        self.assertEqual(terminal(True, True, True), "INCONCLUSIVE")
        self.assertEqual(terminal(False, True, True), "INCONCLUSIVE")

    def test_spatial_benefit_thresholds(self):
        metric = {"auroc": .8966666666666667, "average_precision": .9352228131900764, "spatial": {"effective_support_median": 256, "support_below_128_count": 8, "top11_above_0_35_count": 8}}
        self.assertTrue(endpoint(metric, 1)["self_pass"])
        self.assertFalse(endpoint(metric, 2)["self_pass"])
        metric["spatial"]["support_below_128_count"] = 9
        self.assertFalse(endpoint(metric, 1)["self_pass"])

    def test_prefix_screen_detects_divergence(self):
        self.assertEqual(prefix_distance([1.,2.], [1.,2.]), 0.)
        self.assertGreater(prefix_distance([1.,2.], [2.,4.]), .1)
        with self.assertRaises(RuntimeError):
            prefix_distance([float("nan")], [1.])


if __name__ == "__main__":
    unittest.main()
