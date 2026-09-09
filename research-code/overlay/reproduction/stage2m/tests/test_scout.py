from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from reproduction.stage2m.build_split import select_split
from reproduction.stage2m.metrics import average_precision, auroc, paired_stratified_bootstrap, summarize_metrics
from reproduction.stage2m.train_scout import cosine_multiplier, matched_configuration, provenance
from reproduction.stage2m.training_math import compute_lm_only_block
from reproduction.stage2m.summarize_scout import classify_decision


def _row(label: str, index: int) -> dict[str, object]:
    prefix = "n" if label == "normal" else "a"
    return {
        "sample_id": f"{prefix}-{index:03d}",
        "sha256": f"{index + (0 if label == 'normal' else 1000):064x}",
        "relative_path": f"{label}/{index:03d}.jpg",
        "label": label,
        "bytes": 10,
    }


class ScoutSplitTests(unittest.TestCase):
    def test_split_counts_order_and_exclusions(self):
        normal = [_row("normal", index) for index in range(44)]
        abnormal = [_row("abnormal", index) for index in range(88)]
        d3r = [*normal[:12], *abnormal[:12]]
        pilot = [*normal[12:16], *abnormal[12:20]]
        train, evaluation, unique_train = select_split([*normal, *abnormal], d3r, pilot)
        self.assertEqual((len(train), len(evaluation), len(unique_train)), (72, 32, 64))
        self.assertEqual(sum(row["scout_label"] == "normal" for row in train), 24)
        self.assertEqual(sum(row["scout_label"] == "abnormal" for row in train), 48)
        self.assertEqual(len({row["sample_id"] for row in unique_train if row["scout_label"] == "normal"}), 16)
        self.assertEqual([row["scout_block_id"] for row in train[::3]], list(range(24)))
        excluded = {row["sample_id"] for row in [*d3r, *pilot]}
        self.assertFalse(excluded & {row["sample_id"] for row in [*train, *evaluation]})
        self.assertFalse({row["sample_id"] for row in train} & {row["sample_id"] for row in evaluation})


class ScoutMetricTests(unittest.TestCase):
    def test_perfect_and_reversed_metrics(self):
        perfect = [
            {"sample_id": "n1", "label": "normal", "score": -2.0},
            {"sample_id": "n2", "label": "normal", "score": -1.0},
            {"sample_id": "a1", "label": "abnormal", "score": 1.0},
            {"sample_id": "a2", "label": "abnormal", "score": 2.0},
        ]
        reversed_rows = [{**row, "score": -float(row["score"])} for row in perfect]
        self.assertEqual(auroc(perfect), 1.0)
        self.assertEqual(average_precision(perfect), 1.0)
        self.assertEqual(auroc(reversed_rows), 0.0)
        self.assertAlmostEqual(average_precision(reversed_rows), (1 / 3 + 2 / 4) / 2)

    def test_tied_auroc_and_summary(self):
        rows = [
            {"sample_id": "n", "label": "normal", "score": 0.0},
            {"sample_id": "a", "label": "abnormal", "score": 0.0},
        ]
        result = summarize_metrics(rows)
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["average_precision"], 0.5)

    def test_paired_bootstrap_is_deterministic_and_identity_strict(self):
        baseline = [
            {"sample_id": "n", "label": "normal", "score": 0.0},
            {"sample_id": "a", "label": "abnormal", "score": 1.0},
        ]
        cabg = [{**row, "score": float(row["score"]) + (1.0 if row["label"] == "abnormal" else -1.0)} for row in baseline]
        first = paired_stratified_bootstrap(cabg, baseline, replicates=10, seed=7)
        second = paired_stratified_bootstrap(cabg, baseline, replicates=10, seed=7)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            paired_stratified_bootstrap(cabg[:-1], baseline, replicates=10, seed=7)


class ScoutTrainingMathTests(unittest.TestCase):
    def test_lm_only_is_exact_mean_then_global_clip(self):
        names, shared = ("p", "q"), ("p",)
        lm = [
            {"p": torch.tensor([3.0]), "q": torch.tensor([4.0])},
            {"p": torch.tensor([6.0]), "q": torch.tensor([8.0])},
            {"p": torch.tensor([0.0]), "q": torch.tensor([5.0])},
        ]
        lce = [{"p": torch.tensor([value])} for value in (1.0, 2.0, 3.0)]
        applied, details = compute_lm_only_block(lm, lce, ["normal", "abnormal", "abnormal"], trainable_names=names, s9_names=shared)
        raw = {name: torch.stack([row[name] for row in lm]).mean(0) for name in names}
        norm = torch.sqrt(sum(value.square().sum() for value in raw.values()))
        scale = min(1.0, 1.0 / float(norm))
        for name in names:
            torch.testing.assert_close(applied[name], raw[name] * scale)
        self.assertEqual(details["budget"]["lambda_final"], 0.0)
        self.assertEqual(details["gradient"]["scaled_auxiliary_shared_norm"], 0.0)

    def test_lm_only_rejects_wrong_lce_support(self):
        with self.assertRaises(RuntimeError):
            compute_lm_only_block(
                [{"p": torch.ones(1)}, {"p": torch.ones(1)}],
                [{"wrong": torch.ones(1)}, {"wrong": torch.ones(1)}],
                ["normal", "abnormal"],
                trainable_names=("p",),
                s9_names=("p",),
            )

    def test_scheduler_and_matched_configuration(self):
        self.assertEqual(cosine_multiplier(0), 1.0)
        self.assertAlmostEqual(cosine_multiplier(12), 0.5)
        self.assertAlmostEqual(cosine_multiplier(24), 0.0)
        self.assertEqual(matched_configuration()["blocks"], 24)
        self.assertTrue(matched_configuration()["diagnostic_lce_gradient_both_arms"])

    def test_provenance_differs_only_in_registered_arm_fields(self):
        preflight = {
            "repository": {"commit": "c"},
            "source": {"fingerprint": "s"},
            "base_checkpoint": {"fingerprint": "b"},
        }
        split = {"train_manifest_sha256": "t", "eval_manifest_sha256": "e"}
        cabg = provenance(preflight, split, "cabg_lce")
        baseline = provenance(preflight, split, "lm_only")
        differing = {key for key in cabg if cabg[key] != baseline[key]}
        self.assertEqual(differing, {"arm", "applied_objective"})

    def test_wrapper_uses_fresh_scout_roots_and_no_internal_test(self):
        wrapper = Path(__file__).parents[1] / "run_d4_scout_v1_l4.sh"
        text = wrapper.read_text()
        self.assertIn("d4-scout-v1", text)
        self.assertIn("set -euo pipefail", text)
        self.assertNotIn("internal-test", text.lower())

    def test_midpoint_evaluator_cursor_is_explicit(self):
        source = (Path(__file__).parents[1] / "evaluate_scout.py").read_text()
        self.assertIn("args.expected_cursor not in (12, 24)", source)
        self.assertIn('"checkpoint_cursor"', source)

    def test_registered_abcd_decision_rule(self):
        common = dict(initial_auroc=0.70, initial_ap=0.70)
        self.assertEqual(classify_decision(cabg_auroc=0.74, cabg_ap=0.72, baseline_auroc=0.70, baseline_ap=0.71, stable=True, **common)[0], "A")
        self.assertEqual(classify_decision(cabg_auroc=0.71, cabg_ap=0.69, baseline_auroc=0.70, baseline_ap=0.70, stable=True, **common)[0], "B")
        self.assertEqual(classify_decision(cabg_auroc=0.69, cabg_ap=0.70, baseline_auroc=0.70, baseline_ap=0.70, stable=True, **common)[0], "C")
        self.assertEqual(classify_decision(cabg_auroc=0.80, cabg_ap=0.80, baseline_auroc=0.60, baseline_ap=0.60, stable=False, **common)[0], "D")


if __name__ == "__main__":
    unittest.main()
