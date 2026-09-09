import unittest

from reproduction.stage2g.evaluation_core import compute_metrics, normalize_binary_response


class EvaluationCoreTest(unittest.TestCase):
    def test_normalization_is_strict(self):
        self.assertEqual(normalize_binary_response(" Yes. \n"), ("yes", "yes"))
        self.assertEqual(normalize_binary_response("NO？"), ("no", "no"))
        self.assertEqual(normalize_binary_response("yes.."), ("yes.", None))
        self.assertEqual(normalize_binary_response("Answer: yes"), ("answer: yes", None))
        self.assertEqual(normalize_binary_response(""), ("", None))

    def test_invalid_outputs_are_in_denominators(self):
        records = [
            {"ground_truth": "yes", "prediction": "yes", "normalized_response": "yes"},
            {"ground_truth": "yes", "prediction": None, "normalized_response": "maybe"},
            {"ground_truth": "no", "prediction": "no", "normalized_response": "no"},
            {"ground_truth": "no", "prediction": None, "normalized_response": ""},
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["invalid_responses"], 2)
        self.assertEqual(metrics["empty_responses"], 1)
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
