import unittest

from reproduction.stage2g.aggregate_pilot import classify_decision


def pairs(deltas, a3_invalid=0, b0_invalid=0):
    return [
        {
            "pair_id": f"pair-{index}",
            "balanced_accuracy_delta_a3_minus_b0": delta,
            "a3_invalid_responses": a3_invalid,
            "b0_invalid_responses": b0_invalid,
        }
        for index, delta in enumerate(deltas)
    ]


class DecisionRuleTest(unittest.TestCase):
    def test_proceed(self):
        result = classify_decision(pairs([0.03, 0.04, 0.01, 0.03, 0.04, -0.01]))
        self.assertEqual(result["decision"], "PROCEED")

    def test_pause_for_negative_signal(self):
        result = classify_decision(pairs([-0.03, -0.04, -0.01, -0.03, 0.0, 0.0]))
        self.assertEqual(result["decision"], "PAUSE")

    def test_pause_for_any_invalid_regression(self):
        result = classify_decision(pairs([0.04] * 6, a3_invalid=1, b0_invalid=0))
        self.assertEqual(result["decision"], "PAUSE")

    def test_inconclusive(self):
        result = classify_decision(pairs([0.01, 0.01, 0.0, -0.01, 0.0, 0.01]))
        self.assertEqual(result["decision"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
