from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestGateD3StaticContracts(unittest.TestCase):
    def test_runtime_is_read_only_and_single_forward_source(self):
        source = (REPO_ROOT / "reproduction/stage2i/run_gate_d3.py").read_text(encoding="utf-8")
        for token in (
            ".backward(",
            "torch.optim",
            "Trainer(",
            "DeepSpeed",
            "load_in_4bit",
            "load_in_8bit",
            "device_map=",
            "cpu_offload",
        ):
            self.assertNotIn(token, source)
        self.assertEqual(source.count("outputs = model(**batch)"), 1)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn("retain_graph=mechanism_index < len(MECHANISM_IDS) - 1", source)

    def test_independent_verifier_does_not_import_runner(self):
        source = (REPO_ROOT / "reproduction/stage2i/verify_gate_d3.py").read_text(encoding="utf-8")
        self.assertNotIn("from reproduction.stage2i.run_gate_d3", source)
        self.assertNotIn("import reproduction.stage2i.run_gate_d3", source)

    def test_observed_model_call_order_and_evidence_formula(self):
        source = (
            REPO_ROOT / "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
        ).read_text(encoding="utf-8")
        abnormal = source.index(
            "_, abnormal_map = self.anomaly_attention(abnormal_flat, visual_flat, visual_flat)"
        )
        normal = source.index(
            "_, normal_map = self.anomaly_attention(normal_flat, visual_flat, visual_flat)"
        )
        evidence = source.index("pre_gate_evidence = A_log - N_log")
        self.assertLess(abnormal, normal)
        self.assertLess(normal, evidence)


if __name__ == "__main__":
    unittest.main()
