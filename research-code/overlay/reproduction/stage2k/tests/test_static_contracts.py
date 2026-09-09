from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class TestD3RStaticContracts(unittest.TestCase):
    def test_all_sources_parse(self):
        for path in sorted((ROOT / "reproduction/stage2k").glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_producer_is_diagnostic_only(self):
        source = (ROOT / "reproduction/stage2k/run_d3r.py").read_text(encoding="utf-8")
        for prohibited in (
            ".backward(",
            "torch.optim",
            "optimizer.step",
            "scheduler.step",
            "save_pretrained",
            "torch.save",
            "load_in_4bit",
            "load_in_8bit",
            "cpu_offload",
            "device_map=",
        ):
            self.assertNotIn(prohibited, source)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn("return_anomaly_evidence", source)

    def test_evaluator_does_not_import_producer_or_candidate_math(self):
        source = (ROOT / "reproduction/stage2k/verify_d3r.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name.startswith("reproduction.stage2k") for name in imports))
        self.assertFalse(any("candidate_math" in name or "producer_math" in name or "run_d3r" in name for name in imports))
        self.assertIn("abnormal.float() - normal.float()", source)

    def test_wrapper_is_fail_closed_and_post_wrapper_rechecks(self):
        source = (ROOT / "reproduction/stage2k/run_d3r_v1_l4.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", source)
        self.assertIn("( set -euo pipefail; run_pipeline )", source)
        self.assertIn("pipeline_status=${PIPESTATUS[0]}", source)
        self.assertIn("mark_stage complete", source)
        self.assertIn('sha256sum -c "$output_dir/artifact-inventory.sha256"', source)
        self.assertIn("INVALID_QUARANTINED", source)
        self.assertNotIn("optimizer", source)
        self.assertNotIn("internal-test.json", source)


if __name__ == "__main__":
    unittest.main()
