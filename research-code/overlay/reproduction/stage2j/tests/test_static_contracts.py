import ast
import unittest
from pathlib import Path

from reproduction.stage2j.run_redesign_r0 import (
    EXPECTED_D3_EVIDENCE_FINGERPRINT,
    EXPECTED_D3_VERIFICATION_SHA256,
    EXPECTED_MANIFEST_SHA256,
)


ROOT = Path(__file__).resolve().parents[3]


class TestR0StaticContracts(unittest.TestCase):
    def test_locked_fingerprints_are_sha256(self):
        for value in (EXPECTED_D3_EVIDENCE_FINGERPRINT, EXPECTED_D3_VERIFICATION_SHA256, EXPECTED_MANIFEST_SHA256):
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_runner_has_no_training_or_offloading_path(self):
        source = (ROOT / "reproduction/stage2j/run_redesign_r0.py").read_text(encoding="utf-8")
        for prohibited in (".backward(", "torch.optim", "optimizer.step", "load_in_4bit", "load_in_8bit", "cpu_offload", "device_map="):
            self.assertNotIn(prohibited, source)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn("return_anomaly_evidence", source)

    def test_sources_parse(self):
        for relative in (
            "reproduction/stage2j/raw_observer.py",
            "reproduction/stage2j/candidate_math.py",
            "reproduction/stage2j/statistics.py",
            "reproduction/stage2j/run_redesign_r0.py",
            "reproduction/stage2j/verify_redesign_r0.py",
        ):
            ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)

    def test_v2_wrapper_is_fail_closed_and_explicitly_importable(self):
        source = (ROOT / "reproduction/stage2j/run_redesign_r0_v2_l4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune', source)
        self.assertIn("( set -euo pipefail; run_pipeline )", source)
        self.assertIn('TERMINAL_BLOCKER:%s:%s', source)
        self.assertIn('redesign-r0-v2', source)
        self.assertNotIn('redesign-r0-v1"', source)

    def test_v3_wrapper_uses_a_new_immutable_root_and_all_gates(self):
        source = (ROOT / "reproduction/stage2j/run_redesign_r0_v3_l4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('redesign-r0-v3', source)
        self.assertNotIn('redesign-r0-v2"', source)
        self.assertIn('mark_stage real_model_diagnostic', source)
        self.assertIn('mark_stage independent_evaluator', source)
        self.assertIn('mark_stage artifact_inventory', source)
        self.assertIn('--query-compute-apps=pid,process_name,used_memory', source)
        self.assertIn('--post-run-compute-apps', source)
        self.assertIn('( set -euo pipefail; run_pipeline )', source)
        self.assertIn('TERMINAL_BLOCKER:%s:%s', source)

    def test_v4_wrapper_freezes_completion_before_inventory_and_rechecks_after_pipeline(self):
        source = (ROOT / "reproduction/stage2j/run_redesign_r0_v4_l4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('redesign-r0-v4', source)
        self.assertNotIn('redesign-r0-v3"', source)
        completion = source.index("  mark_stage complete\n  find")
        inventory_write = source.index('xargs -0 sha256sum > "$inventory"')
        pipeline_return = source.index('pipeline_status=${PIPESTATUS[0]}')
        post_wrapper_check = source.index(
            'sha256sum -c "$output_dir/artifact-inventory.sha256"'
        )
        self.assertLess(completion, inventory_write)
        self.assertLess(pipeline_return, post_wrapper_check)
        self.assertIn('TERMINAL_BLOCKER:post_wrapper_inventory:%s', source)
        self.assertIn('! -path "$post_wrapper_inventory_check"', source)



if __name__ == "__main__":
    unittest.main()
