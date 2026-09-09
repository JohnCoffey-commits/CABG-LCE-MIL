import importlib
import json
import unittest
from pathlib import Path

import torch

from reproduction.stage2i.constants import (
    EXPECTED_SHARED_ELEMENTS,
    EXPECTED_SHARED_TENSORS,
    EXPECTED_TRAINABLE_ELEMENTS,
    EXPECTED_TRAINABLE_TENSORS,
    SHARED_SUPPORT_NAMES,
)


class StaticContractTests(unittest.TestCase):
    def test_real_adapter_fixture_matches_locked_scope(self):
        repo_root = Path(__file__).resolve().parents[3]
        fixture = repo_root / (
            "reproduction/stage2f/fixtures/pre-change/"
            "stage2e-seed42-adapter-manifest.json"
        )
        manifest = json.loads(fixture.read_text(encoding="utf-8"))
        parameters = {value["name"]: value for value in manifest["parameters"]}
        self.assertEqual(len(parameters), EXPECTED_TRAINABLE_TENSORS)
        self.assertEqual(
            sum(int(value["elements"]) for value in parameters.values()),
            EXPECTED_TRAINABLE_ELEMENTS,
        )
        self.assertEqual(len(SHARED_SUPPORT_NAMES), EXPECTED_SHARED_TENSORS)
        self.assertEqual(
            sum(int(parameters[name]["elements"]) for name in SHARED_SUPPORT_NAMES),
            EXPECTED_SHARED_ELEMENTS,
        )

    def test_gate_d2_import_does_not_initialize_cuda(self):
        initialized_before = torch.cuda.is_initialized()
        importlib.import_module("reproduction.stage2i")
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)

    def test_gate_d2_has_no_deepspeed_or_quantization_path(self):
        repo_root = Path(__file__).resolve().parents[3]
        source_root = repo_root / "reproduction/stage2i"
        forbidden = ("import deepspeed", "from deepspeed", "bitsandbytes", "cpu_offload")
        for path in source_root.glob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} found in {path}")


if __name__ == "__main__":
    unittest.main()
