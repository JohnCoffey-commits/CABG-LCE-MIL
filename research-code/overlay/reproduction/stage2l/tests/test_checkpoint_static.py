import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import canonical_json_sha256
from reproduction.stage2i.rng_state import rng_state_fingerprint, seed_all, snapshot_rng_state
from reproduction.stage2l.checkpoint import load_strict, save_atomic
from reproduction.stage2l.constants import CHECKPOINT_SCHEMA_VERSION, S9_FINGERPRINT, S9_NAMES, TRAINABLE_NAMES


PROVENANCE = {"source_commit": "0" * 40, "implementation_fingerprint": "1" * 64,
              "base_checkpoint_fingerprint": "2" * 64, "dataset_manifest_sha256": "3" * 64,
              "configuration_sha256": "4" * 64}


def payload():
    adapter = {name: torch.tensor([float(index)], dtype=torch.bfloat16) for index, name in enumerate(TRAINABLE_NAMES)}
    master = {name: value.float() for name, value in adapter.items()}
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_state": adapter, "master_state": master,
        "optimizer_state": {"state": {0: {"step": torch.tensor(2), "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}}, "param_groups": []},
        "scheduler_state": {"last_epoch": 2},
        "controller_state": {"beta": 0.9, "lm_ema": 1.0, "lce_ema": 2.0, "valid_blocks": 2},
        "sampler_state": {"cursor": 2, "manifest_sha256": "5" * 64, "order_sha256": "6" * 64},
        "rng_state": snapshot_rng_state(), "provenance": PROVENANCE,
        "trace_state": {"completed_blocks": 2, "last_record_sha256": "7" * 64},
    }


class CheckpointStaticTests(unittest.TestCase):
    def test_exact_s9_fingerprint(self):
        self.assertEqual(canonical_json_sha256(sorted(S9_NAMES)), S9_FINGERPRINT)

    def test_checkpoint_roundtrip_rng_and_overwrite(self):
        seed_all(42)
        original = payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_atomic(path, original)
            loaded, manifest = load_strict(path, PROVENANCE)
            self.assertEqual(rng_state_fingerprint(loaded["rng_state"]), manifest["rng_fingerprint"])
            for name in TRAINABLE_NAMES:
                self.assertTrue(torch.equal(loaded["master_state"][name], original["master_state"][name]))
            with self.assertRaises(FileExistsError):
                save_atomic(path, original)

    def test_checkpoint_corruption_and_wrong_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_atomic(path, payload())
            with self.assertRaises(CABGContractError):
                load_strict(path, {**PROVENANCE, "configuration_sha256": "f" * 64})
            with path.open("r+b") as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
            with self.assertRaises(CABGContractError):
                load_strict(path, PROVENANCE)

    def test_checkpoint_schema_and_cursor_mismatch_fail_closed(self):
        bad = payload()
        bad["controller_state"]["unexpected"] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CABGContractError):
                save_atomic(Path(directory) / "state.pt", bad)

    def test_fp32_master_adamw_checkpoint_resume_is_exact(self):
        def engine():
            masters = {name: torch.nn.Parameter(torch.tensor([0.01 * (index + 1)], dtype=torch.float32), requires_grad=False)
                       for index, name in enumerate(TRAINABLE_NAMES)}
            optimizer = torch.optim.AdamW(list(masters.values()), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - 0.1 * step)
            return masters, optimizer, scheduler

        def step(masters, optimizer, scheduler, scale):
            optimizer.zero_grad(set_to_none=True)
            for index, name in enumerate(TRAINABLE_NAMES):
                masters[name].grad = torch.tensor([scale * (index + 1)], dtype=torch.float32)
            optimizer.step()
            scheduler.step()

        expected_masters, expected_optimizer, expected_scheduler = engine()
        step(expected_masters, expected_optimizer, expected_scheduler, 0.01)
        step(expected_masters, expected_optimizer, expected_scheduler, -0.02)

        interrupted_masters, interrupted_optimizer, interrupted_scheduler = engine()
        step(interrupted_masters, interrupted_optimizer, interrupted_scheduler, 0.01)
        state = payload()
        state["adapter_state"] = {name: value.detach().to(torch.bfloat16) for name, value in interrupted_masters.items()}
        state["master_state"] = {name: value.detach().clone() for name, value in interrupted_masters.items()}
        state["optimizer_state"] = interrupted_optimizer.state_dict()
        state["scheduler_state"] = interrupted_scheduler.state_dict()
        state["controller_state"]["valid_blocks"] = 1
        state["sampler_state"]["cursor"] = 1
        state["trace_state"]["completed_blocks"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_atomic(path, state)
            loaded, _ = load_strict(path, PROVENANCE)
        resumed_masters, resumed_optimizer, resumed_scheduler = engine()
        for name in TRAINABLE_NAMES:
            resumed_masters[name].data.copy_(loaded["master_state"][name])
        resumed_optimizer.load_state_dict(loaded["optimizer_state"])
        resumed_scheduler.load_state_dict(loaded["scheduler_state"])
        step(resumed_masters, resumed_optimizer, resumed_scheduler, -0.02)
        for name in TRAINABLE_NAMES:
            self.assertTrue(torch.equal(expected_masters[name], resumed_masters[name]), name)
        bad = payload()
        bad["sampler_state"]["cursor"] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CABGContractError):
                save_atomic(Path(directory) / "state.pt", bad)

    def test_independent_evaluator_does_not_import_trainer_controller(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / "reproduction/stage2l/verify_pilot.py").read_text()
        self.assertNotIn("from reproduction.stage2l.controller", source)
        self.assertNotIn("from reproduction.stage2l.run_pilot", source)

    def test_runtime_uses_two_isolated_autograd_grad_calls_and_no_deepspeed(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / "reproduction/stage2l/run_pilot.py").read_text()
        self.assertEqual(source.count("torch.autograd.grad("), 2)
        self.assertNotIn("deepspeed", source.lower())
        self.assertIn("optimizer.step()", source)
        self.assertIn("scheduler.step()", source)
        self.assertIn("any(parameter.grad is not None", source)
        self.assertNotIn('markers.mark("monitor_started", phase=', source)
        self.assertNotIn('markers.mark("phase_completed", phase=', source)

    def test_dataset_lock_explicitly_subtracts_d3r_and_declares_boundaries(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / "reproduction/stage2l/build_dataset.py").read_text()
        self.assertIn("d3r_ids", source)
        self.assertIn("d3r_hashes", source)
        self.assertIn('"protected_internal_test_image_files_opened": 0', source)
        self.assertIn('"effectiveness_evaluation_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
