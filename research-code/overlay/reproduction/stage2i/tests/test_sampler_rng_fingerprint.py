import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reproduction.stage2i.block_sampler import (
    DeterministicBlockSampler,
    _hash_text,
    split_development_pool,
)
from reproduction.stage2i.build_dataset_manifest import build_locked_dataset
from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import (
    dataset_fingerprint,
    implementation_source_record,
    parameter_inventory,
    sha256_file,
)
from reproduction.stage2i.rng_state import (
    restore_rng_state,
    rng_state_fingerprint,
    seed_all,
    snapshot_rng_state,
)


def sample(index: int, label: str) -> dict[str, object]:
    sample_id = f"sample-{label}-{index:03d}"
    return {
        "sample_id": sample_id,
        "sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
        "relative_path": f"brain_mri/test/{label}/{index:03d}.jpg",
        "label": label,
    }


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


class SplitAndSamplerTests(unittest.TestCase):
    def setUp(self):
        self.records = [sample(index, "good") for index in range(56)] + [
            sample(index, "ungood") for index in range(110)
        ]

    def test_locked_split_counts_are_deterministic_and_disjoint(self):
        first = split_development_pool(self.records)
        second = split_development_pool(reversed(self.records))
        self.assertEqual(first["split_fingerprint"], second["split_fingerprint"])
        self.assertEqual(
            first["counts"],
            {
                "source": {"normal": 56, "abnormal": 110},
                "training-development": {"normal": 44, "abnormal": 88},
                "threshold-validation": {"normal": 12, "abnormal": 22},
                "mechanism-diagnostic": {"normal": 6, "abnormal": 6},
            },
        )
        training = {value["sample_id"] for value in first["training-development"]}
        validation = {value["sample_id"] for value in first["threshold-validation"]}
        mechanism = {value["sample_id"] for value in first["mechanism-diagnostic"]}
        self.assertFalse(training & validation)
        self.assertTrue(mechanism.issubset(validation))
        self.assertEqual(len(training | validation), 166)

    def test_protocol_hash_uses_exact_direct_concatenation(self):
        expected = hashlib.sha256("cabg-v1.1-splitabc/path.jpg".encode()).hexdigest()
        self.assertEqual(_hash_text("cabg-v1.1-split", "abc", "/path.jpg"), expected)

    def test_epoch_blocks_use_every_training_image_once(self):
        split = split_development_pool(self.records)
        sampler = DeterministicBlockSampler(split["training-development"], seed=42, epoch=0)
        blocks = sampler.all_blocks()
        self.assertEqual(len(blocks), 44)
        flattened = []
        for block in blocks:
            labels = [value["label"] for value in block]
            self.assertEqual(labels.count("normal"), 1)
            self.assertEqual(labels.count("abnormal"), 2)
            flattened.extend(value["sample_id"] for value in block)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(flattened), 132)
        same = DeterministicBlockSampler(split["training-development"], seed=42, epoch=0)
        different = DeterministicBlockSampler(split["training-development"], seed=123, epoch=0)
        self.assertEqual(sampler.block_permutation_hash, same.block_permutation_hash)
        self.assertNotEqual(sampler.block_permutation_hash, different.block_permutation_hash)

    def test_sampler_state_restores_exact_next_block(self):
        split = split_development_pool(self.records)
        sampler = DeterministicBlockSampler(split["training-development"], seed=42, epoch=1)
        iterator = iter(sampler)
        next(iterator)
        state = sampler.state_dict()
        expected = next(iterator)
        restored = DeterministicBlockSampler(split["training-development"], seed=42, epoch=1)
        restored.load_state_dict(state)
        observed = next(iter(restored))
        self.assertEqual(
            [value["sample_id"] for value in observed],
            [value["sample_id"] for value in expected],
        )


class RNGTests(unittest.TestCase):
    def test_save_restore_is_exact_for_all_available_streams(self):
        seed_all(2026)
        state = snapshot_rng_state()
        fingerprint = rng_state_fingerprint(state)
        first = (random.random(), float(np.random.random()), torch.rand(5))
        restore_rng_state(state)
        self.assertEqual(fingerprint, rng_state_fingerprint(snapshot_rng_state()))
        second = (random.random(), float(np.random.random()), torch.rand(5))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


class FingerprintAndDatasetTests(unittest.TestCase):
    def test_dataset_fingerprint_is_order_invariant_and_content_sensitive(self):
        rows = [sample(0, "good"), sample(1, "ungood")]
        self.assertEqual(dataset_fingerprint(rows), dataset_fingerprint(reversed(rows)))
        changed = [dict(value) for value in rows]
        changed[0]["relative_path"] = "brain_mri/test/good/changed.jpg"
        self.assertNotEqual(dataset_fingerprint(rows), dataset_fingerprint(changed))
        with self.assertRaises(CABGContractError):
            dataset_fingerprint(rows + [rows[0]])

    def test_source_and_parameter_fingerprints_detect_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "b.py").write_text("b = 2\n", encoding="utf-8")
            first = implementation_source_record(root, ["b.py", "a.py"])
            second = implementation_source_record(root, ["a.py", "b.py"])
            self.assertEqual(first, second)
            (root / "a.py").write_text("a = 3\n", encoding="utf-8")
            changed = implementation_source_record(root, ["a.py", "b.py"])
            self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
        parameters = {"b": torch.ones(2), "a": torch.zeros(3)}
        inventory = parameter_inventory(parameters, ["b", "a"])
        self.assertEqual(inventory["tensor_count"], 2)
        self.assertEqual(inventory["element_count"], 5)
        self.assertEqual([value["name"] for value in inventory["parameters"]], ["a", "b"])

    def test_full_dataset_lock_uses_only_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            train_path = root / "train.jsonl"
            internal_path = root / "internal.jsonl"
            audit_path = root / "audit.json"
            source = [sample(index, "good") for index in range(87)] + [
                sample(index, "ungood") for index in range(141)
            ]
            train = source[:8] + source[87:95]
            remaining_normal = source[8:31]
            remaining_abnormal = source[95:118]
            internal = remaining_normal + remaining_abnormal
            write_jsonl(source_path, source)
            write_jsonl(train_path, train)
            write_jsonl(internal_path, internal)
            audit = {
                "status": "SUCCESS",
                "protocol_version": "2.2",
                "internal_test_used_for_selection": False,
                "source_manifest_sha256": sha256_file(source_path),
                "manifest_sha256": {
                    "train": sha256_file(train_path),
                    "internal-test": sha256_file(internal_path),
                },
            }
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            output = root / "locked"
            result = build_locked_dataset(
                source_manifest=source_path,
                stage2h_train_manifest=train_path,
                internal_test_manifest=internal_path,
                stage2h_audit_path=audit_path,
                output_dir=output,
            )
            self.assertEqual(result["counts"]["development-pool"], 166)
            self.assertEqual(result["counts"]["training-development"], 132)
            self.assertEqual(result["counts"]["threshold-validation"], 34)
            self.assertEqual(result["counts"]["mechanism-diagnostic"], 12)
            self.assertEqual(result["image_files_opened"], 0)
            self.assertEqual(result["internal_test_image_files_opened"], 0)


if __name__ == "__main__":
    unittest.main()
