from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.stage2k import build_dataset
from reproduction.stage2k.manifest import intersections, selection_key


class TestD3RManifestAndDataset(unittest.TestCase):
    def test_selection_key_is_content_path_label_bound(self):
        row = {"label": "abnormal", "sha256": "a" * 64, "relative_path": "a/b.jpg", "sample_id": "x"}
        first = selection_key(row)
        self.assertEqual(first, selection_key(dict(row)))
        changed = dict(row, relative_path="a/c.jpg")
        self.assertNotEqual(first, selection_key(changed))

    def test_intersections_cover_three_identities(self):
        row = {"sha256": "a", "relative_path": "b", "sample_id": "c"}
        self.assertEqual(intersections([row], [dict(row)]), {"sha256": 1, "relative_path": 1, "sample_id": 1})

    def test_runtime_dataset_verifies_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            rows = []
            for index in range(24):
                relative = Path("normal" if index >= 12 else "abnormal") / f"{index}.jpg"
                path = images / relative
                path.parent.mkdir(exist_ok=True)
                payload = f"real-image-byte-fixture-{index}".encode()
                path.write_bytes(payload)
                rows.append(
                    {
                        "sample_id": f"s{index}",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "relative_path": str(relative),
                        "bytes": len(payload),
                        "d3r_label": "abnormal" if index < 12 else "normal",
                        "d3r_selection_key": f"k{index}",
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with patch.object(build_dataset, "EXPECTED_MANIFEST_SHA256", build_dataset.sha256_file(manifest)):
                result = build_dataset.build(manifest, images, root / "locked")
                self.assertEqual(result["image_files_verified"], 24)
                self.assertEqual(result["protected_internal_test_image_files_opened"], 0)
                with self.assertRaises(FileExistsError):
                    build_dataset.build(manifest, images, root / "locked")


if __name__ == "__main__":
    unittest.main()
