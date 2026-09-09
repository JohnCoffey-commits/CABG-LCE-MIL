from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduction.stage2i import build_gate_d3_dataset
from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import sha256_file


class TestGateD3Dataset(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        image_root = root / "images"
        image_root.mkdir()
        rows = []
        for index in range(12):
            relative = Path("abnormal" if index < 6 else "normal") / f"image-{index:02d}.png"
            path = image_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"locked-image-{index}".encode("utf-8")
            path.write_bytes(payload)
            rows.append(
                {
                    "sample_id": f"sample-{index:02d}",
                    "sha256": sha256_file(path),
                    "relative_path": relative.as_posix(),
                    "label": "abnormal" if index < 6 else "normal",
                    "bytes": len(payload),
                    "cabg_role": "mechanism-diagnostic",
                }
            )
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        return manifest, image_root

    def test_locked_annotation_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, image_root = self._fixture(root)
            output = root / "locked"
            with mock.patch.object(
                build_gate_d3_dataset, "EXPECTED_MANIFEST_SHA256", sha256_file(manifest)
            ):
                result = build_gate_d3_dataset.build_dataset(manifest, image_root, output)
                self.assertEqual(result["count"], 12)
                self.assertEqual(result["class_counts"], {"abnormal": 6, "normal": 6})
                annotations = json.loads((output / "mechanism-diagnostic.json").read_text())
                self.assertEqual([row["anomaly_label"] for row in annotations], [1] * 6 + [0] * 6)
                self.assertEqual(annotations[0]["conversations"][1]["value"], "Yes")
                self.assertEqual(annotations[-1]["conversations"][1]["value"], "No")
                with self.assertRaises(FileExistsError):
                    build_gate_d3_dataset.build_dataset(manifest, image_root, output)

    def test_manifest_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, image_root = self._fixture(root)
            with self.assertRaisesRegex(CABGContractError, "SHA-256"):
                build_gate_d3_dataset.build_dataset(manifest, image_root, root / "locked")


if __name__ == "__main__":
    unittest.main()
