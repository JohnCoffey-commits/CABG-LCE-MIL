import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reproduction.stage2g.build_brainmri_manifest import build_records, deduplicate


class ManifestBuilderTest(unittest.TestCase):
    def _image(self, path: Path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 5), color=color).save(path)

    def test_deduplicates_within_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._image(root / "brain_mri/test/good/a.png", (0, 0, 0))
            (root / "brain_mri/test/good/b.png").write_bytes(
                (root / "brain_mri/test/good/a.png").read_bytes()
            )
            self._image(root / "brain_mri/test/ungood/c.png", (255, 255, 255))
            candidates, errors = build_records(root)
            self.assertFalse(errors)
            enriched, evaluation, duplicate_groups = deduplicate(candidates)
            self.assertEqual(len(enriched), 3)
            self.assertEqual(len(evaluation), 2)
            self.assertEqual(len(duplicate_groups), 1)
            self.assertEqual(duplicate_groups[0]["canonical_relative_path"], "brain_mri/test/good/a.png")

    def test_rejects_cross_label_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._image(root / "brain_mri/test/good/a.png", (0, 0, 0))
            target = root / "brain_mri/test/ungood/a.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / "brain_mri/test/good/a.png").read_bytes())
            candidates, _ = build_records(root)
            with self.assertRaises(RuntimeError):
                deduplicate(candidates)


if __name__ == "__main__":
    unittest.main()
