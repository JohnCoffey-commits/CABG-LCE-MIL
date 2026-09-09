import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reproduction.stage2g.evaluation_core import (
    run_image_evaluation,
    sha256_file,
    write_jsonl,
)


class DryRunTest(unittest.TestCase):
    def test_fixture_has_one_record_per_image_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            manifest_records = []
            for label, value in (("good", 0), ("ungood", 255)):
                path = data / f"brain_mri/test/{label}/{label}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), (value, value, value)).save(path)
                file_hash = sha256_file(path)
                manifest_records.append(
                    {
                        "sample_index": 0,
                        "sample_id": f"fixture-{label}-{file_hash[:12]}",
                        "relative_path": path.relative_to(data).as_posix(),
                        "label": label,
                        "ground_truth": "no" if label == "good" else "yes",
                        "sha256": file_hash,
                        "bytes": path.stat().st_size,
                        "dimensions": [4, 4],
                        "mode": "RGB",
                        "format": "PNG",
                    }
                )
            manifest_records.sort(
                key=lambda record: (record["label"], record["sha256"], record["relative_path"])
            )
            for index, record in enumerate(manifest_records):
                record["sample_index"] = index
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            result = root / "result.json"
            write_jsonl(manifest, manifest_records)
            calls = []

            def generate(_wrapper, image):
                calls.append(image.getpixel((0, 0)))
                return "no" if image.getpixel((0, 0))[0] == 0 else "yes"

            output = run_image_evaluation(
                wrapper=object(),
                data_root=data,
                manifest_path=manifest,
                predictions_path=predictions,
                result_path=result,
                run_metadata={"run_id": "fixture"},
                generation=generate,
                expected_manifest_sha256=sha256_file(manifest),
                expected_records=2,
                expected_counts={"good": 1, "ungood": 1},
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(output["evaluation_records"], 2)
            self.assertEqual(output["metrics"]["balanced_accuracy"], 1.0)
            self.assertEqual(len(predictions.read_text().splitlines()), 2)
            self.assertEqual(json.loads(result.read_text())["run_id"], "fixture")


if __name__ == "__main__":
    unittest.main()
