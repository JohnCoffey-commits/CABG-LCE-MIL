"""Exercise scientific-claim rejection against the retained metadata capture."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.stage2ab.verify import verify


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(__file__).resolve().parents[3]
        cls.evidence = cls.workspace / 'document/cabg_mil_v1_2/capability-readiness'
        cls.report = json.loads((cls.evidence / 'readiness-report-v1.json').read_text())
        # Retained report paths are relative to the code repository.
        for row in cls.report['historical_development_rankings']:
            row['source'] = str((cls.workspace / 'Medic-AD' / row['source']).resolve())

    def reject(self, mutate):
        report = copy.deepcopy(self.report)
        mutate(report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps(report))
            with self.assertRaises(AssertionError):
                verify(self.workspace, self.evidence / 'remote-snapshot-v1.stdout', path)

    def test_patient_independence_cannot_be_promoted(self):
        self.reject(lambda r: r.update(patient_independence_verified=True))

    def test_unknown_use_history_cannot_be_promoted(self):
        self.reject(lambda r: r['br35h'].update(prior_use_history='CLOSED_UNUSED'))

    def test_unperformed_answer_evaluation_cannot_be_claimed(self):
        self.reject(lambda r: r.update(new_answer_effect='NO_BENEFIT'))

    def test_metadata_capture_cannot_claim_image_execution(self):
        self.reject(lambda r: r.update(images_opened=1))


if __name__ == '__main__':
    unittest.main()
