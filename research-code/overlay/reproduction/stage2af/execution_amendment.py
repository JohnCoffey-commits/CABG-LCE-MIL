"""Sealed engineering correction: match frozen initial state within each seed."""
import json
from pathlib import Path
from reproduction.stage2af.frozen import require_frozen, sha


def load_execution_amendment(root):
    root = Path(root)
    plan, protocol = require_frozen(root)
    path = root/'engineering-amendment-02.json'
    amendment = json.loads(path.read_text())
    identity = sha(path)
    assert json.loads((root/'engineering-amendment-02-seal.json').read_text())['sha256'] == identity
    assert amendment['original_protocol_sha256'] == protocol
    assert amendment['status'] == 'SEALED_ENGINEERING_CORRECTION'
    for filename, expected in amendment['source_identities'].items():
        assert sha(filename) == expected, filename
    for filename, expected in amendment['evidence_identities'].items():
        assert sha(filename) == expected, filename
    assert amendment['actual_updates_in_failed_attempt'] == 0
    assert amendment['no_update_diagnosis']['gradient_tensors_exact'] == 90
    return plan, protocol, amendment, identity


def expected_frozen_parameters(plan, amendment, seed):
    expected = dict(plan['initial_frozen_parameters'])
    if seed == 44:
        overrides = amendment['seed44_frozen_overrides']
        assert len(overrides) == 17
        for name, record in overrides.items():
            assert name in expected and record['name'] == name
            assert name.startswith(('model.visual.anomaly_qformer.diff_q_former.',
                                    'model.visual.anomaly_qformer.diff_post_ffn.',
                                    'segmentation_head.'))
            assert record['shape'] == expected[name]['shape']
            assert record['dtype'] == expected[name]['dtype']
            expected[name] = record
    else:
        assert seed == 43
    return expected
