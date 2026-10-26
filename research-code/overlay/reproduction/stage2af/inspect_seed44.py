"""No-update diagnosis of the cross-seed frozen-inventory gate failure."""
import argparse
import gzip
import io
import json
from pathlib import Path
import torch
from reproduction.stage2af.runtime import Runtime, write, sha_file
from reproduction.stage2af.frozen import require_frozen
from reproduction.stage2af.guard import InputGuard
from reproduction.stage2l.constants import TRAINABLE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan, protocol = require_frozen(root)
    out = root / 'debug/seed44-frozen-inventory-v1'
    out.mkdir(exist_ok=False)
    runtime = Runtime(root, 44)
    guard = InputGuard(root, [r for r in runtime.rows if r['split'] == 'train']).install()
    hooks = []
    try:
        assert runtime.initial_identity == plan['initial_identities']['44']
        write(out/'initial-state.json', runtime.initial_full)
        actual = {r['name']: r for r in runtime.initial_full['parameters'] if r['name'] not in TRAINABLE_NAMES}
        expected = plan['initial_frozen_parameters']
        assert set(actual) == set(expected)
        differences = {name: dict(seed43=expected[name], seed44=actual[name]) for name in actual if actual[name] != expected[name]}
        write(out/'frozen-differences.json', differences)
        assert differences and all(name.startswith(('model.visual.anomaly_qformer.diff_q_former.', 'model.visual.anomaly_qformer.diff_post_ffn.', 'segmentation_head.')) for name in differences)
        assert runtime.initial_full['buffers'] == plan['initial_buffers']
        def forbidden(*_):
            raise RuntimeError('Inactive diff branch unexpectedly invoked')
        for name in ['diff_q_former', 'diff_post_ffn']:
            hooks.append(getattr(runtime.model.visual.anomaly_qformer, name).register_forward_pre_hook(forbidden))
        lm, lce, images, inputs, rng = runtime.collect(1)
        reference = root/'debug/calibration-s44-v1/block1.json'
        record = json.loads(reference.read_text())
        assert images == record['images']
        raw = record['raw']
        assert sha_file(raw['path']) == raw['sha256']
        with gzip.open(raw['path'], 'rb') as stream:
            original = torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=False)
        compared = 0
        for kind, current in [('lm', lm), ('lce', lce)]:
            for new, old in zip(current, original[kind]):
                assert set(new) == set(old)
                for name in new:
                    assert torch.equal(new[name].cpu(), old[name].float()), (kind, name)
                    compared += 1
        runtime.restore_initial()
        frozen = runtime.verify_frozen()
        write(out/'file-open-audit.json', guard.record())
        write(out/'summary.json', dict(status='PASS', seed=44, actual_updates=0,
              protocol_sha256=protocol, difference_names=sorted(differences),
              original_active_T21_identity_exact=True, registered_buffers_equal_seed43=True,
              all_other_frozen_parameters_equal_seed43=True,
              calibration_first_block_sample_loss_score_exact=True,
              calibration_raw_gradient_tensors_exact=compared,
              forbidden_diff_forward_not_invoked=True, frozen=frozen,
              reference_sha256=sha_file(reference), reference_raw_sha256=raw['sha256']))
        print(json.dumps(dict(status='PASS', difference_names=sorted(differences), gradient_tensors_exact=compared, actual_updates=0)), flush=True)
    finally:
        for hook in hooks: hook.remove()
        runtime.close()


if __name__=='__main__': main()
