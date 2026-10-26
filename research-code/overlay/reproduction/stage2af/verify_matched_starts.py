"""Compare retained initial inventories, sequences and first raw gradients."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import torch


def read(path):
    return json.loads(Path(path).read_text())


def raw(record):
    path = Path(record['raw']['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    assert digest.hexdigest() == record['raw']['sha256']
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[43, 44], required=True)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(4)
    plan = read(root / 'protocol.json')
    attempts = read(root / 'accepted-training-attempts.json')
    inventories, sequences, records = {}, {}, {}
    for arm in plan['arms']:
        name = f's{args.seed}-{arm}'
        folder = root / 'formal' / name / attempts[name]
        inventories[arm] = read(folder / 'initial-state.json')
        sequences[arm] = read(folder / 'sequence.json')
        records[arm] = read(folder / 'step0001.json')
    reference = 'lm_only'
    assert len(sequences[reference]) == 3000
    anchor = raw(records[reference])
    compared = 0
    for arm in ['fixed_lce', 'cabg_lce']:
        assert inventories[arm] == inventories[reference]
        assert sequences[arm] == sequences[reference]
        for field in ['images', 'rng', 'lm_rms', 'lce_rms', 'controller', 'lr']:
            assert records[arm][field] == records[reference][field], (arm, field)
        candidate = raw(records[arm])
        for kind, expected_parameters in [('lm', 21), ('lce', 9)]:
            assert len(anchor[kind]) == len(candidate[kind]) == 3
            for left, right in zip(anchor[kind], candidate[kind]):
                assert set(left) == set(right) and len(left) == expected_parameters
                for name in left:
                    assert torch.equal(left[name], right[name]), (arm, kind, name)
                    compared += 1
        del candidate
        gc.collect()
    assert compared == 180
    assert records['lm_only']['applied_coefficient'] == 0
    assert records['fixed_lce']['applied_coefficient'] == plan['fixed_coefficients'][str(args.seed)]
    assert records['cabg_lce']['applied_coefficient'] == records['cabg_lce']['controller']['lambda_final']
    assert [records[arm]['applied_dynamic_cap'] for arm in plan['arms']] == [False, False, True]
    result = dict(status='PASS', seed=args.seed, identical_full_initial_inventories=True,
                  identical_planned_exposures=3000, identical_first_block_observations=True,
                  exact_raw_gradient_tensor_pairs=compared,
                  applied_coefficients={arm: records[arm]['applied_coefficient'] for arm in plan['arms']},
                  raw_sources={arm: records[arm]['raw'] for arm in plan['arms']},
                  scope='Within-seed matched starts and first pre-update gradients. Subsequent updates intentionally use distinct objectives; not a full trajectory replay.')
    with (root / f'matched-raw-starts-seed{args.seed}-v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'raw_sources'}), flush=True)


if __name__ == '__main__':
    main()
