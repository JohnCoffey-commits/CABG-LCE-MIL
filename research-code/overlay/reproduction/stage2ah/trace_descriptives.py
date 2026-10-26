"""Post-hoc descriptive arithmetic on existing development traces; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    x = np.asarray(values, dtype=float)
    assert np.isfinite(x).all()
    return dict(n=len(x), minimum=float(x.min()), median=float(np.median(x)),
                mean=float(x.mean()), maximum=float(x.max()))


def relative_change(base, changed):
    base = np.asarray(base, dtype=float)
    changed = np.asarray(changed, dtype=float)
    return float(np.linalg.norm(changed - base) / max(np.linalg.norm(base), 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    assert Path('/home/Medic-AD').is_dir(), 'Raw traces remain on the authorized server'
    assert read(root/'results-v1/independent-verification.json')['status'] == 'PASS'
    report = dict(
        scope='Post-hoc descriptive magnitudes only; existing development traces; no new forward, update, endpoint, selection or final-test inspection',
        sample_limit='Full traces are the first eight manifest cases per seed, not a random or representative sample. Do not generalize their magnitudes to all 128 cases.',
        per_seed={}, sources=[])
    for seed in [43, 44]:
        name = f's{seed}-aligned-step1000-development'
        native_dir = root/'evaluation'/name/'v1'
        shuffle_dir = root/'evaluation'/(name+'-inference_shuffle')/'v1'
        native = read(native_dir/'observations.json')
        shuffled = read(shuffle_dir/'observations.json')
        assert [r['sample_id'] for r in native] == [r['sample_id'] for r in shuffled]
        cases = []
        for i, (left, right) in enumerate(zip(native, shuffled)):
            if not left['full_transport_trace']:
                continue
            assert right['full_transport_trace']
            paths = [native_dir/left['raw_file'], shuffle_dir/right['raw_file']]
            for path, row in zip(paths, [left, right]):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest == row['raw_sha256']
                report['sources'].append(dict(path=str(path), sha256=digest))
            with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
                assert np.array_equal(a['direct'], b['direct'])
                assert np.array_equal(a['native'], b['native'])
                assert np.array_equal(a['values'], b['values'])
                cases.append(dict(
                    manifest_index=i, sample_id=left['sample_id'], target=left['target'],
                    weight_l1=float(np.abs(a['weight16'].astype(float)-b['weight16'].astype(float)).sum()),
                    pooled_relative_change=relative_change(a['pooled'], b['pooled']),
                    mixed_relative_change=relative_change(a['output'], b['output']),
                    fed_ano_relative_change=relative_change(a['fed_ano'], b['fed_ano']),
                    pooled_to_native_token_rms=float(np.sqrt(np.mean(a['pooled'].astype(float)**2))/max(np.sqrt(np.mean(a['native'].astype(float)**2)), 1e-30)),
                    generated_answer_changed=left['parsed'] != right['parsed']))
        assert len(cases) == 8
        fields = ['weight_l1', 'pooled_relative_change', 'mixed_relative_change',
                  'fed_ano_relative_change', 'pooled_to_native_token_rms']
        gaps = [b['yes_no_logit_gap']-a['yes_no_logit_gap'] for a, b in zip(native, shuffled)]
        report['per_seed'][str(seed)] = dict(
            full_trace_cases=cases,
            first_eight_descriptives={key: distribution([r[key] for r in cases]) for key in fields},
            all_128_logit_gap_change=distribution(gaps),
            all_128_nonzero_logit_gap_changes=sum(v != 0 for v in gaps),
            all_128_generated_answer_changes=sum(a['parsed'] != b['parsed'] for a, b in zip(native, shuffled)))
    path = root/'results-v1/trace-descriptives-posthoc.json'
    with path.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: {key: value for key, value in v.items() if key != 'full_trace_cases'}
                      for k, v in report['per_seed'].items()}), flush=True)


if __name__ == '__main__':
    main()
