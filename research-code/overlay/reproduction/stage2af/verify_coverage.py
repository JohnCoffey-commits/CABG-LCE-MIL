"""Independent delivery coverage, CSV consistency and gate chronology checks.

This complements raw numerical verification; it never changes accepted results.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_cell(value):
    return '' if value is None else str(value)


def verify_csv(path, expected):
    with Path(path).open(newline='') as f:
        actual = list(csv.DictReader(f))
    assert len(actual) == len(expected), (path, len(actual), len(expected))
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert set(a) == set(e), (path, i, set(a) ^ set(e))
        assert a == {k: csv_cell(v) for k, v in e.items()}, (path, i)


def audit_opened(path, count):
    audit = read(path)
    assert audit['violations'] == []
    assert audit['allowed_cache_files'] == audit['opened_unique_cache_files'] == count
    assert audit['historical_image_files_opened'] == audit['raw_images_opened_by_model'] == 0


def audit_classes_and_spatial(rows, report):
    import numpy as np
    for target in (0, 1):
        selected = [r for r in rows if r['target'] == target]
        correct = sum(r['parsed'] is not None and r['parsed'] == target for r in selected)
        expected = dict(n=len(selected), correct=correct, correct_rate=correct / len(selected),
                        invalid=sum(r['parsed'] is None for r in selected))
        assert report['by_class'][str(target)] == expected
    for key in ('effective_support', 'top11_mass', 'yes_no_logit_gap'):
        if key not in rows[0]:
            continue
        values = [r[key] for r in rows]
        assert all(math.isfinite(v) for v in values)
        expected = dict(mean=sum(values) / len(values), median=float(np.median(values)),
                        min=min(values), max=max(values))
        for statistic, value in expected.items():
            assert math.isclose(report[key][statistic], value, rel_tol=1e-12, abs_tol=1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root
    plan = read(root / 'protocol.json')
    protocol = sha(root / 'protocol.json')
    assert read(root / 'protocol-seal.json')['sha256'] == protocol
    for field in ('sources', 'input_files'):
        for path, identity in plan[field].items():
            assert sha(path) == identity, (field, path)
    for path, identity in plan['base_file_identities'].items():
        assert Path(path).stat().st_size == identity['bytes']
        assert sha(path) == identity['sha256'], ('base model/tokenizer/config changed', path)
    terminal = read(root / 'training-terminal-status-v1.json')
    expected_trajectories = {f's{s}-{arm}' for s in plan['seeds'] for arm in plan['arms']}
    assert set(terminal) == expected_trajectories
    attempts = read(root/'accepted-training-attempts.json')
    amendment = read(root/'engineering-amendment-02.json')
    assert sha(root/'engineering-amendment-02.json') == read(root/'engineering-amendment-02-seal.json')['sha256']
    assert attempts == amendment['accepted_training_attempts']
    assert amendment['original_protocol_sha256'] == protocol
    for field in ('source_identities', 'evidence_identities'):
        for path, identity in amendment[field].items(): assert sha(path) == identity
    failed = read(root/'formal/s44-lm_only/attempt-v1/failure.json')
    assert failed['last_completed_step'] == 0 and failed['last_checkpoint'] is None
    assert not list((root/'formal/s44-lm_only/attempt-v1').glob('step????.json'))
    accepted = read(root / 'accepted-endpoints.json')
    expected_endpoints = {'linear-head-fit', 'linear-head-final'}
    init_by_seed = {}
    sequence_by_seed = {}
    train_ends = []
    retained_checkpoints = 0
    for name, status in terminal.items():
        seed = int(name[1:3])
        d = root / 'formal' / name / attempts[name]
        suffix = attempts[name].removeprefix('attempt-')
        process = read(root / 'control' / ('formal-' + name + '-'+suffix) / 'process.json')
        exit_record = read(root / 'control' / ('formal-' + name + '-'+suffix) / 'exit.json')
        assert process['start_epoch'] > plan['frozen_epoch']
        train_ends.append(exit_record['end_epoch'])
        initial = read(d / 'initial-state.json')
        if seed == 44:
            binding = read(d/'execution-amendment.json')
            assert binding['sha256'] == sha(root/'engineering-amendment-02.json')
            assert process['start_epoch'] > amendment['sealed_epoch']
        sequence = read(d / 'sequence.json')
        assert initial == init_by_seed.setdefault(seed, initial), ('different initial state', name)
        assert sequence == sequence_by_seed.setdefault(seed, sequence), ('different sequence', name)
        assert len(sequence) == 3000
        if status['status'] == 'COMPLETE':
            assert status == read(d / 'summary.json') and exit_record['exit_code'] == 0
            last = status['last_step']
            assert last == 1000
            audit_opened(d / 'file-open-audit.json', 1000)
            expected_endpoints.add(name + '-step1000-final')
        else:
            assert status['status'] == 'NUMERICAL_FAILURE'
            assert status == read(d / 'failure.json') and exit_record['exit_code'] != 0
            last = status['last_completed_step']
        expected_steps = {step for step in plan['checkpoint_steps'] if step <= last}
        cp_steps = {int(f.stem[-4:]) for f in d.glob('checkpoint-step????.json')}
        assert cp_steps == expected_steps, ('checkpoint coverage', name)
        retained_checkpoints += len(cp_steps)
        for step in plan['evaluation_nodes']:
            if step in cp_steps:
                expected_endpoints.add(f'{name}-step{step:04}-development')
    assert set(accepted) == expected_endpoints, ('endpoint coverage', set(accepted) ^ expected_endpoints)
    inputs = [json.loads(s) for s in (root / 'data/prepared-v1/manifest.jsonl').read_text().splitlines()]
    split_ids = {split: [r['sample_id'] for r in inputs if r['split'] == split]
                 for split in ('train', 'development', 'final')}
    assert {k: len(v) for k, v in split_ids.items()} == {'train': 1000, 'development': 128, 'final': 542}
    release = read(root / 'final-evaluation-release.json')
    assert release['protocol_sha256'] == protocol and release['methods_unchanged'] is True
    observed = {}
    dev_starts = []
    dev_ends = []
    final_starts = []
    for endpoint, path in accepted.items():
        out = Path(path)
        report = read(out / 'summary.json')
        rows = read(out / 'observations.json')
        assert report['status'] == 'PASS' and report['protocol_sha256'] == protocol
        linear = endpoint.startswith('linear-head')
        job = ('linear-fit-v1' if endpoint == 'linear-head-fit' else 'linear-final-v1') if linear else 'eval-' + endpoint + '-v1'
        control = root / 'control' / job
        process = read(control / 'process.json')
        end = read(control / 'exit.json')
        assert end['exit_code'] == 0
        if endpoint.endswith('final'):
            final_starts.append(process['start_epoch'])
        else:
            dev_starts.append(process['start_epoch'])
            dev_ends.append(end['end_epoch'])
        audit_opened(out / 'file-open-audit.json', len(rows))
        if linear:
            assert read(out / 'initial-state.json') == init_by_seed[43]
            allowed = ('train', 'development') if endpoint == 'linear-head-fit' else ('final',)
            assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in inputs if r['split'] in allowed]
        else:
            split = 'final' if endpoint.endswith('final') else 'development'
            assert [r['sample_id'] for r in rows] == split_ids[split]
            assert len(list(out.glob('record-????.json'))) == len(list(out.glob('raw-????.npz'))) == len(rows)
            assert report['parameter_and_buffer_identity_preserved'] is True
        for split in sorted({r['split'] for r in rows}):
            part = [r for r in rows if r['split'] == split]
            key = 'linear-head-' + split if linear else endpoint
            assert key not in observed
            observed[key] = part
            audit_classes_and_spatial(part, report['metrics'][split] if linear else report['metrics'])
    assert max(train_ends) <= min(dev_starts)
    assert max(dev_ends) <= release['epoch'] <= min(final_starts)
    summary = read(root / 'results-v1/endpoint-summary.json')
    verify_csv(root / 'results-v1/endpoint-summary.csv', summary)
    fields = ('sample_id', 'target', 'parsed', 'score', 'text', 'effective_support', 'top11_mass',
              'yes_no_logit_gap', 'pixel_sha256_prepared')
    csv_rows = [{'endpoint': endpoint, **{k: r.get(k) for k in fields}}
                for endpoint, rows in observed.items() for r in rows]
    verify_csv(root / 'results-v1/per-image.csv', csv_rows)
    expected_pairs = set()
    for seed in plan['seeds']:
        for split in ('development', 'final'):
            for step in plan['evaluation_nodes'] if split == 'development' else [1000]:
                endpoints = {arm: f's{seed}-{arm}-step{step:04}-{split}' for arm in plan['arms']}
                comparisons = [('cabg_lce', 'lm_only'), ('cabg_lce', 'fixed_lce'), ('fixed_lce', 'lm_only')]
                for left, right in comparisons:
                    if endpoints[left] in observed and endpoints[right] in observed:
                        expected_pairs.add(endpoints[left] + '__minus__' + endpoints[right])
                if step == 1000:
                    for endpoint in endpoints.values():
                        if endpoint in observed:
                            expected_pairs.add(endpoint + '__minus__linear-head-' + split)
    pairs = read(root / 'results-v1/paired-comparisons.json')
    assert set(pairs) == expected_pairs
    analysis = read(root / 'results-v1/analysis-status.json')
    assert analysis['accepted_readouts'] == len(accepted)
    assert analysis['summary_rows'] == len(observed) == len(summary)
    assert analysis['per_image_rows'] == len(csv_rows)
    output = dict(status='PASS', protocol_sha256=protocol, trajectories=len(terminal),
                  statuses=dict(Counter(r['status'] for r in terminal.values())),
                  retained_checkpoints=retained_checkpoints, accepted_readouts=len(accepted),
                  summary_rows=len(observed), per_image_rows=len(csv_rows), paired_comparisons=len(pairs),
                  gate_chronology='all training ends before dev/head, then frozen release before first final process',
                  scope='coverage/CSV/class/spatial/input/initial-state/source/chronology checks; complements raw and training numerical verification')
    with (root / 'coverage-independent-verification-v1.json').open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
