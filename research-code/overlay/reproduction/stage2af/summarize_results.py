"""Readable detail tables from already independently verified fixed readouts.

No new endpoints, tests, model execution or method selection. Class counts are
recounted from observations and checked against each verified endpoint report.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(folder, name, rows):
    with (folder / (name + '.json')).open('x') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write('\n')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / (name + '.csv')).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with (folder / (name + '.csv')).open(newline='') as stream:
        actual = list(csv.DictReader(stream))
    expected = [{k: '' if row.get(k) is None else str(row[k]) for k in fields}
                for row in rows]
    assert actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    for name in ['results-v1/independent-verification.json',
                 'coverage-independent-verification-v1.json']:
        assert read(root / name)['status'] == 'PASS'
    accepted = read(root / 'accepted-endpoints.json')
    sources = {}
    endpoint_rows = []
    for name, path in accepted.items():
        folder = Path(path)
        report = read(folder / 'summary.json')
        observations = read(folder / 'observations.json')
        for filename in ['summary.json', 'observations.json']:
            sources[str((folder / filename).relative_to(root))] = sha(folder / filename)
        for split in sorted({row['split'] for row in observations}):
            rows = [row for row in observations if row['split'] == split]
            linear = name.startswith('linear-head')
            endpoint = 'linear-head-' + split if linear else name
            metrics = report['metrics'][split] if linear else report['metrics']
            result = dict(endpoint=endpoint, split=split,
                          readout='fixed linear classification' if linear else 'actual generated answer')
            for key in ['n', 'correct', 'accuracy', 'balanced_accuracy', 'invalid',
                        'auroc', 'average_precision']:
                result[key] = metrics[key]
            for target in (0, 1):
                part = [row for row in rows if row['target'] == target]
                count = len(part)
                correct = sum(row['parsed'] == target for row in part)
                invalid = sum(row['parsed'] is None for row in part)
                computed = dict(n=count, correct=correct, correct_rate=correct/count,
                                invalid=invalid)
                assert metrics['by_class'][str(target)] == computed
                for key, value in computed.items():
                    result[f'class{target}_{key}'] = value
            for key in ['effective_support', 'top11_mass', 'yes_no_logit_gap']:
                if key in rows[0]:
                    median = statistics.median(row[key] for row in rows)
                    assert abs(median - metrics[key]['median']) < 1e-12
                    result[key + '_median'] = median
                else:
                    result[key + '_median'] = None
            endpoint_rows.append(result)
    original = read(root / 'results-v1/endpoint-summary.json')
    assert {r['endpoint'] for r in endpoint_rows} == {r['endpoint'] for r in original}
    assert len(endpoint_rows) == len(original)
    paired = read(root / 'results-v1/paired-comparisons.json')
    sources['results-v1/paired-comparisons.json'] = sha(root / 'results-v1/paired-comparisons.json')
    paired_rows = []
    for name, record in paired.items():
        left, right = name.split('__minus__')
        row = dict(left=left, right=right)
        for key in ['n', 'left_only_correct', 'right_only_correct', 'both_correct',
                    'both_wrong', 'delta_balanced_accuracy', 'delta_accuracy',
                    'delta_auroc', 'delta_average_precision']:
            row[key] = record[key]
        row['balanced_accuracy_ci95_low'], row['balanced_accuracy_ci95_high'] = record['paired_balanced_accuracy_95ci']
        row['zero_observed_discordance'] = record.get('bootstrap_degenerate_zero_discordance', False)
        row['zero_discordance_absolute_BA_bound'] = record.get('zero_discordance_absolute_BA_bound')
        row['uncertainty_scope'] = record['uncertainty_scope']
        paired_rows.append(row)
    output = root / 'results-detail-v1'
    output.mkdir(exist_ok=False)
    emit(output, 'endpoint-details', endpoint_rows)
    emit(output, 'paired-comparisons', paired_rows)
    with (output / 'verification.json').open('x') as stream:
        json.dump(dict(status='PASS', endpoint_rows=len(endpoint_rows),
                       paired_rows=len(paired_rows), sources=sources,
                       scope='Class counts and spatial medians recounted from verified observations; original paired estimates copied exactly; CSV round-trip checked. No new statistical comparison.'), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', endpoint_rows=len(endpoint_rows),
                          paired_rows=len(paired_rows), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
