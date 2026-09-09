"""Metadata-only readiness audit and historical development metric reduction."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

BASE_COMMIT = '71aaf0ad2f17a456d213b7d40b2ae7f080c6c7c2'
VERDICT_HASH = '3a341d15a2333e1e535c753ece2320dec1206b9dd7c8b4f6cc05937ee8bb50ba'
INVENTORY_HASH = '8cc2ec649030aeab58f0790fdf98e2c3a5c41c8e7becc93f835002ee036ed0b1'
IDENTITIES = ('sample_id', 'relative_path', 'sha256')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unpack(snapshot):
    result = {}
    for name, record in snapshot['files'].items():
        raw = record['text'].encode()
        if digest(raw) != record['sha256'] or len(raw) != record['bytes']:
            raise ValueError('Snapshot body identity mismatch: ' + name)
        result[name] = ([json.loads(line) for line in raw.splitlines() if line]
                        if record['path'].endswith('.jsonl') else json.loads(raw))
    if snapshot['head'] != BASE_COMMIT or snapshot['git_status']:
        raise ValueError('Unaccepted remote Git state')
    if len(snapshot['source_checks']) != 146 or not all(snapshot['source_checks'].values()):
        raise ValueError('Historical source mismatch')
    if snapshot['files']['aa_verification']['sha256'] != VERDICT_HASH:
        raise ValueError('Latest independent verdict mismatch')
    if snapshot['files']['aa_inventory']['sha256'] != INVENTORY_HASH:
        raise ValueError('Latest closed inventory mismatch')
    if result['aa_verification']['status'] != 'PASS':
        raise ValueError('Unaccepted prior execution')
    split = result['split']
    for name, key in [('pool', 'training_development'), ('d3r', 'd3r'), ('pilot', 'd4_pilot')]:
        if snapshot['files'][name]['sha256'] != split['source_hashes'][key]:
            raise ValueError('Original locked source mismatch: ' + name)
    for name, field in [('train', 'train_manifest_sha256'), ('development', 'eval_manifest_sha256')]:
        if snapshot['files'][name]['sha256'] != split[field]:
            raise ValueError('Original locked split mismatch: ' + name)
    return result


def partition(data):
    result = {}
    for key in IDENTITIES:
        sets = {name: {r[key] for r in data[name]} for name in ['pool', 'd3r', 'pilot', 'train', 'development']}
        pool, d3r, pilot, train, dev = (sets[n] for n in ['pool', 'd3r', 'pilot', 'train', 'development'])
        if d3r & pilot or train & dev or (train | dev) & (d3r | pilot):
            raise ValueError('Split/exclusion overlap: ' + key)
        if not (d3r | pilot | train | dev) <= pool:
            raise ValueError('Source membership mismatch: ' + key)
        eligible = pool - d3r - pilot
        result[key] = dict(pool=len(pool), excluded=len(d3r | pilot), train=len(train),
                           development=len(dev), eligible=len(eligible), unused=len(eligible - train - dev),
                           used_union_equals_eligible=(train | dev) == eligible)
    return result


def ranking(rows):
    labels = [r['label'] == 'abnormal' for r in rows]
    if any(r['label'] not in ('normal', 'abnormal') for r in rows):
        raise ValueError('Nonbinary label')
    positives = sum(labels); negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError('Both classes required')
    groups = {}
    for r, label in zip(rows, labels):
        score = float(r['score'])
        if not __import__('math').isfinite(score):
            raise ValueError('Nonfinite score')
        groups.setdefault(score, [0, 0])[int(label)] += 1
    below = 0; wins = Fraction(0)
    for score in sorted(groups):
        n, p = groups[score]
        wins += p * (below + Fraction(n, 2)); below += n
    tp = total = 0; ap = Fraction(0)
    for score in sorted(groups, reverse=True):
        n, p = groups[score]; tp += p; total += n + p
        ap += Fraction(p, positives) * Fraction(tp, total)
    return {'auroc': float(wins / (positives * negatives)), 'average_precision': float(ap)}


def endpoint_files(workspace):
    doc = Path(workspace) / 'document/cabg_mil_v1_2'
    for seed in (43, 44):
        for arm, area, campaign, endpoint in [
            ('none', 'lm-only-baseline', 'lm-only-baseline-v1', 'lm_only24'),
            ('full', 'initialization-lce-persistence', 'initialization-lce-persistence-v1', 'dynamic24'),
            ('early', 'initialization-lce-persistence', 'initialization-lce-persistence-v1', 'lce_off24'),
        ]:
            p = doc / area / 'verified-execution-v1' / campaign / f'eval-s{seed}-{endpoint}/predictions.json'
            yield seed, arm, p


def analyze(snapshot, workspace):
    data = unpack(snapshot); parts = partition(data)
    if any(row != dict(pool=132, excluded=36, train=64, development=32,
                       eligible=96, unused=0, used_union_equals_eligible=True) for row in parts.values()):
        raise ValueError('Unexpected source partition; reassessment required')
    grouping_fields = ('patient_id', 'subject_id', 'study_id', 'patient_group_id')
    available = sum(any(r.get(k) for k in grouping_fields) for r in data['pool'])
    br = snapshot['br35h_entries']
    sidecars = [r['path'] for r in br if Path(r['path']).suffix.lower() not in ('.jpg', '.jpeg', '.png')]
    metrics = []
    for seed, arm, path in endpoint_files(workspace):
        raw = path.read_bytes(); rows = json.loads(raw)
        if len(rows) != 32 or {r['sha256'] for r in rows} != {r['sha256'] for r in data['development']}:
            raise ValueError('Historical endpoint membership mismatch')
        measured = ranking(rows)
        prior = data['aa_verification']['summaries'][str(seed)][arm]
        if any(abs(measured[k] - prior[k]) > 1e-12 for k in measured):
            raise ValueError('Historical ranking disagreement')
        metrics.append(dict(seed=seed, arm=arm, source=str(path), sha256=digest(raw), count=len(rows), **measured))
    return {
        'execution_validity': 'VALID_METADATA_AUDIT',
        'scientific_decision': 'BLOCKED_INDEPENDENT_COHORT_NOT_ESTABLISHED',
        'method_value_decision': 'UNDETERMINED',
        'partition': parts, 'pool_group_identifiers_available': available,
        'patient_independence_verified': False,
        'br35h': {'files': len(br), 'class_directory_counts': dict(Counter(str(Path(r['path']).parent) for r in br)),
                   'nonimage_sidecars': sidecars, 'image_bodies_opened': 0,
                   'source_and_group_separation': 'NOT_ESTABLISHED', 'prior_use_history': 'NOT_CLOSED'},
        'historical_development_rankings': metrics,
        'new_answer_effect': 'NOT_MEASURED', 'simple_baseline_effect': 'NOT_MEASURED',
        'new_independent_cases': 0, 'new_development_predictions': 0,
        'model_loads': 0, 'optimizer_updates': 0, 'images_opened': 0,
        'protected_images_or_results_opened': 0,
        'limits': {'l4_instance_on_hours': 10, 'usd_including_stage_storage': 40},
        'scope': 'Existing authorized mounted MRI metadata; no claim that no suitable external dataset exists.',
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = analyze(json.loads(a.snapshot.read_text()), a.workspace)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(result['scientific_decision'])


if __name__ == '__main__':
    main()
