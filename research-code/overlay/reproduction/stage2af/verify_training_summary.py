"""Recount descriptive training tables from immutable scalar/exit records."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def median(values):
    values = sorted(values)
    if not values:
        return None
    return (values[(len(values)-1)//2] + values[len(values)//2]) / 2


def check(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, float):
            assert math.isclose(float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-12), (key, actual[key], value)
        else:
            assert actual[key] == value, (key, actual[key], value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    verification = read(root / 'training-independent-verification-v1.json')
    assert verification['status'] == 'PASS'
    folder = root / 'training-summary-v1'
    report = read(folder / 'summary.json')
    sources = read(folder / 'source-manifest.json')
    attempts = read(root / 'accepted-training-attempts.json')
    states = read(root / 'training-terminal-status-v1.json')
    with (folder / 'step-summary.csv').open(newline='') as stream:
        raw_csv = list(csv.DictReader(stream))
    csv_rows = {}
    for row in raw_csv:
        key = (row['trajectory'], int(row['step']))
        assert key not in csv_rows
        csv_rows[key] = {k: (None if v == '' else v == 'True' if v in ['True', 'False']
                            else int(v) if k == 'step' else v if k == 'trajectory' else float(v))
                         for k, v in row.items()}
    totals = {r['trajectory']: r for r in report['trajectories']}
    assert set(totals) == set(states)
    seen = set()
    for name, state in states.items():
        raw, rows = [], []
        arm = name[4:]
        for path in sorted((root / 'formal' / name / attempts[name]).glob('step????.json')):
            rel = path.relative_to(root).as_posix()
            body = path.read_bytes()
            assert hashlib.sha256(body).hexdigest() == sources[rel]
            seen.add(rel)
            record = json.loads(body)
            raw.append(record)
            g = record['gradient']
            auxiliary, lm = g['scaled_auxiliary_shared_norm'], g['lm_shared_norm']
            row = dict(trajectory=name, step=record['step'], learning_rate=record['lr'],
                       applied_coefficient=record['applied_coefficient'],
                       preclip_auxiliary_shared_norm=auxiliary,
                       postclip_auxiliary_shared_norm=auxiliary*g['clip_scale'],
                       lr_weighted_postclip_auxiliary_shared_norm=auxiliary*g['clip_scale']*record['lr'],
                       auxiliary_to_lm_shared_ratio=auxiliary/lm if lm else None,
                       full_gradient_preclip_norm=g['preclip_full_norm'], clip_scale=g['clip_scale'],
                       actual_dynamic_cap_active=(record['controller']['lambda_cap'] < record['controller']['lambda_raw']) if arm == 'cabg_lce' else None,
                       lm_loss=math.fsum(x['lm_loss'] for x in record['images'])/3,
                       class_balanced_lce_loss=math.fsum(w*x['lce_loss'] for w,x in zip((.5,.25,.25),record['images'])))
            check(csv_rows[(name, record['step'])], row)
            rows.append(row)
        coefficients = [r['applied_coefficient'] for r in rows]
        ratios = [r['auxiliary_to_lm_shared_ratio'] for r in rows if r['auxiliary_to_lm_shared_ratio'] is not None]
        end = read(root / 'control' / ('formal-'+name+'-'+attempts[name].removeprefix('attempt-')) / 'exit.json')
        expected = dict(trajectory=name, status=state['status'], completed_updates=len(rows),
                        unique_train_images_seen=len({x['sample_id'] for r in raw for x in r['images']}),
                        image_exposures=sum(len(r['images']) for r in raw),
                        coefficient_min=min(coefficients) if rows else None,
                        coefficient_median=median(coefficients), coefficient_max=max(coefficients) if rows else None,
                        preclip_auxiliary_norm_sum=math.fsum(r['preclip_auxiliary_shared_norm'] for r in rows),
                        postclip_auxiliary_norm_sum=math.fsum(r['postclip_auxiliary_shared_norm'] for r in rows),
                        lr_weighted_postclip_auxiliary_norm_sum=math.fsum(r['lr_weighted_postclip_auxiliary_shared_norm'] for r in rows),
                        auxiliary_to_lm_shared_ratio_median=median(ratios),
                        auxiliary_to_lm_shared_ratio_max=max(ratios) if ratios else None,
                        global_clipping_fraction=sum(r['clip_scale'] < 1 for r in rows)/len(rows) if rows else None,
                        actual_dynamic_cap_fraction=sum(r['actual_dynamic_cap_active'] for r in rows)/len(rows) if arm == 'cabg_lce' and rows else None,
                        first24_mean_lm_loss=math.fsum(r['lm_loss'] for r in rows[:24])/len(rows[:24]) if rows else None,
                        last24_mean_lm_loss=math.fsum(r['lm_loss'] for r in rows[-24:])/len(rows[-24:]) if rows else None,
                        model_process_seconds=end['wall_seconds'], sampled_peak_memory_MiB=end['peak_memory_MiB'])
        check(totals[name], expected)
    assert seen == set(sources)
    assert len(csv_rows) == len(seen) == verification['steps_verified']
    with (folder / 'trajectory-summary.csv').open(newline='') as stream:
        actual = list(csv.DictReader(stream))
    expected = [{k: '' if v is None else str(v) for k, v in row.items()} for row in report['trajectories']]
    assert actual == expected
    result = dict(status='PASS', trajectories=len(states), scalar_rows=len(seen),
                  scope='Original scalar/exit recount of all step-table fields and trajectory reductions; source hashes and CSV/JSON consistency. Norm sums are not Adam displacements.')
    with (folder / 'verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
