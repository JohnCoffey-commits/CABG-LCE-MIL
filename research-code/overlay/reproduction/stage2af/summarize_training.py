"""Descriptive gradient-dose and resource tables from independently checked runs.

Gradient-norm sums are diagnostics, not Adam parameter displacement or a causal
explanation of model quality. This adds no selection or model execution.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def table(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    verification = read(root / 'training-independent-verification-v1.json')
    assert verification['status'] == 'PASS'
    states = read(root / 'training-terminal-status-v1.json')
    attempts = read(root/'accepted-training-attempts.json')
    out = root / 'training-summary-v1'
    out.mkdir(exist_ok=False)
    sources = {}
    trajectories, steps = [], []
    for trajectory, terminal in states.items():
        records = []
        for path in sorted((root / 'formal' / trajectory / attempts[trajectory]).glob('step????.json')):
            body = path.read_bytes()
            sources[str(path.relative_to(root))] = hashlib.sha256(body).hexdigest()
            records.append(json.loads(body))
        arm = trajectory[4:]
        rows = []
        for record in records:
            gradient = record['gradient']
            scale = gradient['clip_scale']
            auxiliary = gradient['scaled_auxiliary_shared_norm']
            lm_shared = gradient['lm_shared_norm']
            rows.append(dict(trajectory=trajectory, step=record['step'], learning_rate=record['lr'],
                             applied_coefficient=record['applied_coefficient'],
                             preclip_auxiliary_shared_norm=auxiliary,
                             postclip_auxiliary_shared_norm=auxiliary*scale,
                             lr_weighted_postclip_auxiliary_shared_norm=record['lr']*auxiliary*scale,
                             auxiliary_to_lm_shared_ratio=auxiliary/lm_shared if lm_shared else None,
                             full_gradient_preclip_norm=gradient['preclip_full_norm'], clip_scale=scale,
                             actual_dynamic_cap_active=(record['controller']['lambda_cap'] < record['controller']['lambda_raw']) if arm=='cabg_lce' else None,
                             lm_loss=sum(image['lm_loss'] for image in record['images'])/3,
                             class_balanced_lce_loss=sum(weight*image['lce_loss'] for weight,image in zip((.5,.25,.25),record['images']))))
        steps.extend(rows)
        exit_record = read(root / 'control' / ('formal-'+trajectory+'-'+attempts[trajectory].removeprefix('attempt-')) / 'exit.json')
        coefficients = [row['applied_coefficient'] for row in rows]
        if arm=='fixed_lce' and rows:
            assert len(set(coefficients)) == 1
        if arm=='lm_only':
            assert all(value==0 for value in coefficients)
        ratios = [row['auxiliary_to_lm_shared_ratio'] for row in rows if row['auxiliary_to_lm_shared_ratio'] is not None]
        unique_images = {image['sample_id'] for record in records for image in record['images']}
        trajectories.append(dict(trajectory=trajectory, status=terminal['status'], completed_updates=len(rows),
                                 unique_train_images_seen=len(unique_images), image_exposures=3*len(rows),
                                 coefficient_min=min(coefficients) if rows else None,
                                 coefficient_median=statistics.median(coefficients) if rows else None,
                                 coefficient_max=max(coefficients) if rows else None,
                                 preclip_auxiliary_norm_sum=sum(row['preclip_auxiliary_shared_norm'] for row in rows),
                                 postclip_auxiliary_norm_sum=sum(row['postclip_auxiliary_shared_norm'] for row in rows),
                                 lr_weighted_postclip_auxiliary_norm_sum=sum(row['lr_weighted_postclip_auxiliary_shared_norm'] for row in rows),
                                 auxiliary_to_lm_shared_ratio_median=statistics.median(ratios) if ratios else None,
                                 auxiliary_to_lm_shared_ratio_max=max(ratios) if ratios else None,
                                 global_clipping_fraction=sum(row['clip_scale']<1 for row in rows)/len(rows) if rows else None,
                                 actual_dynamic_cap_fraction=sum(row['actual_dynamic_cap_active'] for row in rows)/len(rows) if arm=='cabg_lce' and rows else None,
                                 first24_mean_lm_loss=statistics.mean(row['lm_loss'] for row in rows[:24]) if rows else None,
                                 last24_mean_lm_loss=statistics.mean(row['lm_loss'] for row in rows[-24:]) if rows else None,
                                 model_process_seconds=exit_record['wall_seconds'],
                                 sampled_peak_memory_MiB=exit_record['peak_memory_MiB']))
    assert len(steps)==verification['steps_verified']
    write(out/'source-manifest.json',sources)
    write(out/'summary.json',dict(status='DESCRIPTIVE_REDUCTION_OF_INDEPENDENTLY_VERIFIED_RECORDS',
                                protocol_sha256=verification['protocol_sha256'], trajectories=trajectories,
                                caveats=['Gradient-norm sums and LR-weighted sums are not Adam parameter displacement.',
                                         'Training losses and gradient doses do not establish answer improvement or localization.',
                                         'Fixed lambda is the single predeclared training-statistic recipe, not an optimized family of constant weights.',
                                         'All arms include matched LM/LCE diagnostic collection; runtime is not a minimal deployment benchmark.']))
    table(out/'trajectory-summary.csv',trajectories)
    if steps:
        table(out/'step-summary.csv',steps)
    print(json.dumps(dict(trajectories=len(trajectories), steps=len(steps), output=str(out))),flush=True)


if __name__=='__main__':
    main()
