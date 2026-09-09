"""Prospective selection, parsing, scope and time gates; no model dependency."""
import json
import re
import time
from pathlib import Path

QUESTION = 'Is there any anomaly in the image?\nAnswer the question using a single word or phrase.'
SEEDS = (43, 44)
ARMS = ('none', 'full', 'early')
MAX_NEW_TOKENS = 16


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def selected_development(rows):
    if len(rows) != 32 or len({r['sha256'] for r in rows}) != 32:
        raise ValueError('Original 32-case development membership required')
    chosen = []
    for label in ('normal', 'abnormal'):
        candidates = sorted((r for r in rows if r['scout_label'] == label),
                            key=lambda r: r['scout_eval_rank'])
        if len(candidates) < 2:
            raise ValueError('Two cases of each class required')
        chosen.extend(candidates[:2])
    return sorted(chosen, key=lambda r: r['scout_eval_rank'])


def parse_answer(text):
    # Entire response only: explanations, contradictions and empty answers
    # remain invalid and count as incorrect, never silently leave denominator.
    token = text.strip().lower()
    if re.fullmatch(r'(yes|no)[.!]?', token):
        return int(token.startswith('yes'))
    return None


def answer_metrics(rows):
    if not rows:
        raise ValueError('No answer observations')
    correct = sum(r['parsed'] is not None and r['parsed'] == r['target'] for r in rows)
    return {'n':len(rows), 'correct':correct,
            'invalid':sum(r['parsed'] is None for r in rows),
            'accuracy':correct / len(rows)}


def require_time(plan, now=None):
    now = time.time() if now is None else now
    # Keep 90 minutes for independent verification/closure inside the 10h cap.
    if now >= plan['deadline_epoch'] - 5400:
        raise RuntimeError('No new model work: reserved closure time reached')


def unique_training(rows):
    by_hash = {}
    for row in rows:
        if row['sha256'] in by_hash:
            old = by_hash[row['sha256']]
            if (old['sample_id'], old['scout_label']) != (row['sample_id'], row['scout_label']):
                raise ValueError('Inconsistent repeated training exposure')
        by_hash[row['sha256']] = row
    return sorted(by_hash.values(), key=lambda r:r['sample_id'])
