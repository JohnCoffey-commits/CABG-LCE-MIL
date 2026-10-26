"""Lightweight analysis of existing development numerical records; no model imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import spearmanr, rankdata


def auc(y, x):
    y, x = np.asarray(y), np.asarray(x)
    n1, n0 = int(y.sum()), int((1-y).sum())
    return float((rankdata(x)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mirror', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows, identities, summaries = [], {}, {}
    for seed in [43, 44]:
        for arm in ['lm_only', 'fixed_lce', 'cabg_lce']:
            name = f's{seed}-{arm}-step1000-development'
            root = a.mirror/'evaluation'/name/'v1'
            observations = root/'observations.json'
            records = json.loads(observations.read_text())
            assert len(records) == 128 and {r['split'] for r in records} == {'development'}
            identities[str(observations)] = hashlib.sha256(observations.read_bytes()).hexdigest()
            group = []
            for r in records:
                raw_path = root/r['raw_file']
                digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                assert digest == r['raw_sha256']
                identities[str(raw_path)] = digest
                with np.load(raw_path, allow_pickle=False) as z:
                    aa, nn = z['abnormal'].astype('float64'), z['normal'].astype('float64')
                d = aa-nn
                c = (aa+nn)/2
                soft = d/(1+np.abs(d))
                official = expit(aa)-expit(nn)
                centered = np.tanh(d/4)
                tau = 1/np.log(d.shape[-1])
                score = float(tau*(logsumexp(soft.mean(axis=1)/tau)-np.log(d.shape[-1])))
                def rms(x): return float(np.sqrt(np.mean(x*x)))
                ef, sf = official.mean(axis=1).ravel(), soft.mean(axis=1).ravel()
                row = dict(endpoint=name, seed=seed, arm=arm, sample_id=r['sample_id'],
                           target=r['target'], parsed=r['parsed'], correct=int(r['parsed']==r['target']),
                           score=score, yes_no_logit_gap=r['yes_no_logit_gap'],
                           common_abs_mean=float(np.abs(c).mean()), common_mean=float(c.mean()),
                           logit_gap_abs_mean=float(np.abs(d).mean()), probability_rms=rms(official),
                           centered_rms=rms(centered), softsign_rms=rms(soft),
                           common_suppression_ratio=rms(official)/max(rms(centered),1e-30),
                           probability_nearzero_fraction=float((np.abs(official)<1e-3).mean()),
                           representation_cosine=float(np.dot(ef,sf)/(np.linalg.norm(ef)*np.linalg.norm(sf)+1e-30)),
                           representation_rank_correlation=corr(ef,sf),
                           top11_overlap=len(set(np.argsort(ef)[-11:]) & set(np.argsort(sf)[-11:]))/11)
                group.append(row)
            rows.extend(group)
            y = [r['target'] for r in group]
            err = [1-r['correct'] for r in group]
            summaries[name] = {
                'n': len(group), 'correct': sum(r['correct'] for r in group),
                'class_correct': {str(t): sum(r['correct'] for r in group if r['target']==t) for t in [0,1]},
                'score_auroc': auc(y,[r['score'] for r in group]),
                'answer_logit_auroc': auc(y,[r['yes_no_logit_gap'] for r in group]),
                'score_answer_rank_correlation': corr([r['score'] for r in group],[r['yes_no_logit_gap'] for r in group]),
                'median': {k: float(np.median([r[k] for r in group])) for k in [
                    'common_abs_mean','logit_gap_abs_mean','probability_rms','centered_rms',
                    'softsign_rms','common_suppression_ratio','probability_nearzero_fraction',
                    'representation_cosine','representation_rank_correlation','top11_overlap']},
                'within_class_error_correlations': {str(t): {k:corr([r[k] for r in group if r['target']==t],
                    [1-r['correct'] for r in group if r['target']==t]) for k in ['common_suppression_ratio','common_abs_mean','representation_cosine']} for t in [0,1]},
            }
    with (a.output/'per-image.csv').open('x') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report = {'scope':'Existing 128-image development set only; observational exploratory diagnostic. Analytic sigmoid in float64, not a new model forward or exact BF16 gate reconstruction.',
              'summaries':summaries,'source_identities':identities}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,indent=2,allow_nan=False))


if __name__=='__main__': main()
