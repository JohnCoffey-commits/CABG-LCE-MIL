"""Producer descriptive summaries for the locked full-objective ablation."""
import argparse
from statistics import mean,median
from pathlib import Path
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2z.common import read,write,Inputs,SEEDS

def summary(rows):
    p=summarize_metrics(rows)
    return dict(auroc=p['auroc'],average_precision=p['average_precision'],support_median=median(r['effective_support'] for r in rows),
        support_below128=sum(r['effective_support']<128 for r in rows),top11_above035=sum(r['top11_mass']>.35 for r in rows),
        lm_loss_image_mean=mean(r['lm_loss'] for r in rows),lm_loss_token_weighted=sum(r['lm_loss']*r['valid_tokens'] for r in rows)/sum(r['valid_tokens'] for r in rows))

def relation(reference,baseline):
    a=reference['auroc']-baseline['auroc'];p=reference['average_precision']-baseline['average_precision']
    return dict(delta_auroc=a,delta_average_precision=p,
        pattern='REFERENCE_HIGHER_BOTH' if a>0 and p>0 else 'BASELINE_MATCHES_OR_EXCEEDS_BOTH' if a<=0 and p<=0 else 'MIXED_METRICS_OR_TIE')

def decide(comparisons):
    assert set(comparisons)=={'43','44'}
    out={}
    for arm in ('dynamic','early_only'):
        ps=[comparisons[s][arm]['pattern'] for s in ('43','44')]
        out[arm]='DESCRIPTIVE_LCE_GAIN_2_OF_2' if all(p=='REFERENCE_HIGHER_BOTH' for p in ps) else 'LM_ONLY_MATCHES_OR_EXCEEDS_2_OF_2' if all(p=='BASELINE_MATCHES_OR_EXCEEDS_BOTH' for p in ps) else 'MIXED_OR_METRIC_DEPENDENT_GAIN'
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();ins=Inputs()
    summaries={};comparisons={}
    for seed in SEEDS:
        b=summary(read(a.campaign/f'eval-s{seed}-lm_only24/predictions.json'))
        d=summary(read(a.campaign/f'eval-ref-s{seed}-dynamic24/predictions.json'))
        e=summary(ins.js(f'eval-s{seed}-lce_off24/predictions.json'))
        summaries[str(seed)]=dict(lm_only=b,dynamic=d,early_only=e)
        comparisons[str(seed)]={name:relation(value,b) for name,value in (('dynamic',d),('early_only',e))}
    write(a.campaign/'comparison-result.json',dict(status='PASS',summaries=summaries,comparisons=comparisons,decision=decide(comparisons)))
    print(decide(comparisons))
if __name__=='__main__':main()
