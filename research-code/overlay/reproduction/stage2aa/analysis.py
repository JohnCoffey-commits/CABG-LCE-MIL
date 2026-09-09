"""Producer fixed-history summaries, rational rank comparisons and full contrasts."""
import argparse
from fractions import Fraction
from pathlib import Path
from reproduction.stage2z.analysis import summary
from reproduction.stage2aa.common import read,write,Inputs,SEEDS,HISTORIES,CONTRASTS

def exact_ranks(rows):
    positives=[r for r in rows if r['label']=='abnormal'];negatives=[r for r in rows if r['label']=='normal']
    assert positives and negatives and len(positives)+len(negatives)==len(rows)
    auc=sum((Fraction(int(p['score']>n['score']),1)+Fraction(int(p['score']==n['score']),2)) for p in positives for n in negatives)/(len(positives)*len(negatives))
    seen=tp=0;ap=Fraction()
    for score in sorted({r['score'] for r in rows},reverse=True):
        group=[r for r in rows if r['score']==score];new=sum(r['label']=='abnormal' for r in group)
        tp+=new;seen+=len(group);ap+=Fraction(new,len(positives))*Fraction(tp,seen)
    return dict(auroc=auc,average_precision=ap)

def encode_ranks(value):return {k:str(v) for k,v in value.items()}

def relations(ranks):
    late_none={k:ranks['late'][k]-ranks['none'][k] for k in ranks['late']}
    late_early={k:ranks['late'][k]-ranks['early'][k] for k in ranks['late']}
    gain='GAIN_BOTH' if all(v>0 for v in late_none.values()) else 'NO_GAIN_BOTH' if all(v<=0 for v in late_none.values()) else 'MIXED_METRICS_OR_TIE'
    timing='LATE_MATCHES_OR_EXCEEDS_BOTH' if all(v>=0 for v in late_early.values()) else 'EARLY_HIGHER_BOTH' if all(v<0 for v in late_early.values()) else 'MIXED_METRICS_OR_TIE'
    return dict(late_vs_none=dict(exact=encode_ranks(late_none),pattern=gain),late_vs_early=dict(exact=encode_ranks(late_early),pattern=timing))

def decide(comparisons):
    assert set(comparisons)=={'43','44'}
    gains=[comparisons[s]['late_vs_none']['pattern'] for s in ('43','44')]
    timing=[comparisons[s]['late_vs_early']['pattern'] for s in ('43','44')]
    gain='DESCRIPTIVE_LATE_ONLY_GAIN_2_OF_2' if all(x=='GAIN_BOTH' for x in gains) else 'NO_LATE_ONLY_GAIN_2_OF_2' if all(x=='NO_GAIN_BOTH' for x in gains) else 'MIXED_OR_METRIC_DEPENDENT_LATE_GAIN'
    order='LATE_MATCHES_OR_EXCEEDS_EARLY_2_OF_2' if all(x=='LATE_MATCHES_OR_EXCEEDS_BOTH' for x in timing) else 'EARLY_HIGHER_BOTH_2_OF_2' if all(x=='EARLY_HIGHER_BOTH' for x in timing) else 'MIXED_EARLY_LATE_ORDERING'
    return dict(late_gain=gain,early_late_order=order)

def endpoint_rows(root,ins,seed):
    return dict(none=read(root/f'eval-ref-s{seed}-lm_only24/predictions.json'),
        late=read(root/f'eval-s{seed}-late_only24/predictions.json'),
        early=ins.js(f'eval-s{seed}-lce_off24/predictions.json'),full=ins.js(f'eval-s{seed}-dynamic24/predictions.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();ins=Inputs()
    summaries={};rational={};comparisons={};contrasts={}
    for seed in SEEDS:
        s=str(seed);rows=endpoint_rows(a.campaign,ins,seed)
        summaries[s]={h:summary(rows[h]) for h in HISTORIES};ranks={h:exact_ranks(rows[h]) for h in HISTORIES}
        rational[s]={h:encode_ranks(ranks[h]) for h in HISTORIES};comparisons[s]=relations(ranks);contrasts[s]={}
        for name,weights in CONTRASTS.items():
            contrasts[s][name]={k:float(sum(weight*(ranks[h][k] if k in ranks[h] else summaries[s][h][k]) for h,weight in weights.items())) for k in summaries[s]['none']}
    write(a.campaign/'timing-result.json',dict(status='PASS',decision=decide(comparisons),summaries=summaries,
        exact_ranks=rational,comparisons=comparisons,contrasts=contrasts))
    print(decide(comparisons))
if __name__=='__main__':main()
