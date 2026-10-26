"""Production reduction of immutable diagnostic records; no model computation."""
import argparse,json
from pathlib import Path
import numpy as np
from reproduction.stage2af.metrics import summary,paired

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--attempts',nargs=2,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);endpoints={};comparisons={}
    for seed,attempt in zip([43,44],a.attempts):
        rows=json.loads((a.root/'diagnostics'/f'seed{seed}'/attempt/'observations.json').read_text())
        conditions=['native','center_rms','softsign_rms','softsign_shuffle_rms','ano_swap','visual_swap']
        groups={c:[r for r in rows if r['condition']==c] for c in conditions}
        for c,rs in groups.items():
            assert len(rs)==128
            d=summary(rs)
            native=groups['native']
            d['answers_changed']=sum(r['output_ids']!=b['output_ids'] for r,b in zip(rs,native))
            d['median_absolute_logit_change']=float(np.median([abs(r['yes_no_logit_gap']-b['yes_no_logit_gap']) for r,b in zip(rs,native)]))
            if c!='native':d['median_relative_change']={k:float(np.median([r['relative_change'][k] for r in rs])) for k in rs[0]['relative_change']}
            endpoints[f's{seed}-{c}']=d
        for c in conditions[1:]:comparisons[f's{seed}-{c}__minus__native']=paired(groups[c],groups['native'],bootstrap_seed=seed)
        comparisons[f's{seed}-softsign_rms__minus__softsign_shuffle_rms']=paired(groups['softsign_rms'],groups['softsign_shuffle_rms'],bootstrap_seed=seed)
    report=dict(status='PRODUCER_REDUCTION',endpoints=endpoints,comparisons=comparisons)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
