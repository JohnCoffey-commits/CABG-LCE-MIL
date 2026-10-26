"""Explicit post-hoc amendment; reuse the existing bootstrap, no new draws."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
z=np.load(ROOT/'data/derived/bootstrap_final.npz');eps=z['endpoints'].tolist();a=z['values'][:,:,2]
d=pd.read_csv(ROOT/'tables/02_scale_paired_original.csv')
final=d[d.left.str.endswith('-final')&d.right.str.endswith('-final')].copy()
assert len(final)==12
rows=[]
for _,r in final.iterrows():
    diff=a[:,eps.index(r.left)]-a[:,eps.index(r.right)]
    rows.append({**r.to_dict(),'new_ci95_low':np.quantile(diff,.025),'new_ci95_high':np.quantile(diff,.975),
                 'family12_ci_low':np.quantile(diff,.05/24),'family12_ci_high':np.quantile(diff,1-.05/24),
                 'family_size':12,'replicates':len(diff),'analysis_status':'post-hoc amendment01; final-only comparison family'})
out=pd.DataFrame(rows);out.to_csv(ROOT/'tables/18_ba_final_multiplicity_sensitivity.csv',index=False,float_format='%.15g')
print(out[['left','right','delta_balanced_accuracy','ba_ci_low','ba_ci_high','family12_ci_low','family12_ci_high']].to_string(index=False))
