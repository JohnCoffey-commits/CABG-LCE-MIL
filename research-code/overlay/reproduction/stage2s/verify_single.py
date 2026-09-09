"""Independent archived-gradient arithmetic and grouped single-step comparisons."""
import argparse
import json
from pathlib import Path
import torch
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2q.run import load_raw
from reproduction.stage2q.verify import compare_map,verify_raw_math,small_enough
from reproduction.stage2i.fingerprint import sha256_file
from reproduction.stage2s.common import write,state_digest

VPT=tuple(n for n in TRAINABLE_NAMES if 'deep_prompt_embeddings' in n)

def groups(a,b):
    return {k:compare_map({n:a[n] for n in ns},{n:b[n] for n in ns}) for k,ns in
            [('T21',TRAINABLE_NAMES),('S9',S9_NAMES),('VPT',VPT)]}

def passed(comparisons,bound):
    checks=[]
    for c in comparisons.values():
        checks.extend([v for k,v in c.items()])
        checks.extend(c['VPT']['per_tensor'].values())
    return all(v['relative_l2']<=bound for v in checks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True)
    p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    def get(path):
        d=load_raw(path);rec=json.loads(path.with_suffix('').with_suffix('.json').read_text())
        assert sha256_file(path)==rec['raw']['sha256']
        start=load_raw(path.parent/'start.pt.gz')
        check=verify_raw_math(d,rec,start)
        return d,rec,start,check
    x,xr,xs,xc=get(a.left);y,yr,ys,yc=get(a.right)
    assert state_digest(xs['trainable'])==state_digest(ys['trainable'])
    assert state_digest(x['inputs'])==state_digest(y['inputs']) and xr['rng']==yr['rng']
    assert state_digest(xs['runtime'])==state_digest(ys['runtime'])
    assert json.loads((a.left.parent/'full-initial-state.json').read_text())==json.loads((a.right.parent/'full-initial-state.json').read_text())
    delta=lambda d,s:{n:d['updated_master'][n]-s['trainable'][n].float() for n in TRAINABLE_NAMES}
    comparisons={'applied':groups(x['applied'],y['applied']),
                 'predicted_update':groups(delta(x,xs),delta(y,ys))}
    raw={key:[compare_map(g,h) for g,h in zip(x[key],y[key])] for key in ('lm','lce')}
    raw.update({key:compare_map(x[key],y[key]) for key in ('aggregate_lm','aggregate_lce')})
    # Counterfactual: identical reference lambda with the second raw gradients.
    # This is an offline diagnostic, never an alternate controller used in training.
    lam=xr['controller']['lambda_final']; scale=yr['gradient']['clip_scale']
    fixed={n:(g+(y['aggregate_lce'][n]*lam if n in S9_NAMES else 0))*scale for n,g in y['aggregate_lm'].items()}
    nonvpt=[n for n in S9_NAMES if n not in VPT]
    propagation={'reference_lambda':lam,'other_lambda':yr['controller']['lambda_final'],
      'lambda_relative_difference':abs(lam-yr['controller']['lambda_final'])/max(abs(lam),abs(yr['controller']['lambda_final']),1e-12),
      'nonvpt_s9_applied':compare_map({n:x['applied'][n] for n in nonvpt},{n:y['applied'][n] for n in nonvpt}),
      'nonvpt_s9_fixed_lambda':compare_map({n:x['applied'][n] for n in nonvpt},{n:fixed[n] for n in nonvpt})}
    strict=all(small_enough(g) and all(small_enough(v) for v in g['per_tensor'].values()) for c in comparisons.values() for g in c.values())
    report={'left':str(a.left),'right':str(a.right),'state_input_rng_exact':True,'independent_math':[xc,yc],
       'raw_gradients':raw,'comparisons':comparisons,'controller_propagation':propagation,
       'strict_pass':strict,'bounded_pass':passed(comparisons,.02),
       'all_outputs_exact':all(g['exact'] for c in comparisons.values() for g in c.values())}
    write(a.output,report)
    print(json.dumps({k:report[k] for k in ('strict_pass','bounded_pass','all_outputs_exact','controller_propagation')},indent=2))

if __name__=='__main__':main()
