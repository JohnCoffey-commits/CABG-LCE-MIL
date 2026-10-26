"""One frozen initial visual extractor and the prior fixed convex head recipe."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from reproduction.stage2af.runtime import Runtime,write,sha_file
from reproduction.stage2af.frozen import require_frozen
from reproduction.stage2af.inputs import load_prepared
from reproduction.stage2af.guard import InputGuard
from reproduction.stage2af.budget import Budget
from reproduction.stage2af.metrics import summary
from reproduction.stage2s.common import restore_runtime
from reproduction.stage2q.run import full_state
from reproduction.stage2ac.linear import fit_head,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['fit','final'],required=True);p.add_argument('--attempt',default='v1');a=p.parse_args()
    root=a.root;plan,protocol=require_frozen(root)
    if a.phase=='final':assert (root/'final-evaluation-release.json').is_file()
    out=root/'evaluation/linear-head'/a.phase/a.attempt;out.mkdir(parents=True,exist_ok=False)
    r=Runtime(root,43,training=False);rows=[x for x in r.rows if (x['split'] in ['train','development'] if a.phase=='fit' else x['split']=='final')]
    guard=InputGuard(root,rows).install();budget=Budget(root);budget.check(space=500_000_000);started=time.time();features=[];observations=[]
    try:
        write(out/'initial-state.json',r.initial_full)
        for i,row in enumerate(rows):
            budget.check();inputs=load_prepared(row,r.wrapper.device);restore_runtime(r.model,r.clean_runtime)
            with torch.inference_mode():
                visual=r.model.visual(inputs['pixel_values'].to(r.model.visual.dtype),grid_thw=inputs['image_grid_thw'])
                feat=visual[0].float().mean(0).cpu().numpy()
            assert feat.shape==(3584,) and np.isfinite(feat).all()
            features.append(feat);observations.append({k:row[k] for k in ['sample_id','target','group_id','split','sha256','pixel_sha256_prepared','image_grid_thw','input_ids','prepared_sha256']})
            if (i+1)%100==0:print(json.dumps({'linear_phase':a.phase,'features':i+1,'total':len(rows)}),flush=True)
            del inputs,visual,feat;gc.collect()
        features=np.stack(features)
        with (out/'features.npy').open('xb') as f:np.save(f,features)
        if a.phase=='fit':
            mask=np.array([x['split']=='train' for x in observations]);assert mask.sum()==1000
            head=fit_head(features[mask],[x['target'] for x in observations if x['split']=='train'])
            head={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in head.items()};write(out/'head.json',head)
        else:
            fitroot=root/'evaluation/linear-head/fit/v1';head=json.loads((fitroot/'head.json').read_text())
            assert json.loads((fitroot/'summary.json').read_text())['status']=='PASS'
        numeric={k:np.asarray(v) if isinstance(v,list) else v for k,v in head.items()};scores=predict(numeric,features)
        for row,score in zip(observations,scores):row.update(score=float(score),parsed=int(score>=0))
        write(out/'observations.json',observations)
        restore_runtime(r.model,r.clean_runtime);assert full_state(r.model)==r.initial_full
        write(out/'file-open-audit.json',guard.record());assert not guard.violations
        splits=sorted({r['split'] for r in observations})
        write(out/'summary.json',{'status':'PASS','protocol_sha256':protocol,'phase':a.phase,'readout':'fixed linear classifier, not generated answer',
            'metrics':{s:summary([r for r in observations if r['split']==s]) for s in splits},
            'features_sha256':sha_file(out/'features.npy'),'initial_parameter_identity_unchanged':True,'wall_seconds':time.time()-started})
    finally:r.close()
if __name__=='__main__':main()
