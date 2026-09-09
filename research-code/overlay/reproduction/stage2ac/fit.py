"""Fit the single frozen-feature baseline once, with no model/GPU import."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reproduction.stage2ac.contracts import write
from reproduction.stage2ac.linear import fit_head,predict
from reproduction.stage2ab.readiness import ranking


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    root=a.campaign;out=root/'linear-head';out.mkdir(exist_ok=False)
    origin=root/'features';rows=json.loads((origin/'observations.json').read_text())
    assert json.loads((origin/'result.json').read_text())['status']=='PASS'
    x=np.load(origin/'features.npy',allow_pickle=False)
    assert len(rows)==len(x)==96 and len({r['sha256'] for r in rows})==96
    train=np.array([r['role']=='train' for r in rows]);assert train.sum()==64
    assert sum(r['role']=='development' for r in rows)==32
    y=np.array([r['target'] for r in rows]);h=fit_head(x[train],y[train])
    scores=predict(h,x)
    predictions=[dict(sample_id=r['sample_id'],sha256=r['sha256'],label='abnormal' if r['target'] else 'normal',
                      role=r['role'],target=r['target'],score=float(s),prediction=int(s>=0)) for r,s in zip(rows,scores)]
    h={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in h.items()}
    write(out/'head.json',h);write(out/'predictions.json',predictions)
    metrics={}
    for role in ('train','development'):
        subset=[r for r in predictions if r['role']==role]
        metrics[role]={**ranking(subset),'n':len(subset),'accuracy':sum(r['prediction']==r['target'] for r in subset)/len(subset)}
    write(out/'result.json',{'status':'PASS','metrics':metrics,'fit_iterations':h['iterations'],
                            'gradient_max':h['gradient_max'],'feature_sha256':hashlib.sha256((origin/'features.npy').read_bytes()).hexdigest(),
                            'scope':'train_fit_and_reused_development_screen_only','number_of_heads':1,'hyperparameter_searches':0})
    print(json.dumps(metrics))


if __name__=='__main__':main()
