"""Independent NumPy reconstruction of the train-only fixed coefficient rule."""
import argparse,gc,gzip,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def norm(row):return math.sqrt(sum(float(np.sum(v.astype(np.float64)**2)) for v in row.values()))
def rms(rows):return math.sqrt(.5*norm(rows[0])**2+.25*norm(rows[1])**2+.25*norm(rows[2])**2+1e-16)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;results={};maxrel=0.
    for seed in [43,44]:
        out=root/'debug'/f'calibration-s{seed}-v1';expected=json.loads((out/'summary.json').read_text());values=[]
        for step in range(1,9):
            record=json.loads((out/f'block{step}.json').read_text());raw=Path(record['raw']['path']);assert sha(raw)==record['raw']['sha256']
            with gzip.open(raw,'rb') as f:data=torch.load(f,map_location='cpu',weights_only=False)
            assert [r['label'] for r in data['images']]==['normal','abnormal','abnormal']
            aux=[{n:v.float().numpy() for n,v in row.items()} for row in data['lce']]
            names=set(aux[0]);lm=[{n:row[n].float().numpy() for n in names} for row in data['lm']]
            gl={n:(lm[0][n].astype(float)+lm[1][n]+lm[2][n])/3 for n in names}
            ga={n:.5*aux[0][n].astype(float)+.25*aux[1][n]+.25*aux[2][n] for n in names}
            actual=min(.1*rms(lm)/(rms(aux)+1e-8),.2*norm(gl)/(norm(ga)+1e-8));values.append(actual)
            rel=abs(actual-record['coefficient'])/max(abs(actual),1e-12);maxrel=max(rel,maxrel);assert rel<2e-6,(seed,step,rel)
            del data,aux,lm,gl,ga;gc.collect()
        chosen=statistics.median(values);assert abs(chosen-expected['coefficient'])/chosen<2e-6
        results[str(seed)]={'coefficient_producer':expected['coefficient'],'coefficient_recomputed':chosen,'blocks':8}
    result={'status':'PASS','scope':'separate NumPy RMS/aggregation/median implementation from raw BF16 gradient bodies','seeds':results,'maximum_relative_error':maxrel,'fixed_relative_tolerance':2e-6,'new_updates':0,'heldout_used':False}
    with (root/'debug/calibration-independent-verification-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
