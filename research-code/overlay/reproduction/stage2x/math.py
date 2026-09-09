"""Producer float64 response reductions; no training policy changes."""
import math
from statistics import mean
import torch

def response(maps,label):
    gap=maps['abnormal_raw'].double()-maps['normal_raw'].double()
    assert gap.ndim==3 and gap.shape[0]==1 and gap.shape[-1]==1024
    z=(gap/(1+gap.abs())).mean(1)*math.log(1024)
    logw=torch.log_softmax(z,-1);w=logw.exp();entropy=-(w*logw).sum(-1)
    score=(torch.logsumexp(z,-1)-math.log(1024))/math.log(1024)
    loss=torch.nn.functional.softplus(score*(-1 if label=='abnormal' else 1))
    return dict(log_support=float(entropy[0]),top11=float(w.topk(11,dim=-1).values.sum()),score=float(score[0]),lce_loss=float(loss[0]))

def contrast(values):
    out={}
    for metric in values['00']:
        f={k:values[k][metric] for k in values};total=f['11']-f['00']
        direct=.5*((f['10']-f['00'])+(f['11']-f['01']));clip=.5*((f['01']-f['00'])+(f['11']-f['10']))
        assert abs(direct+clip-total)<=1e-12
        out[metric]=dict(total=total,direct=direct,clip=clip,clip_g0=f['01']-f['00'],clip_g1=f['11']-f['10'])
    return out

def decision(reports):
    vals=list(reports.values());assert len(vals)==10
    means=[r['means'] for r in vals]
    if all(r['policy_raw_equal'] for r in vals):main='NO_RESOLVED_LOCAL_POLICY_RESPONSE'
    elif all(r['log_support']['total']<0 and r['top11']['total']>0 for r in means):main='CONSISTENT_LOCAL_CONCENTRATING_RESPONSE'
    elif all(r['log_support']['total']>0 and r['top11']['total']<0 for r in means):main='CONSISTENT_LOCAL_DIFFUSING_RESPONSE'
    else:main='STATE_DEPENDENT_OR_MIXED_LOCAL_RESPONSE'
    return dict(policy_response=main,clipping_response='CLIP_MEDIATED_ATTENTION_RESPONSE' if any(not r['clip_raw_equal'] for r in vals) else 'NO_RESOLVED_CLIP_ATTENTION_RESPONSE')

def summarize(rows):
    assert len(rows)==3
    return {metric:{part:mean(row[metric][part] for row in rows) for part in rows[0][metric]} for metric in rows[0]}

def main():
    import argparse,time
    from pathlib import Path
    from reproduction.stage2x.common import read,write,ANCHORS
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--attempts',type=Path);a=p.parse_args()
    attempts=read(a.attempts or a.output/'attempts.json');assert set(attempts)==set(ANCHORS)
    reports={}
    for name in ANCHORS:
        r=read(a.output/name/attempts[name]/'result.json');assert r['status']=='PASS'
        rows=[];policy_equal=clip_equal=True
        for j in range(3):
            cases={k:r['rows'][r['aliases'][k]][j] for k in r['aliases']}
            effects=contrast({k:x['analysis'] for k,x in cases.items()})
            policy_equal &= cases['11']['maps_sha256']==cases['00']['maps_sha256']
            clip_equal &= cases['01']['maps_sha256']==cases['00']['maps_sha256'] and cases['11']['maps_sha256']==cases['10']['maps_sha256']
            rows.append(dict(sample_id=cases['00']['original']['sample_id'],label=cases['00']['original']['label'],effects=effects))
        reports[name]=dict(rows=rows,means=summarize([r['effects'] for r in rows]),policy_raw_equal=policy_equal,clip_raw_equal=clip_equal)
    out=dict(status='PASS',reports=reports,decision=decision(reports),attempts=attempts,finished_epoch=time.time())
    write(a.output/'response-result.json',out);print(out['decision'])
if __name__=='__main__':main()
