"""Independent CPU token, bridge and convex-head verification, no producer maths."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw(p):
    with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def metrics(rows):
    pos=[r['score'] for r in rows if r['target']==1];neg=[r['score'] for r in rows if r['target']==0]
    auc=sum((Fraction(1) if p>n else Fraction(1,2) if p==n else Fraction(0)) for p in pos for n in neg)/(len(pos)*len(neg))
    ap=sum((Fraction(sum(q>=p for q in pos),sum(r['score']>=p for r in rows)) for p in pos),Fraction(0))/len(pos)
    return {'auroc':float(auc),'average_precision':float(ap),'n':len(rows),
            'accuracy':sum(r['prediction']==r['target'] for r in rows)/len(rows)}


def cabg_score(abnormal, normal):
    gap=np.asarray(abnormal,dtype=np.float32)-np.asarray(normal,dtype=np.float32)
    assert gap.shape==(1,4,1024) and np.isfinite(gap).all()
    pooled=np.mean(gap/(1+np.abs(gap)),axis=1,dtype=np.float32)
    tau=1/math.log(1024)  # Formal v1.2 temperature, not an assumed unit value.
    scaled=pooled.astype(np.float64)/tau;m=float(np.max(scaled))
    return tau*(m+math.log(float(np.exp(scaled-m).sum()))-math.log(1024))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    root=a.campaign;plan=read(root/'plan.json');repo=Path(__file__).resolve().parents[2]
    checker_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    checker_path=str(Path(__file__).resolve().relative_to(repo))
    assert sha(__file__)==hashlib.sha256(subprocess.check_output(['git','show',checker_commit+':'+checker_path],cwd=repo)).hexdigest()
    for name,h in plan['sources'].items():
        if name!=checker_path:assert sha(repo/name)==h,name
        assert hashlib.sha256(subprocess.check_output(['git','show',plan['code_commit']+':'+name],cwd=repo)).hexdigest()==h
    tokenizer=AutoTokenizer.from_pretrained('/home/checkpoints/Lingshu-7B')
    control=Path('/home/Medic-AD/outputs/cabg-mil-v1.2')
    inventories={}
    for campaign,expected in [('initialization-lce-persistence-v1','25e5f03ab27ac4febc60cd27b16438a209b5567df4244a039423cfc14e6f2e90'),
                              ('lm-only-baseline-v1','0b79305794b3cb57c945671387ef13eb6b0e73cbc490bb1e9854ca535df27012')]:
        path=control/campaign.replace('-v1','-control-v1')/'artifact-inventory-v1.json'
        assert sha(path)==expected
        inventories[campaign]={e['path']:e for e in read(path)['entries'] if e['root']==campaign}
    expected_ids=[r['sample_id'] for r in plan['development_check']]
    contexts={};answers={};maximum_score_error=0.
    for name,cfg in plan['phases'].items():
        out=root/name;result=read(out/'result.json');rows=read(out/'observations.json')
        assert result['status']=='PASS' and result['optimizer_updates']==0 and result['no_backward']
        assert read(out/'parameters-before.json')['parameters']==read(out/'parameters-after.json')['parameters']
        opened=read(out/'file-open-audit.json')
        assert opened['status']=='SUCCESS' and not opened['non_allowlisted_paths']
        init_name=f's{cfg["seed"]}-prefix/full-initial-state.json'
        init_path=control/'initialization-lce-persistence-v1'/init_name
        assert sha(init_path)==inventories['initialization-lce-persistence-v1'][init_name]['sha256']
        expected_parameters={r['name']:r for r in read(init_path)['parameters']}
        before={r['name']:r for r in read(out/'parameters-before.json')['parameters']}
        if cfg['mode']=='features':
            assert before==expected_parameters
            rep=read(out/'representation.json');assert rep['no_lce_or_lm_adapters_loaded'] is True
            vpt=[r for r in rep['initial_trainable']['parameters'] if 'deep_prompt_embeddings' in r['name']]
            assert len(vpt)==4
            assert all(r['dtype']=='bfloat16' and r['tensor_sha256']==hashlib.sha256(bytes(r['elements']*2)).hexdigest() for r in vpt)
            continue
        assert [r['sample_id'] for r in rows]==expected_ids
        seed,arm=cfg['seed'],cfg['arm']
        campaign='lm-only-baseline-v1' if arm=='none' else 'initialization-lce-persistence-v1'
        oldname=f"eval-s{seed}-"+{'none':'lm_only24','full':'dynamic24','early':'lce_off24'}[arm]
        oldroot=control/campaign/oldname
        historical=read(oldroot/'predictions.json')
        entries=inventories[campaign]
        cp=read(out/'checkpoint.json');cp_path=Path(cp['path']);cp_rel=str(cp_path.relative_to(control/campaign))
        assert sha(cp_path)==entries[cp_rel]['sha256']==cp['sha256']
        payload=torch.load(cp_path,map_location='cpu',weights_only=False)
        assert payload['sampler_state']['cursor']==24 and len(payload['adapter_state'])==21
        for parameter,tensor in payload['adapter_state'].items():
            assert tensor.dtype==torch.bfloat16
            expected_parameters[parameter]={**expected_parameters[parameter],
                'sha256':hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        assert before==expected_parameters
        del payload
        old_predictions=entries[oldname+'/predictions.json'];assert sha(oldroot/'predictions.json')==old_predictions['sha256']
        correct=invalid=0
        for r in rows:
            index=r['bridge_index'];old=historical[index];assert old['sha256']==r['sha256']
            assert r['target']==int(old['label']=='abnormal')
            bridge=raw(out/f'bridge-{index:02}.pt.gz')
            oldpath=oldroot/f'image{index:02}.pt.gz'
            assert sha(oldpath)==entries[oldname+f'/image{index:02}.pt.gz']['sha256']
            assert equal(bridge,raw(oldpath))
            score=cabg_score(bridge['maps']['abnormal_raw'].float().numpy(),bridge['maps']['normal_raw'].float().numpy())
            error=abs(score-old['score']);maximum_score_error=max(maximum_score_error,error);assert error<2e-6
            context=tokenizer.decode(r['input_ids'],skip_special_tokens=False)
            assert context==r['context'] and context.endswith('<|im_start|>assistant\n')
            assert context.count('<|im_start|>user\n')==1 and context.count('<|im_start|>assistant\n')==1
            assert 'Is there any anomaly in the image?\nAnswer the question using a single word or phrase.' in context
            text=tokenizer.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert text==r['text'] and 0<len(r['output_ids'])<=16
            clean=text.strip().lower();match=re.fullmatch('(yes|no)[.!]?',clean)
            parsed=int(match[1]=='yes') if match else None
            assert parsed==r['parsed'];invalid+=parsed is None
            correct+=parsed is not None and parsed==r['target']
            identity=(r['input_ids'],r['pixel_sha256'],r['image_grid_thw'])
            if r['sample_id'] in contexts:assert identity==contexts[r['sample_id']]
            contexts[r['sample_id']]=identity
            assert r['generation_pixels_equal_bridge']==(r['pixel_sha256']==r['bridge_pixel_sha256'])
        answers[name]={'n':4,'correct':int(correct),'invalid':int(invalid),'accuracy':correct/4}
    out=root/'features';rows=read(out/'observations.json');x=np.load(out/'features.npy',allow_pickle=False).astype(np.float64)
    assert len(rows)==len(x)==96 and np.isfinite(x).all()
    assert [(r['sample_id'],r['sha256'],r['role']) for r in rows]==[(r['sample_id'],r['sha256'],r['role']) for r in plan['feature_rows']]
    assert all(r['target']==int(p['scout_label']=='abnormal') for r,p in zip(rows,plan['feature_rows']))
    for r in rows:
        if r['sample_id'] in contexts:assert (r['input_ids'],r['pixel_sha256'],r['image_grid_thw'])==contexts[r['sample_id']]
    train=np.array([r['role']=='train' for r in rows]);assert train.sum()==64
    y=np.array([r['target'] for r in rows]);h=read(root/'linear-head/head.json');pred=read(root/'linear-head/predictions.json')
    assert len(pred)==96 and h['training_n']==64
    mean=np.mean(x[train],axis=0);sd=np.std(x[train],axis=0);sd[sd<1e-6]=1
    np.testing.assert_array_equal(h['mean'],mean);np.testing.assert_array_equal(h['scale'],sd)
    z=(x-mean)/sd;w=np.array(h['weight']);scores=np.sum(z*w,axis=1)+h['bias']
    np.testing.assert_allclose(scores,[r['score'] for r in pred],atol=1e-10,rtol=1e-10)
    assert all(p['sample_id']==r['sample_id'] and p['target']==r['target'] and p['role']==r['role'] and p['prediction']==int(s>=0) for p,r,s in zip(pred,rows,scores))
    yt=y[train];weight=np.array([.5/sum(yt==v) for v in yt])
    prob=np.array([1/(1+math.exp(-s)) if s>=0 else math.exp(s)/(1+math.exp(s)) for s in scores[train]])
    residual=weight*(prob-yt);gradient=np.sum(z[train]*residual[:,None],axis=0)+w/64
    gradient_max=max(float(np.max(np.abs(gradient))),abs(float(residual.sum())))
    assert gradient_max<=1e-6 and h['penalty']==1/64
    linear={role:metrics([r for r in pred if r['role']==role]) for role in ('train','development')}
    reported=read(root/'linear-head/result.json')
    for role in linear:
        for key,value in linear[role].items():assert abs(reported['metrics'][role][key]-value)<1e-12
    assert reported['feature_sha256']==sha(out/'features.npy')
    b=read(root/'budget.json')
    assert b['counts']=={'load':7,'bridge':24,'generation':24,'feature':96}
    assert b['model_wall_seconds']<5400
    result={'status':'PASS','code_commit':plan['code_commit'],'checker_commit':checker_commit,'checker_sha256':sha(__file__),'source_files_verified':len(plan['sources']),
            'answer_pipeline':answers,'score_reconstruction_max_error':maximum_score_error,
            'simple_baseline':linear,'baseline_matched_four_correct':sum(r['prediction']==r['target'] for r in pred if r['sample_id'] in expected_ids),'convex_gradient_max':gradient_max,'counts':b['counts'],
            'independent_cases':0,'scientific_scope':'development_pipeline_and_baseline_screen_only',
            'independence':'Separate CPU token/numerical verification, not independent patient validation'}
    with (root/'independent-verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
