"""Independent raw NumPy response/LM reconstruction; no producer math imports."""
import argparse,gc,math,signal,time
import numpy as np
from reproduction.stage2x.common import *
from reproduction.stage2t.verify import check_spatial

def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
    else:assert a==b,(a,b)

def measure(raw,label):
    maps=raw['maps'];gap=maps['abnormal_raw'].double().numpy()-maps['normal_raw'].double().numpy()
    assert gap.ndim==3 and gap.shape[0]==1 and gap.shape[-1]==1024 and np.isfinite(gap).all()
    z=np.mean(gap/(1+np.abs(gap)),axis=1)*np.log(1024.)
    maximum=z.max(axis=-1,keepdims=True);centered=z-maximum;lse=np.log(np.exp(centered).sum(axis=-1,keepdims=True))
    logw=centered-lse;w=np.exp(logw);entropy=-np.sum(w*logw,axis=-1)
    score=float(((maximum+lse-np.log(1024.))/np.log(1024.)).item())
    logits=raw['lm_logits'].float().numpy().astype(np.float64);target=raw['targets'].numpy()
    assert logits.ndim==2 and len(logits)==len(target)>0 and np.isfinite(logits).all()
    assert np.all((target>=0)&(target<logits.shape[1]))
    mx=logits.max(1);lm=float(np.mean(mx+np.log(np.exp(logits-mx[:,None]).sum(1))-logits[np.arange(len(target)),target]))
    return dict(log_support=float(entropy.item()),top11=float(np.sort(w,axis=-1)[...,-11:].sum()),score=score,
        lce_loss=float(np.logaddexp(0.,score*(-1 if label=='abnormal' else 1))),lm_loss=lm)

def effects(values):
    out={}
    for metric in values['00']:
        x0,x1,x2,x3=(values[k][metric] for k in ('00','01','10','11'))
        total=x3-x0;direct=((x2-x0)+(x3-x1))/2;clip=((x1-x0)+(x3-x2))/2
        assert abs(total-direct-clip)<=1e-12
        out[metric]=dict(total=total,direct=direct,clip=clip,clip_g0=x1-x0,clip_g1=x3-x2)
    return out

def verdict(reports):
    rows=list(reports.values());assert len(rows)==10
    if all(r['policy_raw_equal'] for r in rows):policy='NO_RESOLVED_LOCAL_POLICY_RESPONSE'
    elif all(r['means']['log_support']['total']<0 and r['means']['top11']['total']>0 for r in rows):policy='CONSISTENT_LOCAL_CONCENTRATING_RESPONSE'
    elif all(r['means']['log_support']['total']>0 and r['means']['top11']['total']<0 for r in rows):policy='CONSISTENT_LOCAL_DIFFUSING_RESPONSE'
    else:policy='STATE_DEPENDENT_OR_MIXED_LOCAL_RESPONSE'
    return dict(policy_response=policy,clipping_response='NO_RESOLVED_CLIP_ATTENTION_RESPONSE' if all(r['clip_raw_equal'] for r in rows) else 'CLIP_MEDIATED_ATTENTION_RESPONSE')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    assert not torch.cuda.is_available();torch.set_num_threads(4)
    def expired(*_):raise TimeoutError('CPU verification two-hour ceiling')
    signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
    started=time.time();guard=WriteGuard(a.output);ins=Inputs();source=source_identity(a.code_commit,ins)
    plan=read(a.output/'plan.json');assert plan['code_commit']==a.code_commit and source==read(a.output/'source.json')
    producer=read(a.output/'response-result.json');assert producer['status']=='PASS';attempts=producer['attempts'];assert set(attempts)==set(ANCHORS)
    reports={};raw_files={};forwards=0;maximum_lm_error=0.
    for name,cfg in ANCHORS.items():
        out=a.output/name/attempts[name];r=read(out/'result.json');assert r['status']=='PASS' and r['code_commit']==a.code_commit
        assert r['real_updates']==r['backward']==0 and r['complete_anchor_state_restored'] and not r['historical_write_violations']
        entry=read(out/'entry.json');assert entry['initial_full_exact'] and entry['anchor_full_exact'] and entry['source_sha256']==canonical_json_sha256(source)
        cfg,old,oldmaps,before,runtime,rng,_=state(ins,name)
        assert entry['master_sha256']==old['master_before'] and entry['runtime_sha256']==old['runtime_before'] and entry['rng_sha256']==old['rng']['before_block']
        reps,aliases=variants(ins.wjs('corners/'+name+'.json'));assert aliases==r['aliases']
        gate=read(out/'baseline-gate.json');assert gate['status']=='PASS' and all(gate['gates'].values())
        assert read(out/'monitor-summary.json')['peak_memory_used_mib']<=22500
        audit=read(out/'file-open-audit.json');assert audit['status']=='SUCCESS' and not audit['non_allowlisted_paths']
        assert len(audit['allowed_paths'])==3 and set(audit['allowed_paths']).issubset(plan['allowed_images'])
        allrows={};allmaps={}
        for key in ('baseline',*reps):
            vi=read(out/f'{key}-state.json');assert vi==r['variants'][key]
            expected={n:v.bfloat16() for n,v in before.items()} if key=='baseline' else candidate(ins,name,key,before)
            assert vi['trainable_sha256']==state_digest(expected)
            assert vi['runtime_entry_sha256']==old['runtime_before'] and vi['input_list_sha256']==old['inputs_sha256']
            assert vi['rng_before']==old['rng']['before_block'] and vi['rng_after']==old['rng']['after_block']
            new=[];maps=[]
            for j,row in enumerate(r['rows'][key]):
                assert row==read(out/f'{key}-image{j}.json')
                path=out/f'{key}-image{j}.pt.gz';assert str(path)==row['raw']['file'] and sha256_file(path)==row['raw']['sha256']
                raw=load_raw(path);raw_files[str(path)]=row['raw']['sha256'];assert state_digest(raw['maps'])==row['maps_sha256']
                label=old['images'][j]['label'];assert all(row['original'][k]==old['images'][j][k] for k in ('sample_id','sha256','label','exposure_id'))
                assert row['input_sha256']==vi['inputs'][j]==r['rows']['baseline'][j]['input_sha256']
                if key=='baseline':assert row['original']==old['images'][j] and state_digest(raw['maps'])==state_digest(oldmaps[j])
                check_spatial(raw['maps'],row['original']);observed=measure(raw,label);close(observed,row['analysis'])
                error=abs(observed['lm_loss']-row['original']['lm_loss']);maximum_lm_error=max(maximum_lm_error,error)
                assert math.isclose(observed['lm_loss'],row['original']['lm_loss'],rel_tol=2e-6,abs_tol=2e-6)
                new.append(observed);maps.append(raw['maps']);forwards+=1;del raw
            allrows[key]=new;allmaps[key]=maps;del expected
        paired=[];policy_equal=clip_equal=True
        for j in range(3):
            values={k:allrows[aliases[k]][j] for k in aliases};mapped={k:state_digest(allmaps[aliases[k]][j]) for k in aliases}
            policy_equal &= mapped['11']==mapped['00'];clip_equal &= mapped['01']==mapped['00'] and mapped['11']==mapped['10']
            paired.append(dict(sample_id=old['images'][j]['sample_id'],label=old['images'][j]['label'],effects=effects(values)))
        means={metric:{part:float(np.mean([x['effects'][metric][part] for x in paired])) for part in paired[0]['effects'][metric]} for metric in paired[0]['effects']}
        reports[name]=dict(rows=paired,means=means,policy_raw_equal=policy_equal,clip_raw_equal=clip_equal);close(reports[name],producer['reports'][name])
        del before,runtime,allrows,allmaps,oldmaps;gc.collect();print(json.dumps(dict(verified_anchor=name,forwards=forwards)),flush=True)
    assert forwards==102 and len(raw_files)==102
    decision=verdict(reports);assert decision==producer['decision']
    budget=read(a.output/'budget.json');assert budget['real_updates']==budget['backward']==0
    assert 102<=budget['counts']['forward']<=156 and 10<=budget['counts']['process']<=14
    assert producer['finished_epoch']-budget['started_epoch']<=3600
    write(a.output/'verification.json',dict(status='PASS',decision=decision,reports=reports,accepted_forwards=forwards,raw_files=raw_files,
        budget=budget['counts'],real_updates=0,backward=0,development_evaluations=0,protected_access=0,code_commit=a.code_commit,
        source_sha256=canonical_json_sha256(source),producer_sha256=sha256_file(a.output/'response-result.json'),
        maximum_lm_error=maximum_lm_error,prior_inputs_checked=ins.checked,candidate_inputs_checked=ins.wchecked,
        seconds=time.time()-started,scope='Fixed-input immediate response at ten measured states; not a training trajectory or generalization study'))
    print(json.dumps(dict(status='PASS',decision=decision),indent=2))
if __name__=='__main__':main()
