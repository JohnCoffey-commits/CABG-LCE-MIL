"""Separate NumPy/float64 reconstruction; never imports producer math.py."""
import argparse,gc,hashlib,json,math,time
import numpy as np
from reproduction.stage2w.common import *
from reproduction.stage2t.verify import check_spatial

def equal(a,b,atol=1e-12):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k],atol)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y,atol)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=atol),(a,b)
    else:assert a==b,(a,b)

def independent_gradients(d,r):
    labels=[x['label'] for x in r['images']];assert labels.count('normal')==1 and labels.count('abnormal')==2
    lm={n:torch.stack([row[n].float() for row in d['lm']]).mean(0) for n in TRAINABLE_NAMES}
    classes={label:[i for i,x in enumerate(labels) if x==label] for label in ('normal','abnormal')}
    aux={n:(torch.stack([d['lce'][i][n].float() for i in classes['normal']]).mean(0)+torch.stack([d['lce'][i][n].float() for i in classes['abnormal']]).mean(0))*.5 for n in S9_NAMES}
    assert state_digest(lm)==r['aggregate_lm'] and state_digest(aux)==r['aggregate_lce']
    combined={n:lm[n].clone() for n in lm}
    for n in aux:combined[n]=combined[n]+aux[n]*r['controller']['lambda_final']
    scales=[]
    for g in (lm,combined):
        total=torch.tensor(0.,dtype=torch.float32)
        for n in sorted(g):total+=torch.sum(g[n]*g[n])
        scales.append(min(1.,1./(float(torch.sqrt(total))+1e-12)))
    candidates={a+b:{n:g[n]*scales[int(b)] for n in g} for a,g in (('0',lm),('1',combined)) for b in ('0','1')}
    key='00' if r['policy']['arm']=='lce_off' else '11';assert state_digest(candidates[key])==r['applied_sha256']
    for maps,row in zip(d['attention_maps'],r['images']):check_spatial(maps,row)
    return lm,aux,candidates,scales

def numpy_geometry(lm,aux,coefficient):
    out={}
    for label,names in GROUPS.items():
        x2=y2=xy=0.
        for n in names:
            x=lm[n].numpy().astype(np.float64);y=(aux[n].numpy()*np.float32(coefficient)).astype(np.float64) if n in aux else np.zeros_like(x)
            x2+=float(np.sum(x*x));y2+=float(np.sum(y*y));xy+=float(np.sum(x*y))
        den=np.sqrt(x2*y2)
        out[label]=dict(lm_norm=float(np.sqrt(x2)),scaled_lce_norm=float(np.sqrt(y2)),dot=xy,cosine=xy/float(den) if den else None,
            cancellation_ratio=float(np.sqrt(max(0,x2+y2+2*xy))/(np.sqrt(x2)+np.sqrt(y2))) if x2+y2 else None)
    return out

def numpy_attribution(values):
    result={}
    for label,names in GROUPS.items():
        totals=np.zeros(5,dtype=np.float64);maximum=0.
        counts={k:0 for k in ('fp32_policy','bf16_policy','fp32_direct_c0','fp32_direct_c1','bf16_direct_c0','bf16_direct_c1','fp32_clip_g0','fp32_clip_g1','bf16_clip_g0','bf16_clip_g1')}
        for n in names:
            f={k:v[n].numpy().astype(np.float64) for k,v in values.items()}
            effect=f['11']-f['00'];direct=(f['11']-f['01']+f['10']-f['00'])/2;clip=effect-direct
            totals+=np.array([np.sum(effect**2),np.sum(direct**2),np.sum(clip**2),np.sum(direct*effect),np.sum(clip*effect)])
            maximum=max(maximum,float(np.max(np.abs(direct+clip-effect))))
            for x,y,name in [('11','00','policy'),('10','00','direct_c0'),('11','01','direct_c1'),('01','00','clip_g0'),('11','10','clip_g1')]:
                counts['fp32_'+name]+=int(np.count_nonzero(f[x]!=f[y]))
                counts['bf16_'+name]+=int(np.count_nonzero(values[x][n].bfloat16().float().numpy()!=values[y][n].bfloat16().float().numpy()))
        t,a,b,ad,bd=map(float,totals)
        result[label]=dict(total_squared_norm=t,direct_squared_norm=a,clip_squared_norm=b,direct_projection=ad/t if t else None,clip_projection=bd/t if t else None,direct_dot_total=ad,clip_dot_total=bd,closure_max_abs=maximum,**counts)
    return result

def analytic(before,opt,gradients,step,values):
    maximum=0.
    for i,n in enumerate(TRAINABLE_NAMES):
        p=before[n].numpy().astype(np.float64);state=opt['state'].get(i)
        m=state['exp_avg'].numpy().astype(np.float64) if state else np.zeros_like(p);v=state['exp_avg_sq'].numpy().astype(np.float64) if state else np.zeros_like(p)
        for key in values:
            g=gradients[key][n].numpy().astype(np.float64)
            mhat=(.9*m+(1-.9)*g)/(1-.9**step);vhat=(.999*v+(1-.999)*g*g)/(1-.999**step)
            expected=p-opt['param_groups'][0]['lr']*mhat/(np.sqrt(vhat)+1e-8)
            maximum=max(maximum,float(np.max(np.abs(expected-values[key][n].numpy().astype(np.float64)))))
    assert maximum<=2e-7,maximum
    return maximum

def numpy_memory(opt,g,step):
    out={}
    for label,names in GROUPS.items():
        totals=np.zeros(5,dtype=np.float64)
        for n in names:
            i=TRAINABLE_NAMES.index(n);state=opt['state'].get(i);x=g[n].numpy().astype(np.float64)
            oldm=state['exp_avg'].numpy().astype(np.float64) if state else np.zeros_like(x);oldv=state['exp_avg_sq'].numpy().astype(np.float64) if state else np.zeros_like(x)
            divisor=np.sqrt((.999*oldv+(1-.999)*x*x)/(1-.999**step))+1e-8
            hist=-opt['param_groups'][0]['lr']*(.9*oldm/(1-.9**step))/divisor
            cur=-opt['param_groups'][0]['lr']*((1-.9)*x/(1-.9**step))/divisor;total=hist+cur
            totals+=np.array([np.sum(total*total),np.sum(hist*hist),np.sum(cur*cur),np.sum(hist*total),np.sum(cur*total)])
        t,h,c,hp,cp=map(float,totals);out[label]=dict(total_squared_norm=t,memory_squared_norm=h,current_squared_norm=c,memory_projection=hp/t if t else None,current_projection=cp/t if t else None)
    return out

def decode_numpy(before,bits):
    # NumPy ufuncs return a scalar for zero-dimensional inputs; retain ndarray
    # identity before handing the lossless view to torch.from_numpy.
    binary=np.asarray(np.bitwise_xor(before.contiguous().numpy().view(np.int32),bits.contiguous().numpy()),dtype=np.int32)
    return torch.from_numpy(binary.copy().view(np.float32).reshape(tuple(before.shape)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--execution-commit');p.add_argument('--verification-output',type=Path);a=p.parse_args()
    started=time.time()
    if a.execution_commit:
        assert a.verification_output is not None and a.verification_output.is_relative_to(a.output)
        a.verification_output.mkdir(exist_ok=False);guard=AccessGuard(a.output);cpu_deadline();ins=Inputs()
        checker_source=identity(a.code_commit);executed=read(a.output/'source.json');repo=Path(__file__).resolve().parents[2]
        for name,h in executed.items():assert hashlib.sha256(subprocess.check_output(['git','show',a.execution_commit+':'+name],cwd=repo)).hexdigest()==h
        changed=subprocess.check_output(['git','diff','--name-only',a.execution_commit,a.code_commit],cwd=repo,text=True).splitlines()
        assert set(changed)=={'reproduction/stage2w/verify.py','reproduction/stage2w/test_contracts.py'}
        lock=read(a.output/'protocol-lock.json');assert lock['code_commit']==a.execution_commit and lock['protocol_sha256']==sha256_file(Path(__file__).with_name('PROTOCOL.md'))
        write(a.verification_output/'checker-source.json',checker_source)
        write(a.verification_output/'code-bridge.json',dict(status='PASS',execution_commit=a.execution_commit,checker_commit=a.code_commit,changed_paths=changed,execution_source_sha256=canonical_json_sha256(executed),checker_source_sha256=canonical_json_sha256(checker_source),mechanical_change='Zero-dimensional NumPy XOR decode; preserved original producer and failed verifier evidence; no new GPU computation'))
    else:ins,guard=open_existing(a.output,a.code_commit)
    assert not torch.cuda.is_available(),'Verifier must run with CUDA hidden'
    context=read(a.output/'context-result.json');producer=read(a.output/'corners-result.json');pilot=read(a.output/'reference-result.json')
    assert context['status']==producer['status']==pilot['status']=='PASS'
    count=0
    for phase,steps in PHASES.items():
        for step in steps:
            r,d=ins.record(phase,step);lm,aux,gs,scales=independent_gradients(d,r);result=read(a.output/'context'/f'{phase}-b{step}.json')
            assert result['raw_sha256']==r['raw']['sha256'] and result['record_sha256']==r['record_sha256']
            equal(numpy_geometry(lm,aux,r['policy']['shadow_lambda']),result['geometry'],1e-10)
            equal(scales,[result['clip_off'],result['clip_dynamic']]);assert result['images']==r['images']
            count+=1;del d,lm,aux,gs;gc.collect()
    assert count==72
    reports={};maximum=0.
    for name in ANCHORS:
        cfg,r,d,before,opt,refs=anchor(ins,name);lm,aux,gradients,scales=independent_gradients(d,r)
        record=read(a.output/'corners'/f'{name}.json');assert record==producer['reports'][name]
        assert record['input_state_sha256']==state_digest((before,opt)) and record['record_sha256']==r['record_sha256']
        values={}
        for key in ('00','01','10','11'):
            info=record['corners'][key];p=a.output/'corners'/f'{name}-{key}.pt.gz';assert sha256_file(p)==info['raw']['sha256'] and str(p)==info['raw']['file']
            raw=load_raw(p)['master_xor'];after={}
            for n in TRAINABLE_NAMES:
                after[n]=decode_numpy(before[n],raw[n])
            assert state_digest(after)==info['master_sha256'] and state_digest({n:t.bfloat16() for n,t in after.items()})==info['bf16_sha256']
            assert state_digest(gradients[key])==info['gradient_sha256'];values[key]=after
            if key in refs:
                assert all(torch.equal(after[n],refs[key]['after'][n]) for n in TRAINABLE_NAMES)
                saved=pilot['reports'][name]['corners'][key]
                assert saved['native_master_and_optimizer_exact'] and all(info[k]==saved[k] for k in ('gradient_sha256','master_sha256','bf16_sha256'))
            del raw
        maximum=max(maximum,analytic(before,opt,gradients,cfg['step'],values))
        attr=numpy_attribution(values);equal(attr,record['attribution'])
        assert all(attr['T12'][k]==0 for k in ('fp32_direct_c0','fp32_direct_c1','bf16_direct_c0','bf16_direct_c1'))
        key='00' if cfg['phase'].endswith('lce_off') else '11';memory=numpy_memory(opt,gradients[key],cfg['step']);equal(memory,record['memory'])
        reports[name]=dict(attribution=attr,memory=memory,clip_off=scales[0],clip_dynamic=scales[1],step=cfg['step'],native_reference_count=len(refs))
        del values,d,before,opt,refs,lm,aux,gradients;gc.collect()
        print(json.dumps(dict(verified_anchor=name,maximum_master_error=maximum)),flush=True)
    fp32=any(r['attribution']['T12']['fp32_policy'] for r in reports.values());bf16=any(r['attribution']['T12']['bf16_policy'] for r in reports.values())
    spill='CLIP_MEDIATED_BF16_SPILLOVER' if bf16 else 'FP32_ONLY_CLIP_SPILLOVER' if fp32 else 'NO_LOCAL_T12_SPILLOVER_AT_REGISTERED_ANCHORS'
    clips=[r['attribution']['T21']['clip_projection'] for r in reports.values()];defined=[x for x in clips if x is not None]
    decision=dict(spillover=spill,dominance='DIRECT_DOMINANT_AT_ALL_ANCHORS' if len(defined)==10 and all(abs(x)<=.1 for x in defined) else 'STATE_DEPENDENT_MIXED',undefined_anchors=10-len(defined))
    assert decision==producer['decision']
    budget=read(a.output/'gpu-budget.json');assert budget['arithmetic_clones']==52 and budget['real_updates']==budget['model_forwards']==0
    expected=[dict(phase=phase,anchor=n,corner=k) for phase in ('reference','corners') for n,cfg in ANCHORS.items() for k in (cfg['references'] if phase=='reference' else ('00','01','10','11'))]
    assert [{k:e[k] for k in ('phase','anchor','corner')} for e in budget['events']]==expected
    assert producer['finished_epoch']-budget['started_epoch']<=1800
    assert max(p['monitor']['peak_memory_used_mib'] for p in (pilot,producer))<=8192
    write((a.verification_output or a.output)/'verification.json',dict(status='PASS',decision=decision,anchors=reports,context_records=count,arithmetic_clones=52,
        float64_master_max_abs=maximum,checked_inputs=ins.checked,access=guard.report(),code_commit=a.code_commit,
        execution_commit=a.execution_commit or a.code_commit,source_sha256=canonical_json_sha256(read(a.output/'source.json')),producer_sha256=sha256_file(a.output/'corners-result.json'),
        reference_sha256=sha256_file(a.output/'reference-result.json'),seconds=time.time()-started,real_updates=0,model_loads=0,image_forwards=0,
        scope='Same-state optimizer subsystem and candidate weight values; no counterfactual model-forward or spatial mediation claim'))
    print(json.dumps(dict(status='PASS',decision=decision),indent=2))
if __name__=='__main__':main()
