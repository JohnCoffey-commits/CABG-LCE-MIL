"""Independent raw-evidence verifier. No calls to producer controller/LCE/update math."""
import argparse
import gc
import json
import math
from pathlib import Path
from statistics import mean,median
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2l.checkpoint import load_strict,tensor_inventory
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2p import INITIAL_RNG_SHA256,TRAIN_SHA256
from reproduction.stage2q.verify import compare_map,close,norm32
from reproduction.stage2s.common import state_digest,write
from reproduction.stage2s.verify_single import groups,VPT
from reproduction.stage2t.common import load_raw,BOUNDARIES,source_identity,LIMITS


def read_json(path):return json.loads(Path(path).read_text())

def spatial(maps,label):
    a,n=maps['abnormal_raw'].float(),maps['normal_raw'].float()
    assert a.shape==n.shape and a.shape[0]==1 and a.shape[-1]==1024
    gap=a-n;contrast=gap/(1+gap.abs());average=contrast.mean(1);tau=1/math.log(1024)
    score=tau*(torch.logsumexp(average/tau,-1)-math.log(1024))
    loss=torch.nn.functional.softplus((-1 if label=='abnormal' else 1)*score)
    weights=torch.softmax(average/tau,-1)
    support=(-(weights*weights.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)).exp()
    return {'score':float(score[0]),'lce_loss':float(loss[0]),'effective_support':float(support[0]),
            'top11_mass':float(weights.topk(11,dim=-1).values.sum(-1)[0])}

def check_spatial(maps,row):
    result=spatial(maps,row['label'])
    for k,v in result.items():
        actual=row['lce_score'] if k=='score' and 'lce_score' in row else row[k]
        close(v,actual,'independent raw attention '+k,rtol=2e-6,atol=2e-6)
    return result

def aggregation(d,r):
    labels=[x['label'] for x in r['images']]
    assert labels.count('normal')==1 and labels.count('abnormal')==2
    assert len(d['lm'])==len(d['lce'])==len(d['attention_maps'])==3
    for kind,names in [('lm',TRAINABLE_NAMES),('lce',S9_NAMES)]:
        for g in d[kind]:
            assert set(g)==set(names) and all(v.dtype==torch.bfloat16 and torch.isfinite(v).all() for v in g.values())
    lm={n:torch.stack([g[n].float() for g in d['lm']]).mean(0) for n in TRAINABLE_NAMES}
    lce={n:.5*(torch.stack([d['lce'][i][n].float() for i,l in enumerate(labels) if l=='normal']).mean(0)+
                  torch.stack([d['lce'][i][n].float() for i,l in enumerate(labels) if l=='abnormal']).mean(0)) for n in S9_NAMES}
    assert state_digest(lm)==r['aggregate_lm'] and state_digest(lce)==r['aggregate_lce']
    rms={}
    for k in ('lm','lce'):
        ps=[norm32({n:g[n] for n in S9_NAMES})**2 for g in d[k]]
        rms[k]=math.sqrt(.5*(sum(v for v,l in zip(ps,labels) if l=='normal')+sum(v for v,l in zip(ps,labels) if l=='abnormal')/2)+1e-16)
    b,c=r['controller_before'],r['controller'];step=int(b['valid_blocks'])+1
    assert c['valid_blocks']==step==r['block'] and b['beta']==.9
    for k in ('lm','lce'):
        ema=.9*b[k+'_ema']+(1-.9)*rms[k]
        close(c[k+'_ema'],ema,k+' EMA');close(c['corrected_'+k+'_rms'],ema/(1-.9**step),k+' corrected RMS')
    ln=math.sqrt(sum(float(lm[n].double().square().sum()) for n in S9_NAMES))
    an=math.sqrt(sum(float(lce[n].double().square().sum()) for n in S9_NAMES))
    close(c['lambda_raw'],.1*c['corrected_lm_rms']/(c['corrected_lce_rms']+1e-8),'lambda raw')
    close(c['lambda_cap'],.2*ln/(an+1e-8),'lambda cap');close(c['lambda_final'],min(c['lambda_raw'],c['lambda_cap']),'lambda final')
    combined={n:g.clone() for n,g in lm.items()}
    for n in lce:combined[n].add_(lce[n]*c['lambda_final'])
    close(r['gradient']['clip_scale'],min(1.,1./(norm32(combined)+1e-12)),'clip')
    assert all(torch.equal(combined[n]*r['gradient']['clip_scale'],d['applied'][n]) for n in TRAINABLE_NAMES)
    for m,row in zip(d['attention_maps'],r['images']):check_spatial(m,row)
    return {'aggregation_and_applied_exact':True,'rms_ema_cap_clip_recomputed':True,'raw_spatial_verified':3}


def adam(d,r,before,moments):
    step=r['block'];lr=1e-4*.5*(1+math.cos(math.pi*(step-1)/24))
    close(r['update']['learning_rate_used'],lr,'24-step LR',rtol=1e-12)
    maximum=0.
    for n in TRAINABLE_NAMES:
        g=d['applied'][n].double()
        if n not in moments:moments[n]=(torch.zeros_like(g),torch.zeros_like(g))
        m,v=moments[n];m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
        expected=before[n].double()-lr*(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8)
        maximum=max(maximum,float((expected-d['updated_master'][n].double()).abs().max()))
    assert maximum<=2e-7,maximum
    return maximum

def paired(d,e,r,s,before,other_before,replay=False):
    exact_fields=('inputs_sha256','rng','controller_before','controller','gradient','runtime_before','runtime_after',
                  'optimizer_before','optimizer_devices','aggregate_lm','aggregate_lce','master_before','master_after','images')
    fields={k:r[k]==s[k] for k in exact_fields}
    if not replay:fields['optimizer_after']=r['optimizer_after']==s['optimizer_after']
    raw={kind:all(torch.equal(v,e[kind][i][n]) for i,g in enumerate(d[kind]) for n,v in g.items()) for kind in ('lm','lce')}
    raw['maps']=state_digest(d['attention_maps'])==state_digest(e['attention_maps'])
    differences={'applied':groups(d['applied'],e['applied']),
                 'update':groups({n:d['updated_master'][n]-before[n].float() for n in TRAINABLE_NAMES},
                                 {n:e['updated_master'][n]-other_before[n].float() for n in TRAINABLE_NAMES}),
                 'master':groups(d['updated_master'],e['updated_master'])}
    return {'exact':all(fields.values()) and all(raw.values()) and all(v['exact'] for g in differences.values() for v in g.values()),
            'fields':fields,'per_image_raw':raw,'differences':differences}

def phase_for(root,run,step):return root/('R0' if run==0 else 'R1a' if step<=12 else 'R1b')

def read_block(p,step):
    r=read_json(p/f'block{step}.json');copy=dict(r);claimed=copy.pop('record_sha256')
    assert canonical_json_sha256(copy)==claimed
    path=p/f'block{step}.pt.gz';assert sha256_file(path)==r['raw']['sha256']
    d=load_raw(path)
    assert state_digest(d['updated_master'])==r['master_after']
    return d,r

def check_checkpoint(path,r,d,moments,prov):
    payload,manifest=load_strict(path/'checkpoint.pt',prov)
    step=r['block'];assert payload['sampler_state']['cursor']==payload['trace_state']['completed_blocks']==step
    assert payload['trace_state']['last_record_sha256']==r['record_sha256']
    assert payload['sampler_state']['manifest_sha256']==TRAIN_SHA256
    assert state_digest(payload['master_state'])==r['master_after']
    assert all(torch.equal(payload['adapter_state'][n],d['updated_master'][n].bfloat16()) for n in TRAINABLE_NAMES)
    assert state_digest(payload['optimizer_state'])==r['optimizer_after']
    assert rng_state_fingerprint(payload['rng_state'])==r['rng']['after_block']
    assert payload['controller_state']==dict(beta=.9,lm_ema=r['controller']['lm_ema'],lce_ema=r['controller']['lce_ema'],valid_blocks=step)
    opt=payload['optimizer_state'];g=opt['param_groups'][0]
    assert tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
    close(g['lr'],1e-4*.5*(1+math.cos(math.pi*step/24)),'checkpoint LR',rtol=1e-12)
    max_moment=0.
    for n,i in zip(TRAINABLE_NAMES,g['params']):
        state=opt['state'][i];assert float(state['step'])==step
        for key,expected in zip(('exp_avg','exp_avg_sq'),moments[n]):
            actual=state[key].double();error=float((actual-expected).norm())/max(float(expected.norm()),1e-12)
            max_moment=max(max_moment,error)
    assert max_moment<=2e-6,max_moment
    side=read_json(path/'runtime-state.json');sidepath=path/'runtime-state.pt.gz'
    assert side['checkpoint_sha256']==manifest['checkpoint_sha256'] and side['raw']['sha256']==sha256_file(sidepath)
    assert state_digest(load_raw(sidepath))==r['runtime_after']
    return {'tensor_inventory_sha256':manifest['tensor_inventory_sha256'],'rng':manifest['rng_fingerprint'],
            'scheduler':payload['scheduler_state'],'controller':payload['controller_state'],
            'sampler':payload['sampler_state'],'full_state':read_json(path/'full-state.json'),
            'runtime_devices':side['runtime_devices'],'optimizer_devices':side['optimizer_devices'],
            'moment_max_relative_l2':max_moment}

def metrics(rows):
    ps=[r['score'] for r in rows if r['label']=='abnormal'];ns=[r['score'] for r in rows if r['label']=='normal']
    auc=sum((p>n)+.5*(p==n) for p in ps for n in ns)/(len(ps)*len(ns))
    tp=seen=0;ap=0.
    for score in sorted({r['score'] for r in rows},reverse=True):
        group=[r for r in rows if r['score']==score];positive=sum(r['label']=='abnormal' for r in group)
        tp+=positive;seen+=len(group);ap+=positive/len(ps)*tp/seen
    return {'auroc':auc,'average_precision':ap,'support_median':median(r['effective_support'] for r in rows),
            'support_below128':sum(r['effective_support']<128 for r in rows),'top11_above035':sum(r['top11_mass']>.35 for r in rows),
            'lm_loss_image_mean':mean(r['lm_loss'] for r in rows),
            'lm_loss_token_weighted':sum(r['lm_loss']*r['valid_tokens'] for r in rows)/sum(r['valid_tokens'] for r in rows)}

def phenotype(endpoint,late):
    components={'support_median_below128':endpoint['support_median']<128,
                'at_least16_low_support':endpoint['support_below128']>=16,
                'at_least16_high_top11':endpoint['top11_above035']>=16,
                'at_least2_late_blocks':sum(late)>=2}
    return {'concentrated':all(components.values()),'components':components,'late_failed_blocks':sum(late)}

def verify_eval(root,run,step,manifest):
    path=root/f'eval-R{run}-b{step}';rows=read_json(path/'predictions.json')
    assert len(rows)==32
    for i,(row,expected) in enumerate(zip(rows,manifest)):
        assert (row['sample_id'],row['sha256'],row['label'])==(expected['sample_id'],expected['sha256'],expected['scout_label'])
        rawpath=path/f'image{i:02}.pt.gz';assert sha256_file(rawpath)==row['raw']['sha256'];d=load_raw(rawpath)
        check_spatial(d['maps'],row)
        t=d['lm_terms'];assert t['valid_tokens']==row['valid_tokens']==t['target_logit'].numel()
        value=float((t['logsumexp'].double()-t['target_logit'].double()).mean())
        close(value,row['lm_loss'],'teacher-forced shifted LM loss',atol=2e-6,rtol=2e-6)
    summary=metrics(rows);producer=read_json(path/'metrics.json')
    for k in ('auroc','average_precision'):close(summary[k],producer[k],k,rtol=1e-12)
    result=read_json(path/'result.json');assert result['status']=='SUCCESS' and result['cursor']==step
    ck=phase_for(root,run,step)/f'block{step}-checkpoint/checkpoint.pt.manifest.json'
    assert result['checkpoint_sha256']==read_json(ck)['checkpoint_sha256']
    return rows,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--code-commit',required=True);a=p.parse_args();root=a.campaign;a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    source=source_identity(a.code_commit);assert source==read_json(root/'source.json')
    phases=['R0','R1a','R1b','replay-R0-b13','replay-R0-b24','replay-R1-b24']
    evals=[f'eval-R{i}-b{s}' for i in range(2) for s in (12,24)]
    initial_full=read_json(root/'R0/full-initial-state.json');frozen=lambda d:[v for v in d['parameters'] if v['name'] not in TRAINABLE_NAMES]
    for phase in phases+evals:
        path=root/phase;assert read_json(path/'source.json')==source
        assert read_json(path/'result.json')['status']=='SUCCESS'
        assert read_json(path/'file-open-audit.json')['status']=='SUCCESS'
        assert read_json(path/'monitor-summary.json')['peak_memory_used_mib']<=22500
        if phase in phases:assert read_json(path/'full-initial-state.json')==initial_full
        assert frozen(read_json(path/'full-final-state.json'))==frozen(initial_full)
        if phase in phases[2:]:assert all(read_json(path/'resume-audit.json')['exact'].values())
    starts=[load_raw(root/p/'start.pt.gz') for p in ('R0','R1a')]
    assert state_digest(starts[0]['trainable'])==state_digest(starts[1]['trainable'])
    assert state_digest(starts[0]['runtime'])==state_digest(starts[1]['runtime'])
    assert all(rng_state_fingerprint(s['rng'])==INITIAL_RNG_SHA256 for s in starts)
    before=[{n:v.float() for n,v in s['trainable'].items()} for s in starts];initial=before[0]
    moments=[{},{}];previous=[None,None];exact=True;math_max=0.;moment_max=0.;late=[[],[]];fresh={};resumes={};checkpoint_checks={}
    manifest=read_json(root/'train-manifest.json')
    for step in range(1,25):
        ds=[];rs=[];cs=[]
        for i in range(2):
            phase=phase_for(root,i,step);d,r=read_block(phase,step)
            assert r['real_updates']==1 and r['master_before']==state_digest(before[i])
            expected_rows=manifest[(step-1)*3:step*3]
            assert [(x['sample_id'],x['sha256'],x['label'],x['exposure_id']) for x in r['images']]==[(x['sample_id'],x['sha256'],x['scout_label'],x['scout_exposure_id']) for x in expected_rows]
            if previous[i]:
                pr=previous[i];assert r['previous_record_sha256']==pr['record_sha256']
                assert r['runtime_before']==pr['runtime_after'] and r['optimizer_before']==pr['optimizer_after'] and r['rng']['before_block']==pr['rng']['after_block']
                assert r['controller_before']==dict(beta=.9,lm_ema=pr['controller']['lm_ema'],lce_ema=pr['controller']['lce_ema'],valid_blocks=step-1)
            else:assert r['previous_record_sha256']=='0'*64 and r['controller_before']==dict(beta=.9,lm_ema=0.,lce_ema=0.,valid_blocks=0)
            m=aggregation(d,r);m['adamw_max_abs_error']=adam(d,r,before[i],moments[i]);math_max=max(math_max,m['adamw_max_abs_error'])
            if step in BOUNDARIES:
                c=check_checkpoint(phase/f'block{step}-checkpoint',r,d,moments[i],read_json(phase/'provenance.json'));cs.append(c)
                moment_max=max(moment_max,c['moment_max_relative_l2'])
            write(a.output/f'R{i}-block{step}-math.json',m)
            if step>=21:late[i].append(median(x['effective_support'] for x in r['images'])<128 or median(x['top11_mass'] for x in r['images'])>.35)
            ds.append(d);rs.append(r)
        fresh[str(step)]=paired(*ds,*rs,*before);exact &= fresh[str(step)]['exact']
        write(a.output/f'block{step}-comparison.json',fresh[str(step)])
        if cs:
            exclude={'moment_max_relative_l2'}
            same={k:cs[0][k]==cs[1][k] for k in cs[0] if k not in exclude};assert all(same.values())
            checkpoint_checks[str(step)]=same
        for i in range(2):
            if (i,step) in ((0,13),(0,24),(1,24)):
                rp=root/f'replay-R{i}-b{step}';z,zr=read_block(rp,step);assert zr['real_updates']==0
                aggregation(z,zr)
                comparison=paired(ds[i],z,rs[i],zr,before[i],before[i],replay=True)
                assert zr['optimizer_before']==zr['optimizer_after']
                resumes[f'R{i}-b{step}']=comparison;exact &= comparison['exact'];del z
        if step==24:
            drift={}
            for group,ns in [('T21',TRAINABLE_NAMES),('S9',S9_NAMES),('VPT',VPT)]+[(n,(n,)) for n in VPT]:
                diff=math.sqrt(sum(float((ds[0]['updated_master'][n].double()-ds[1]['updated_master'][n].double()).square().sum()) for n in ns))
                displacement=[math.sqrt(sum(float((d['updated_master'][n].double()-initial[n].double()).square().sum()) for n in ns)) for d in ds]
                drift[group]={'absolute_difference':diff,'24step_displacements':displacement,'relative_to_24step_displacement':diff/max(*displacement,1e-12)}
        before=[d['updated_master'] for d in ds];previous=rs
        print(json.dumps({'verified_block':step,'pair_exact':fresh[str(step)]['exact'],'adam_max':math_max}),flush=True)
        del ds,d,cs;gc.collect()
    ev_manifest=read_json(root/'eval-manifest.json');evaluation={};phenotypes={};eval_exact={}
    for step in (12,24):
        rows=[]
        for i in range(2):
            rr,mm=verify_eval(root,i,step,ev_manifest);evaluation[f'R{i}-b{step}']=mm;rows.append(rr)
            if step==24:phenotypes[f'R{i}']=phenotype(mm,late[i])
        comparable=lambda rows:[{k:v for k,v in r.items() if k!='raw'} for r in rows]
        eval_exact[str(step)]=comparable(rows[0])==comparable(rows[1]);exact &= eval_exact[str(step)]
    budget=read_json(root/'budget.json');assert all(budget['counts'][k]<=v for k,v in LIMITS.items())
    assert budget['counts']==dict(update=48,block=51,eval=128)
    same=phenotypes['R0']['concentrated']==phenotypes['R1']['concentrated']
    result={'decision':'PASS' if exact and same else 'FAIL','exact':bool(exact),'concentration':('REPRODUCED' if phenotypes['R0']['concentrated'] else 'NOT_REPRODUCED') if same else 'DISCORDANT',
            'fresh_step_exact':{k:v['exact'] for k,v in fresh.items()},'resume':resumes,'checkpoint_exact':checkpoint_checks,
            'endpoint_drift':drift,'evaluation':evaluation,'evaluation_exact':eval_exact,'phenotypes':phenotypes,
            'independent_adamw_max_abs_error':math_max,'independent_moment_max_relative_l2':moment_max,
            'budget':budget['counts'],'code_commit':a.code_commit,'source_sha256':canonical_json_sha256(source),
            'scope':'seed42 original64 train/32 development, corrected L4 execution only; no effectiveness or generalization claim'}
    write(a.output/'verification.json',result);print(json.dumps({k:result[k] for k in ('decision','exact','concentration','evaluation','phenotypes')},indent=2))
if __name__=='__main__':main()
