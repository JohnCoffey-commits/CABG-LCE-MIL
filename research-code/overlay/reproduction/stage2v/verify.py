"""Independent raw arithmetic, sampled bridge, paired forks and seed-specific verdicts."""
import argparse,gc,json,math
from collections import Counter
from pathlib import Path
from statistics import median
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2m.constants import TRAINABLE_NAMES
from reproduction.stage2q.verify import close
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2t.verify import adam,check_checkpoint,check_spatial,metrics,phenotype
from reproduction.stage2u.verify import reconstruct
from reproduction.stage2v.common import *

def verify_phase_audit(root,phase,source,rows):
    path=root/phase
    assert read(path/'source.json')==source
    assert read(path/'result.json')['status']=='SUCCESS'
    a=read(path/'file-open-audit.json')
    allowed=sorted({str(Path('/home/data/medic-ad/official/med_anomaly')/r['relative_path']) for r in rows})
    assert a['status']=='SUCCESS' and a['allowed_paths']==a['opened_paths']==allowed
    assert not a['non_allowlisted_paths'] and not a['unopened_allowlisted_paths']
    assert a['open_event_count']==len(rows) and a['internal_test_outputs_read']==0
    m=read(path/'monitor-summary.json');assert m['status']=='SUCCESS' and m['peak_memory_used_mib']<=22500
    return dict(unique_images=len(allowed),image_opens=len(rows),peak_mib=m['peak_memory_used_mib'])

def load_record(path,step):
    r=read(path/f'block{step}.json');copy=dict(r);claimed=copy.pop('record_sha256')
    assert canonical_json_sha256(copy)==claimed
    raw=path/f'block{step}.pt.gz';assert sha256_file(raw)==r['raw']['sha256']
    d=load_raw(raw);assert len(d['lm'])==len(d['lce'])==len(d['attention_maps'])==len(r['images'])==3
    return r,d

def verify_train(root,phase,plan,source):
    cfg=PHASES[phase];path=root/phase;manifest=read(root/'train-manifest.json')
    audit=verify_phase_audit(root,phase,source,manifest[cfg['start']*3:cfg['end']*3])
    prov=provenance(plan,phase);assert read(path/'provenance.json')==prov
    assert read(path/'configuration.json')==configuration(cfg['seed'])
    result=read(path/'result.json')
    assert all(result[k]==cfg[k] for k in ('seed','start','end','arm','mode'))
    initial=read(path/'initial-identity.json')
    pp,parent_prov=parent(root,phase,plan)
    if pp:
        payload,pm=load_strict(pp,parent_prov)
        assert all(read(path/'resume-audit.json')['exact'].values())
        assert read(path/'resume-audit.json')['checkpoint_sha256']==pm['checkpoint_sha256']
        assert read(path/'parent-lock.json')['checkpoint_sha256']==pm['checkpoint_sha256']
        before=payload['master_state'];g=payload['optimizer_state']['param_groups'][0]
        moments={n:(payload['optimizer_state']['state'][i]['exp_avg'].double(),payload['optimizer_state']['state'][i]['exp_avg_sq'].double()) for n,i in zip(TRAINABLE_NAMES,g['params'])}
        previous_path=FULL24/'R0' if cfg['mode']=='replay' else root/f"s{cfg['seed']}-prefix"
        previous=read(previous_path/f"block{cfg['start']}.json")
        assert payload['trace_state']['last_record_sha256']==previous['record_sha256']
        if cfg['mode']=='train':
            assert initial==read(previous_path/'initial-identity.json')
            assert read(path/'full-initial-state.json')==read(previous_path/'full-initial-state.json')
    else:
        startpath=path/'start.pt.gz';assert sha256_file(startpath)==read(path/'start.json')['raw']['sha256']
        d=load_raw(startpath);before={n:v.float() for n,v in d['trainable'].items()};moments={}
        assert state_digest(before)==read(path/'start.json')['master_sha256']
        assert rng_state_fingerprint(d['rng'])==initial['rng_fingerprint']
        assert state_digest(d['runtime'])==initial['runtime_sha256']
        opt=read(path/'initial-optimizer.json');assert opt['state']=={}
        group=opt['param_groups'][0]
        assert group['params']==list(range(len(TRAINABLE_NAMES))) and group['betas']==[.9,.999] and group['lr']==1e-4 and group['eps']==1e-8 and group['weight_decay']==0
        previous=dict(record_sha256='0'*64,optimizer_after=state_digest(opt),runtime_after=initial['runtime_sha256'],rng={'after_block':initial['rng_fingerprint']},controller=dict(lm_ema=0.,lce_ema=0.))
        del d
    maximum=moment_max=0.;coefficients=[];checkpoint_checks={};updates=0
    for step in range(cfg['start']+1,cfg['end']+1):
        r,d=load_record(path,step)
        assert r['phase']==phase and r['seed']==cfg['seed']
        assert r['policy']['arm']==cfg['arm'] and r['policy']['frozen_ratio']==0.
        assert r['previous_record_sha256']==previous['record_sha256'] and r['master_before']==state_digest(before)
        assert r['optimizer_before']==previous['optimizer_after'] and r['runtime_before']==previous['runtime_after']
        assert r['rng']['before_block']==previous['rng']['after_block']
        assert r['controller_before']==dict(beta=.9,lm_ema=previous['controller']['lm_ema'],lce_ema=previous['controller']['lce_ema'],valid_blocks=step-1)
        expected=manifest[(step-1)*3:step*3]
        assert [(x['sample_id'],x['sha256'],x['label'],x['exposure_id']) for x in r['images']]==[(x['sample_id'],x['sha256'],x['scout_label'],x['scout_exposure_id']) for x in expected]
        assert r['inputs_sha256']==read(FULL24/'R0'/f'block{step}.json')['inputs_sha256']
        d['updated_master']=decode_master(before,d['master_xor']);assert state_digest(d['updated_master'])==r['master_after']
        d['applied']=reconstruct(d,r,0.);maximum=max(maximum,adam(d,r,before,moments))
        if cfg['mode']=='replay' or (cfg['arm']=='lce_off' and step==13):
            reference_path=FULL24/'R0' if cfg['mode']=='replay' else root/f"s{cfg['seed']}-dynamic"
            ref,rd=load_record(reference_path,step)
            for k in ('lm','lce','attention_maps'):assert state_digest(d[k])==state_digest(rd[k])
            for k in ('images','inputs_sha256','rng','controller_before','controller','optimizer_before','runtime_before','master_before'):
                assert r[k]==ref[k],(phase,step,k)
            assert all(read(path/f'block{step}-reference-gate.json').values())
            if cfg['mode']=='replay':
                assert state_digest(d['applied'])==state_digest(rd['applied'])
                assert state_digest(d['updated_master'])==state_digest(rd['updated_master'])
                assert r['master_after']==ref['master_after'] and r['runtime_after']==ref['runtime_after']
                assert r['actual_master_after']==r['master_before'] and r['optimizer_after']==r['optimizer_before'] and r['real_updates']==0
            else:assert r['applied_sha256']!=ref['applied_sha256'],'Intervention did not engage'
            del rd
        if cfg['mode']=='train':
            assert r['real_updates']==1 and r['actual_master_after']==r['master_after']
            if step in (8,12,20,24):
                ck=check_checkpoint(path/f'block{step}-checkpoint',r,d,moments,prov)
                assert ck['scheduler']['last_epoch']==step
                moment_max=max(moment_max,ck['moment_max_relative_l2']);checkpoint_checks[str(step)]=ck['moment_max_relative_l2']
        coefficients.append(dict(block=step,**r['policy'],cap=r['controller']['lambda_cap'],preclip_lce_norm=r['gradient']['scaled_auxiliary_shared_norm'],postclip_lce_norm=r['gradient']['scaled_auxiliary_shared_norm']*r['gradient']['clip_scale']))
        updates+=r['real_updates'];before=d['updated_master'];previous=r;del d;gc.collect()
    full0,full1=read(path/'full-initial-state.json'),read(path/'full-final-state.json')
    frozen=lambda v:[r for r in v['parameters'] if r['name'] not in TRAINABLE_NAMES]
    assert frozen(full0)==frozen(full1)
    if cfg['mode']=='replay':assert full1==read(pp.with_name('full-state.json'))
    assert updates==result['real_updates']==(cfg['end']-cfg['start'] if cfg['mode']=='train' else 0)
    return dict(status='PASS',audit=audit,updates=updates,adam_max_abs=maximum,moment_max_relative_l2=moment_max,checkpoints=checkpoint_checks,coefficients=coefficients,initial_identity=initial)

def verify_eval(root,phase,plan,source):
    path=root/phase;expected=read(root/'eval-manifest.json')
    audit=verify_phase_audit(root,phase,source,expected)
    checkpoint,prov=eval_checkpoint(root,phase,plan)
    result=read(path/'result.json');assert result['count']==32 and result['updates']==0 and result['protected_internal_test_opened']==0
    assert result['checkpoint_sha256']==sha256_file(checkpoint)
    assert read(path/'provenance.json')==prov
    assert read(path/'full-initial-state.json')['parameters']==read(path/'full-final-state.json')['parameters']
    rows=read(path/'predictions.json');assert len(rows)==32
    for i,(r,e) in enumerate(zip(rows,expected)):
        assert (r['sample_id'],r['sha256'],r['label'])==(e['sample_id'],e['sha256'],e['scout_label'])
        p=path/f'image{i:02}.pt.gz';assert sha256_file(p)==r['raw']['sha256'];d=load_raw(p)
        check_spatial(d['maps'],r);t=d['lm_terms'];assert t['valid_tokens']==r['valid_tokens']==t['target_logit'].numel()
        close(float((t['logsumexp'].double()-t['target_logit'].double()).mean()),r['lm_loss'],'LM',atol=2e-6,rtol=2e-6)
    summary=metrics(rows)
    for k in ('auroc','average_precision'):close(summary[k],read(path/'metrics.json')[k],k,rtol=1e-12)
    if phase.startswith('bridge'):
        old=read(FULL24/'accepted-v2/eval-R0-b24/predictions.json')
        clean=lambda rows:[{k:v for k,v in r.items() if k!='raw'} for r in rows]
        assert clean(rows)==clean(old),'Bridge endpoint not exact'
    return rows,dict(status='PASS',audit=audit,metrics=summary)

def spatial_outcome(root,phase,summary,steps):
    late=[]
    for step in steps:
        images=read(root/phase/f'block{step}.json')['images']
        late.append(median(r['effective_support'] for r in images)<128 or median(r['top11_mass'] for r in images)>.35)
    state=phenotype(summary,late)
    stable=summary['support_median']>=256 and summary['support_below128']<=8 and summary['top11_above035']<=8 and sum(late)<=1
    return dict(**state,stable=stable)

def decide(per_seed):
    replicated={seed:v['dynamic']['spatial']['concentrated'] and v['lce_off']['spatial']['stable'] for seed,v in per_seed.items()}
    return ('SPATIAL_CONTRAST_REPLICATED_2_OF_2' if len(replicated)==2 and all(replicated.values()) else 'INITIALIZATION_SENSITIVE_OR_NOT_REPLICATED'),replicated

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--stage',choices=('bridge-train','bridge','final'),required=True);a=p.parse_args();torch.set_num_threads(4)
    root=a.campaign;plan,source=read_campaign(root,a.code_commit)
    if a.stage=='bridge-train':
        phases={n:verify_train(root,n,plan,source) for n in ('bridge-b13','bridge-b24')}
        write(root/'bridge-train-verification.json',dict(status='PASS',phases=phases,real_updates=0,code_commit=a.code_commit))
        print('BRIDGE_TRAIN_PASS');return
    assert read(root/'bridge-train-verification.json')['status']=='PASS'
    if a.stage=='bridge':
        _,evaluation=verify_eval(root,'bridge-eval-b24',plan,source)
        write(root/'bridge-verification.json',dict(status='PASS',sampled_replays_exact=True,endpoint_exact=True,evaluation=evaluation,train_verification_sha256=sha256_file(root/'bridge-train-verification.json'),code_commit=a.code_commit))
        print('RESTORED_HOST_BRIDGE_PASS');return
    assert read(root/'bridge-verification.json')['status']=='PASS'
    train={n:verify_train(root,n,plan,source) for n,c in PHASES.items() if c['mode']=='train'}
    per_seed={};evaluations={};paired={}
    for seed in SEEDS:
        _,parent_eval=verify_eval(root,f'eval-s{seed}-parent12',plan,source)
        parent_eval['spatial']=spatial_outcome(root,f's{seed}-prefix',parent_eval['metrics'],range(9,13))
        seed_report={'parent12':parent_eval};rows={}
        for arm in ('dynamic','lce_off'):
            phase=f'eval-s{seed}-{arm}24';rows[arm],ev=verify_eval(root,phase,plan,source)
            ev['spatial']=spatial_outcome(root,f's{seed}-{arm}',ev['metrics'],range(21,25))
            seed_report[arm]=ev;evaluations[phase]=ev
        per_seed[str(seed)]=seed_report
        paired[str(seed)]=[dict(sample_id=x['sample_id'],support_delta=x['effective_support']-y['effective_support'],score_delta=x['score']-y['score'],top11_delta=x['top11_mass']-y['top11_mass'],lm_loss_delta=x['lm_loss']-y['lm_loss']) for x,y in zip(rows['lce_off'],rows['dynamic'])]
    initial=[train[f's{s}-prefix']['initial_identity'] for s in SEEDS]
    assert initial[0]['trainable_state_sha256']!=initial[1]['trainable_state_sha256']
    assert initial[0]['rng_fingerprint']!=initial[1]['rng_fingerprint']
    states=[read(root/f's{s}-prefix/full-initial-state.json')['parameters'] for s in SEEDS]
    assert [r['name'] for r in states[0]]==[r['name'] for r in states[1]]
    changed=[x['name'] for x,y in zip(*states) if x['sha256']!=y['sha256']]
    budget=read(root/'budget.json');assert budget['counts']==dict(update=72,block=74,eval=224)
    expected=Counter()
    for phase,cfg in PHASES.items():
        for step in range(cfg['start']+1,cfg['end']+1):
            expected['block',f'{phase}:block{step}']+=1
            if cfg['mode']=='train':expected['update',f'{phase}:block{step}']+=1
    for phase in EVALUATIONS:
        for i in range(32):expected['eval',f'{phase}:{i}']+=1
    assert Counter((e['kind'],e['context']) for e in budget['events'])==expected
    ordered=[]
    for phase in ('bridge-b13','bridge-b24'):
        ordered.append(('block',f"{phase}:block{PHASES[phase]['end']}"))
    ordered.extend(('eval',f'bridge-eval-b24:{i}') for i in range(32))
    for phase,cfg in PHASES.items():
        if cfg['mode']=='train':
            for step in range(cfg['start']+1,cfg['end']+1):
                ordered.extend((kind,f'{phase}:block{step}') for kind in ('block','update'))
    for phase in EVALUATIONS:
        if not phase.startswith('bridge'):
            ordered.extend(('eval',f'{phase}:{i}') for i in range(32))
    assert [(e['kind'],e['context']) for e in budget['events']]==ordered
    last_update=max(e['epoch'] for e in budget['events'] if e['kind']=='update')
    assert all(e['epoch']>last_update for e in budget['events'] if e['kind']=='eval' and e['context'].startswith('eval-s'))
    decision,replicated=decide(per_seed)
    out=dict(status='PASS',decision=decision,replicated=replicated,per_seed=per_seed,paired_off_vs_dynamic=paired,
        train_verification=train,initial_parameters_changed_between_seeds=changed,budget=budget['counts'],code_commit=a.code_commit,
        source_sha256=plan['source_sha256'],bridge_verification_sha256=sha256_file(root/'bridge-verification.json'),
        scope='Exploratory fixed-initialization sensitivity only; no seed selection, cutoff tuning, formal effectiveness or generalization claim')
    write(root/'verification.json',out);print(json.dumps(dict(status='PASS',decision=decision,replicated=replicated),indent=2))
if __name__=='__main__':main()
