"""Independent raw training/recovery/endpoint checks and fixed-seed baseline verdicts."""
import argparse,gc,json,math
from collections import Counter
from pathlib import Path
from statistics import median
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2q.verify import close
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2t.verify import adam,check_checkpoint,check_spatial,metrics,phenotype
from reproduction.stage2u.verify import reconstruct
from reproduction.stage2z.common import *

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

def start_identity(value):
    # RNG includes a NumPy MT19937 array; use its established exact fingerprint
    # rather than the tensor/JSON state digest used for model/runtime trees.
    assert set(value)=={'trainable','runtime','rng'}
    return dict(trainable=state_digest(value['trainable']),runtime=state_digest(value['runtime']),
        rng=rng_state_fingerprint(value['rng']))

def checker_identity(producer_commit,checker_commit,producer_source):
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==checker_commit
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).strip()
    checked={name:sha256_file(repo/name) for name in producer_source}
    changed=[name for name in checked if checked[name]!=producer_source[name]]
    expected=['reproduction/stage2z/test_contracts.py','reproduction/stage2z/verify.py']
    assert sorted(changed)==(expected if checker_commit!=producer_commit else [])
    diff=subprocess.check_output(['git','diff','--name-status',producer_commit,checker_commit],cwd=repo,text=True).splitlines()
    assert sorted(diff)==['M\t'+name for name in sorted(changed)]
    return dict(commit=checker_commit,source=checked,changed_from_producer=changed,
        checker_sha256=sha256_file(Path(__file__)),producer_commit=producer_commit)

def verify_train(root,phase,plan,source):
    cfg=PHASES[phase];path=root/phase;manifest=read(root/'train-manifest.json');ins=Inputs()
    audit=verify_phase_audit(root,phase,source,manifest[cfg['start']*3:cfg['end']*3])
    prov=provenance(plan,phase);assert read(path/'provenance.json')==prov
    assert read(path/'configuration.json')==configuration(cfg['seed'])
    result=read(path/'result.json')
    assert all(result[k]==cfg[k] for k in ('seed','start','end','arm','mode'))
    initial=read(path/'initial-identity.json')
    assert initial==ins.js(f"s{cfg['seed']}-prefix/initial-identity.json")
    assert read(path/'full-initial-state.json')==ins.js(f"s{cfg['seed']}-prefix/full-initial-state.json")
    pp,parent_prov=parent(root,phase,plan)
    if pp:
        payload,pm=load_strict(pp,parent_prov)
        assert all(read(path/'resume-audit.json')['exact'].values())
        assert read(path/'resume-audit.json')['checkpoint_sha256']==pm['checkpoint_sha256']
        assert read(path/'parent-lock.json')['checkpoint_sha256']==pm['checkpoint_sha256']
        before=payload['master_state'];g=payload['optimizer_state']['param_groups'][0]
        moments={n:(payload['optimizer_state']['state'][i]['exp_avg'].double(),payload['optimizer_state']['state'][i]['exp_avg_sq'].double()) for n,i in zip(TRAINABLE_NAMES,g['params'])}
        previous_path=root/f"s{cfg['seed']}-lm_only"
        previous=read(previous_path/f"block{cfg['start']}.json")
        assert payload['trace_state']['last_record_sha256']==previous['record_sha256']
        if cfg['mode']=='train':
            assert initial==read(previous_path/'initial-identity.json')
            assert read(path/'full-initial-state.json')==read(previous_path/'full-initial-state.json')
    else:
        startpath=path/'start.pt.gz';assert sha256_file(startpath)==read(path/'start.json')['raw']['sha256']
        d=load_raw(startpath);before={n:v.float() for n,v in d['trainable'].items()};moments={}
        assert state_digest(before)==read(path/'start.json')['master_sha256']
        assert start_identity(d)==start_identity(ins.raw(f"s{cfg['seed']}-prefix/start.pt.gz"))
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
        original=ins.js(original_phase(cfg['seed'],step)+f'/block{step}.json')
        assert r['inputs_sha256']==original['inputs_sha256'] and r['rng']==original['rng']
        assert r['policy']['selected_lambda']==0. and r['gradient']['scaled_auxiliary_shared_norm']==0.
        for kind,names in (('lm',TRAINABLE_NAMES),('lce',S9_NAMES)):
            assert all(set(g)==set(names) and all(t.dtype==torch.bfloat16 and torch.isfinite(t).all() for t in g.values()) for g in d[kind])
        d['updated_master']=decode_master(before,d['master_xor']);assert state_digest(d['updated_master'])==r['master_after']
        d['applied']=reconstruct(d,r,0.);maximum=max(maximum,adam(d,r,before,moments))
        if cfg['mode']=='replay' or step==1:
            if cfg['mode']=='replay':
                reference_path=root/f"s{cfg['seed']}-lm_only";ref,rd=load_record(reference_path,step)
            else:ref,rd=ins.record(f"s{cfg['seed']}-prefix",1)
            for k in ('lm','lce','attention_maps'):assert state_digest(d[k])==state_digest(rd[k])
            for k in ('images','inputs_sha256','rng','controller_before','controller','optimizer_before','runtime_before','master_before'):
                assert r[k]==ref[k],(phase,step,k)
            assert all(read(path/f'block{step}-reference-gate.json').values())
            if cfg['mode']=='replay':
                assert state_digest(d['applied'])==ref['applied_sha256']
                assert state_digest(d['applied'])==state_digest(reconstruct(rd,ref,0.))
                assert state_digest(d['updated_master'])==state_digest(decode_master(before,rd['master_xor']))
                assert r['master_after']==ref['master_after'] and r['runtime_after']==ref['runtime_after']
                assert r['actual_master_after']==r['master_before'] and r['optimizer_after']==r['optimizer_before'] and r['real_updates']==0
            else:assert r['applied_sha256']!=ref['applied_sha256'],'Intervention did not engage'
            del rd
        if cfg['mode']=='train':
            assert r['real_updates']==1 and r['actual_master_after']==r['master_after']
            if step in BOUNDARIES:
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
    path=root/phase;expected=read(root/'eval-manifest.json');ins=Inputs();cfg=EVALUATIONS[phase]
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
        if cfg['reference']:
            old=f"eval-s{cfg['seed']}-dynamic24"
            assert state_digest(d)==state_digest(ins.raw(old+f'/image{i:02}.pt.gz'))
    summary=metrics(rows)
    for k in ('auroc','average_precision'):close(summary[k],read(path/'metrics.json')[k],k,rtol=1e-12)
    if cfg['reference']:
        old=ins.js(f"eval-s{cfg['seed']}-dynamic24/predictions.json")
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

def verdict(summaries):
    assert set(summaries)=={'43','44'}
    result={};comparisons={s:{} for s in summaries}
    for arm in ('dynamic','early_only'):
        better=[];no_lower=[]
        for s in ('43','44'):
            r,b=summaries[s][arm],summaries[s]['lm_only'];a=r['auroc']-b['auroc'];p=r['average_precision']-b['average_precision']
            plus=a>0 and p>0;minus=a<=0 and p<=0
            comparisons[s][arm]=dict(delta_auroc=a,delta_average_precision=p,
                pattern='REFERENCE_HIGHER_BOTH' if plus else 'BASELINE_MATCHES_OR_EXCEEDS_BOTH' if minus else 'MIXED_METRICS_OR_TIE')
            better.append(plus);no_lower.append(minus)
        result[arm]='DESCRIPTIVE_LCE_GAIN_2_OF_2' if all(better) else 'LM_ONLY_MATCHES_OR_EXCEEDS_2_OF_2' if all(no_lower) else 'MIXED_OR_METRIC_DEPENDENT_GAIN'
    return result,comparisons

def old_endpoint(ins,seed,arm):
    phase=f'eval-s{seed}-{arm}24';rows=ins.js(phase+'/predictions.json');expected=ins.js('eval-manifest.json')
    assert len(rows)==32
    for i,(r,e) in enumerate(zip(rows,expected)):
        assert (r['sample_id'],r['sha256'],r['label'])==(e['sample_id'],e['sha256'],e['scout_label'])
        d=ins.raw(phase+f'/image{i:02}.pt.gz');assert r['raw']['sha256']==ins.entries[phase+f'/image{i:02}.pt.gz']['sha256']
        check_spatial(d['maps'],r);t=d['lm_terms'];assert t['valid_tokens']==r['valid_tokens']==t['target_logit'].numel()
        close(float((t['logsumexp'].double()-t['target_logit'].double()).mean()),r['lm_loss'],'historical LM',atol=2e-6,rtol=2e-6)
    return rows,metrics(rows)

def old_spatial(ins,seed,phase,summary):
    late=[]
    for step in range(21,25):
        rows=ins.js(f's{seed}-{phase}/block{step}.json')['images']
        late.append(median(r['effective_support'] for r in rows)<128 or median(r['top11_mass'] for r in rows)>.35)
    severe=phenotype(summary,late)
    return dict(**severe,stable=summary['support_median']>=256 and summary['support_below128']<=8 and summary['top11_above035']<=8 and sum(late)<=1)

def main():
    import signal,time
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--stage',choices=('train','reference','final'),required=True)
    p.add_argument('--checker-commit');a=p.parse_args()
    assert not torch.cuda.is_available();torch.set_num_threads(4)
    def timeout(*_):raise TimeoutError('Independent CPU two-hour limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(7200);started=time.time()
    root=a.campaign;guard=WriteGuard(root);ins=Inputs();plan,source=read_campaign(root,a.code_commit)
    checker=checker_identity(a.code_commit,a.checker_commit or a.code_commit,source)
    if a.stage=='train':
        results={n:verify_train(root,n,plan,source) for n in PHASES}
        assert sum(x['updates'] for x in results.values())==48
        assert read(root/'budget.json')['counts']==dict(update=48,block=50,eval=0,model=4)
        write(root/'training-verification.json',dict(status='PASS',phases=results,code_commit=a.code_commit,
            source_sha256=canonical_json_sha256(source),checker_identity=checker,seconds=time.time()-started))
        print('TRAIN_AND_FRESH_RESTORE_PASS');return
    training=read(root/'training-verification.json');assert training['status']=='PASS' and training['code_commit']==a.code_commit
    assert training['checker_identity']==checker
    if a.stage=='reference':
        refs={}
        for seed in SEEDS:
            _,refs[str(seed)]=verify_eval(root,f'eval-ref-s{seed}-dynamic24',plan,source)
        assert read(root/'budget.json')['counts']==dict(update=48,block=50,eval=64,model=6)
        write(root/'reference-verification.json',dict(status='PASS',references=refs,exact_raw_endpoint_rows=64,
            code_commit=a.code_commit,checker_identity=checker,seconds=time.time()-started))
        print('BOTH_HISTORICAL_ENDPOINTS_EXACT');return
    refs=read(root/'reference-verification.json');assert refs['status']=='PASS' and refs['code_commit']==a.code_commit
    assert refs['checker_identity']==checker
    producer=read(root/'comparison-result.json');assert producer['status']=='PASS'
    summaries={};spatial={};pairs={};evaluations={}
    for seed in SEEDS:
        s=str(seed);b,ev=verify_eval(root,f'eval-s{seed}-lm_only24',plan,source);evaluations[s]=ev
        old_d,ds=old_endpoint(ins,seed,'dynamic');old_e,es=old_endpoint(ins,seed,'lce_off')
        # Old refs were independently remeasured exactly; preserve both raw chains.
        for k,value in ds.items():close(value,refs['references'][s]['metrics'][k],k,rtol=1e-12)
        summaries[s]=dict(lm_only=ev['metrics'],dynamic=ds,early_only=es)
        spatial[s]=dict(lm_only=spatial_outcome(root,f's{seed}-lm_only',ev['metrics'],range(21,25)),
            dynamic=old_spatial(ins,seed,'dynamic',ds),early_only=old_spatial(ins,seed,'lce_off',es))
        pairs[s]={}
        for arm,rows in (('dynamic',old_d),('early_only',old_e)):
            assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in b]
            pairs[s][arm]=[dict(sample_id=x['sample_id'],label=x['label'],
                support_delta=x['effective_support']-y['effective_support'],score_delta=x['score']-y['score'],top11_delta=x['top11_mass']-y['top11_mass'],lm_loss_delta=x['lm_loss']-y['lm_loss']) for x,y in zip(rows,b)]
    decision,comparisons=verdict(summaries)
    assert decision==producer['decision']
    for s in summaries:
        for arm in summaries[s]:
            for k,value in summaries[s][arm].items():close(value,producer['summaries'][s][arm][k],k,rtol=1e-12)
        for arm in comparisons[s]:
            assert comparisons[s][arm]['pattern']==producer['comparisons'][s][arm]['pattern']
            for k in ('delta_auroc','delta_average_precision'):close(comparisons[s][arm][k],producer['comparisons'][s][arm][k],k,rtol=1e-12)
    budget=read(root/'budget.json');assert budget['counts']==PLANNED
    ordered=[]
    for phase,cfg in PHASES.items():
        ordered.append(('model',phase))
        for step in range(cfg['start']+1,cfg['end']+1):
            ordered.append(('block',f'{phase}:block{step}'))
            if cfg['mode']=='train':ordered.append(('update',f'{phase}:block{step}'))
    for phase in EVALUATIONS:
        ordered.append(('model',phase));ordered.extend(('eval',f'{phase}:{i}') for i in range(32))
    assert [(e['kind'],e['context']) for e in budget['events']]==ordered
    last_update=max(e['epoch'] for e in budget['events'] if e['kind']=='update')
    assert all(e['epoch']>last_update for e in budget['events'] if e['kind']=='eval')
    assert max(e['epoch'] for e in budget['events'])-budget['started_epoch']<=SECONDS
    # Inventory every new tensor body after all model writers stopped; validate
    # processed body hashes against their producing record/manifest in checks above.
    raw_files={str(p):sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name.endswith(('.pt','.pt.gz'))}
    assert len(raw_files)==188 #50 blocks+2 starts+8 checkpoint/runtime bodies+128 eval
    write(root/'verification.json',dict(status='PASS',decision=decision,summaries=summaries,comparisons=comparisons,spatial=spatial,
        paired_reference_minus_lm_only=pairs,training_verification_sha256=sha256_file(root/'training-verification.json'),
        reference_verification_sha256=sha256_file(root/'reference-verification.json'),producer_sha256=sha256_file(root/'comparison-result.json'),
        new_evaluations=evaluations,historical_inputs_checked=ins.checked,raw_files=raw_files,budget=budget['counts'],
        code_commit=a.code_commit,source_sha256=canonical_json_sha256(source),checker_identity=checker,seconds=time.time()-started,
        scope='Fixed-seed exploratory development objective ablation; no formal efficacy, generalization, calibration or cutoff selection'))
    print(json.dumps(dict(status='PASS',decision=decision,summaries=summaries,spatial=spatial),indent=2))
if __name__=='__main__':main()
