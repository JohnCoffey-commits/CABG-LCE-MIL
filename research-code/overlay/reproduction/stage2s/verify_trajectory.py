"""Read raw evidence; independently reconstruct controller and AdamW arithmetic."""
import argparse
import gc
import json
import math
from pathlib import Path
import torch
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2i.fingerprint import sha256_file,canonical_json_sha256
from reproduction.stage2q.run import load_raw
from reproduction.stage2q.verify import compare_map,close,norm32,small_enough
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2s.verify_single import groups,VPT


def verify_math(d,r):
    labels=[x['label'] for x in r['images']]
    assert labels.count('normal')==1 and labels.count('abnormal')==2
    lm={n:torch.stack([g[n].float() for g in d['lm']]).mean(0) for n in TRAINABLE_NAMES}
    lce={n:.5*(torch.stack([d['lce'][i][n].float() for i,l in enumerate(labels) if l=='normal']).mean(0)+
                  torch.stack([d['lce'][i][n].float() for i,l in enumerate(labels) if l=='abnormal']).mean(0)) for n in S9_NAMES}
    assert all(torch.equal(v,d['aggregate_lm'][n]) for n,v in lm.items())
    assert all(torch.equal(v,d['aggregate_lce'][n]) for n,v in lce.items())
    rms={}
    for k in ('lm','lce'):
        ps=[norm32({n:g[n] for n in S9_NAMES})**2 for g in d[k]]
        rms[k]=math.sqrt(.5*(sum(v for v,l in zip(ps,labels) if l=='normal')+sum(v for v,l in zip(ps,labels) if l=='abnormal')/2)+1e-16)
    b=r['controller_before'];c=r['controller'];step=int(b['valid_blocks'])+1
    assert c['valid_blocks']==step==r['block'] and b['beta']==.9
    for k in ('lm','lce'):
        ema=.9*b[k+'_ema']+(1-.9)*rms[k]
        close(c[k+'_ema'],ema,k+' EMA')
        close(c['corrected_'+k+'_rms'],ema/(1-.9**step),k+' corrected RMS')
    ln=math.sqrt(sum(float(lm[n].double().square().sum()) for n in S9_NAMES))
    an=math.sqrt(sum(float(lce[n].double().square().sum()) for n in S9_NAMES))
    close(c['lambda_raw'],.1*c['corrected_lm_rms']/(c['corrected_lce_rms']+1e-8),'lambda raw')
    close(c['lambda_cap'],.2*ln/(an+1e-8),'lambda cap')
    close(c['lambda_final'],min(c['lambda_raw'],c['lambda_cap']),'lambda final')
    combined={n:g.clone() for n,g in lm.items()}
    for n in lce:combined[n].add_(lce[n]*c['lambda_final'])
    close(r['gradient']['clip_scale'],min(1.,1./(norm32(combined)+1e-12)),'clip')
    assert all(torch.equal(combined[n]*r['gradient']['clip_scale'],d['applied'][n]) for n in TRAINABLE_NAMES)
    opt=d['optimizer_before'];group=opt['param_groups'][0]
    assert group['betas']==(.9,.999) and group['weight_decay']==0 and group['eps']==1e-8
    lr=1e-4*.5*(1+math.cos(math.pi*(step-1)/24))
    close(group['lr'],lr,'24-step schedule',rtol=1e-12);close(r['update']['learning_rate_used'],lr,'used LR',rtol=1e-12)
    max_error=0.;moment_error=0.
    for n,index in zip(TRAINABLE_NAMES,group['params']):
        g=d['applied'][n].double();before=d['master_before'][n].double();s=opt['state'].get(index)
        if s:
            assert float(s['step'])==step-1
            m=.9*s['exp_avg'].double()+.1*g;v=.999*s['exp_avg_sq'].double()+.001*g.square()
        else:
            assert step==1;m=.1*g;v=.001*g.square()
        expected=before-lr*(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8)
        max_error=max(max_error,float((expected-d['updated_master'][n].double()).abs().max()))
        if r['real_updates']:
            actual=d['optimizer_after']['state'][index];assert float(actual['step'])==step
            for key,value in [('exp_avg',m),('exp_avg_sq',v)]:
                e=float((actual[key].double()-value).square().sum().sqrt())/max(float(value.square().sum().sqrt()),1e-12)
                moment_error=max(moment_error,e)
    assert max_error<=2e-7 and moment_error<=2e-6
    return {'aggregation_applied_exact':True,'rms_ema_cap_clip_recomputed':True,
            'schedule_horizon':24,'adamw_max_abs_error':max_error,'adam_moment_max_relative_l2':moment_error}


def read(p,block):
    r=json.loads((p/f'block{block}.json').read_text());d=load_raw(p/f'block{block}.pt.gz')
    assert sha256_file(p/f'block{block}.pt.gz')==r['raw']['sha256']
    rc=dict(r);claimed=rc.pop('record_sha256');assert canonical_json_sha256(rc)==claimed
    if block>1 and r['real_updates']:
        prev=json.loads((p/f'block{block-1}.json').read_text());assert r['previous_record_sha256']==prev['record_sha256']
    return d,r,verify_math(d,r)


def compare(d,e,r,s):
    assert state_digest(d['inputs'])==state_digest(e['inputs']) and r['rng']==s['rng']
    result={'applied':groups(d['applied'],e['applied']),
            'update':groups({n:d['updated_master'][n]-d['master_before'][n] for n in TRAINABLE_NAMES},
                            {n:e['updated_master'][n]-e['master_before'][n] for n in TRAINABLE_NAMES}),
            'raw_lm':compare_map(d['aggregate_lm'],e['aggregate_lm']),
            'raw_lce':compare_map(d['aggregate_lce'],e['aggregate_lce']),
            'per_image_lm_exact':all(all(torch.equal(v,e['lm'][i][n]) for n,v in g.items()) for i,g in enumerate(d['lm'])),
            'per_image_lce_exact':all(all(torch.equal(v,e['lce'][i][n]) for n,v in g.items()) for i,g in enumerate(d['lce'])),
            'controller_exact':r['controller']==s['controller'],
            'runtime_before_exact':state_digest(d['runtime_before'])==state_digest(e['runtime_before']),
            'runtime_after_exact':state_digest(d['runtime_after'])==state_digest(e['runtime_after'])}
    checks=[v for k in ('applied','update') for v in result[k].values()]
    checks.extend(v for k in ('applied','update') for v in result[k]['VPT']['per_tensor'].values())
    result['strict_pass']=all(small_enough(v) for v in checks)
    result['bounded_pass']=all(v['relative_l2']<=.02 for v in checks)
    result['exact']=all(v['exact'] for v in checks) and result['per_image_lm_exact'] and result['per_image_lce_exact'] and result['controller_exact'] and result['runtime_before_exact'] and result['runtime_after_exact']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    root=a.campaign;ps=[root/f'train-p{i}' for i in range(2)]
    for pth in ps+[root/f'resume-p{i}' for i in range(2)]:
        assert json.loads((pth/'result.json').read_text())['status']=='SUCCESS'
        assert json.loads((pth/'file-open-audit.json').read_text())['status']=='SUCCESS'
        assert json.loads((pth/'monitor-summary.json').read_text())['peak_memory_used_mib']<=22500
        assert json.loads((pth/'provenance.json').read_text())==json.loads((ps[0]/'provenance.json').read_text())
        assert json.loads((pth/'full-initial-state.json').read_text())==json.loads((ps[0]/'full-initial-state.json').read_text())
    math_checks={};fresh={};resumes={};lengths={str(i):{k:0. for k in ('T21','S9','VPT')} for i in range(2)}
    for block in range(1,5):
        d,r,c=read(ps[0],block);e,s,v=read(ps[1],block)
        math_checks[f'p0-b{block}']=c;math_checks[f'p1-b{block}']=v
        fresh[str(block)]=compare(d,e,r,s)
        for i,item in enumerate((d,e)):
            for name,ns in [('T21',TRAINABLE_NAMES),('S9',S9_NAMES),('VPT',VPT)]:
                lengths[str(i)][name]+=math.sqrt(sum(float((item['updated_master'][n].double()-item['master_before'][n].double()).square().sum()) for n in ns))
        if block==4:
            initial=load_raw(ps[0]/'start.pt.gz')['trainable'];drift={}
            for name,ns in [('T21',TRAINABLE_NAMES),('S9',S9_NAMES),('VPT',VPT)]+[(n,(n,)) for n in VPT]:
                diff=math.sqrt(sum(float((d['updated_master'][n].double()-e['updated_master'][n].double()).square().sum()) for n in ns))
                displacements=[math.sqrt(sum(float((x['updated_master'][n].double()-initial[n].double()).square().sum()) for n in ns)) for x in (d,e)]
                param=max(math.sqrt(sum(float(x['updated_master'][n].double().square().sum()) for n in ns)) for x in (d,e))
                drift[name]={'relative_to_four_step_displacement':diff/max(*displacements,1e-12),'relative_to_parameter_norm':diff/max(param,1e-12),
                             'absolute_difference':diff,'four_step_displacements':displacements}
                if name in lengths['0']:drift[name]['relative_to_sum_step_lengths']=diff/max(lengths['0'][name],lengths['1'][name],1e-12)
            for i,(original,rec) in enumerate(((d,r),(e,s))):
                rr=root/f'resume-p{i}';z,zr,zc=read(rr,4);math_checks[f'resume-p{i}']=zc
                assert state_digest(original['master_before'])==state_digest(z['master_before'])
                assert state_digest(original['optimizer_before'])==state_digest(z['optimizer_before'])
                resumes[str(i)]=compare(original,z,rec,zr)
                assert all(json.loads((rr/'resume-audit.json').read_text())['exact'].values())
                del z;gc.collect()
        del d,e;gc.collect()
    exact=all(c['exact'] for c in fresh.values()) and all(c['exact'] for c in resumes.values())
    bounded=all(c['bounded_pass'] for c in fresh.values()) and all(c['bounded_pass'] for c in resumes.values()) and all(v['relative_to_four_step_displacement']<=.02 for v in drift.values())
    # A restore must not add variability beyond the fresh-process comparison.
    envelope=True
    for rr in resumes.values():
        for key in ('applied','update'):
            for group in ('T21','S9','VPT'):
                envelope &= rr[key][group]['relative_l2']<=fresh['4'][key][group]['relative_l2']+1e-12
            for n in VPT:envelope &= rr[key]['VPT']['per_tensor'][n]['relative_l2']<=fresh['4'][key]['VPT']['per_tensor'][n]['relative_l2']+1e-12
    decision='READY_EXACT' if exact else 'READY_WITH_REPLICATES' if bounded and envelope else 'NOT_READY'
    report={'decision':decision,'fresh':fresh,'resume':resumes,'four_step_drift':drift,'independent_math':math_checks,
            'restore_within_fresh_envelope':bool(envelope),'exact':exact,'bounded':bounded,'real_updates':8,'scope':'engineering readiness on first four training blocks only'}
    write(root/'trajectory-verification.json',report)
    print(json.dumps({'decision':decision,'exact':exact,'bounded':bounded,'restore_within_fresh_envelope':bool(envelope),'real_updates':8},indent=2))

if __name__=='__main__':main()
