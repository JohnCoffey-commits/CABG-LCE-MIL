"""Independent policy, raw-gradient, AdamW, endpoint and exact-control checks."""
import argparse,gc,json,math
from pathlib import Path
from statistics import median
import torch
from reproduction.stage2i.fingerprint import sha256_file,canonical_json_sha256
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2q.verify import close,norm32
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2t.verify import adam,check_checkpoint,check_spatial,metrics,phenotype
from reproduction.stage2u.common import ARMS,FULL24,read_plan,provenance,load_raw,decode_master

def reconstruct(d,r,frozen):
    labels=[x['label'] for x in r['images']];assert labels.count('normal')==1 and labels.count('abnormal')==2
    lm={n:torch.stack([g[n].float() for g in d['lm']]).mean(0) for n in TRAINABLE_NAMES}
    lce={n:.5*(torch.stack([d['lce'][i][n].float() for i,l in enumerate(labels) if l=='normal']).mean(0)+torch.stack([d['lce'][i][n].float() for i,l in enumerate(labels) if l=='abnormal']).mean(0)) for n in S9_NAMES}
    assert state_digest(lm)==r['aggregate_lm'] and state_digest(lce)==r['aggregate_lce']
    c,b=r['controller'],r['controller_before'];step=r['block'];assert c['valid_blocks']==step==b['valid_blocks']+1
    for kind in ('lm','lce'):
        norms=[norm32({n:g[n] for n in S9_NAMES})**2 for g in d[kind]]
        rms=math.sqrt(.5*(sum(v for v,l in zip(norms,labels) if l=='normal')+sum(v for v,l in zip(norms,labels) if l=='abnormal')/2)+1e-16)
        ema=.9*b[kind+'_ema']+.1*rms
        close(ema,c[kind+'_ema'],'shadow EMA');close(ema/(1-.9**step),c['corrected_'+kind+'_rms'],'shadow corrected RMS')
    raw=.1*c['corrected_lm_rms']/(c['corrected_lce_rms']+1e-8)
    norm=lambda g:math.sqrt(sum(float(v.double().square().sum()) for v in g.values()))
    cap=.2*norm({n:lm[n] for n in S9_NAMES})/(norm(lce)+1e-8)
    close(raw,c['lambda_raw'],'raw');close(cap,c['lambda_cap'],'cap');close(min(raw,cap),c['lambda_final'],'shadow coefficient')
    policy=r['policy'];assert policy['arm'] in ARMS and policy['frozen_ratio']==frozen
    expected=c['lambda_final'] if policy['arm']=='dynamic' else min(frozen,c['lambda_cap']) if policy['arm']=='frozen_ratio' else 0.
    assert policy['selected_lambda']==expected
    combined={n:g.clone() for n,g in lm.items()}
    for n in S9_NAMES:combined[n].add_(lce[n]*expected)
    scale=min(1.,1./(norm32(combined)+1e-12));assert scale==r['gradient']['clip_scale']
    applied={n:g*scale for n,g in combined.items()};assert state_digest(applied)==r['applied_sha256']
    for maps,row in zip(d['attention_maps'],r['images']):check_spatial(maps,row)
    return applied

def verify_eval(root,arm):
    p=root/f'eval-{arm}';rows=json.loads((p/'predictions.json').read_text());expected=json.loads((FULL24/'eval-manifest.json').read_text())
    assert len(rows)==32
    for i,(row,e) in enumerate(zip(rows,expected)):
        assert (row['sample_id'],row['sha256'],row['label'])==(e['sample_id'],e['sha256'],e['scout_label'])
        raw=p/f'image{i:02}.pt.gz';assert sha256_file(raw)==row['raw']['sha256'];d=load_raw(raw)
        check_spatial(d['maps'],row);t=d['lm_terms'];assert row['valid_tokens']==t['valid_tokens']==len(t['target_logit'])
        close(float((t['logsumexp'].double()-t['target_logit'].double()).mean()),row['lm_loss'],'LM',rtol=2e-6,atol=2e-6)
    summary=metrics(rows);prod=json.loads((p/'metrics.json').read_text())
    for k in ('auroc','average_precision'):close(summary[k],prod[k],k,rtol=1e-12)
    if arm=='dynamic':
        old=json.loads((FULL24/'accepted-v2/eval-R0-b24/predictions.json').read_text())
        clean=lambda rows:[{k:v for k,v in r.items() if k!='raw'} for r in rows]
        assert clean(rows)==clean(old),'INVALID_CONTROL endpoint'
    return rows,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    plan,source=read_plan(a.plan,a.code_commit);parent,_=load_strict(Path(plan['parent_checkpoint']),plan['parent_provenance'])
    report={};all_rows={};total_updates=0
    manifest=json.loads((FULL24/'train-manifest.json').read_text())
    for arm in ARMS:
        path=a.campaign/arm
        for phase in (path,a.campaign/f'eval-{arm}'):
            assert json.loads((phase/'source.json').read_text())==source
            assert json.loads((phase/'result.json').read_text())['status']=='SUCCESS'
            audit=json.loads((phase/'file-open-audit.json').read_text());assert audit['status']=='SUCCESS' and not audit['non_allowlisted_paths']
            assert json.loads((phase/'monitor-summary.json').read_text())['peak_memory_used_mib']<=22500
        assert all(json.loads((path/'resume-audit.json').read_text())['exact'].values())
        before=parent['master_state'];group=parent['optimizer_state']['param_groups'][0]
        moments={n:(parent['optimizer_state']['state'][idx]['exp_avg'].double(),parent['optimizer_state']['state'][idx]['exp_avg_sq'].double()) for n,idx in zip(TRAINABLE_NAMES,group['params'])}
        previous=json.loads((FULL24/'R0/block12.json').read_text());maximum=moment_max=0.;late=[];coefficients=[]
        for step in range(13,25):
            r=json.loads((path/f'block{step}.json').read_text());rc=dict(r);claimed=rc.pop('record_sha256');assert canonical_json_sha256(rc)==claimed
            assert r['previous_record_sha256']==previous['record_sha256'] and r['master_before']==state_digest(before)
            assert r['optimizer_before']==previous['optimizer_after'] and r['runtime_before']==previous['runtime_after'] and r['rng']['before_block']==previous['rng']['after_block']
            assert r['controller_before']==dict(beta=.9,lm_ema=previous['controller']['lm_ema'],lce_ema=previous['controller']['lce_ema'],valid_blocks=step-1)
            expected=manifest[(step-1)*3:step*3]
            assert [(x['sample_id'],x['sha256'],x['label'],x['exposure_id']) for x in r['images']]==[(x['sample_id'],x['sha256'],x['scout_label'],x['scout_exposure_id']) for x in expected]
            raw=path/f'block{step}.pt.gz';assert sha256_file(raw)==r['raw']['sha256'];d=load_raw(raw)
            d['updated_master']=decode_master(before,d['master_xor']);assert state_digest(d['updated_master'])==r['master_after']
            d['applied']=reconstruct(d,r,plan['frozen_ratio']);maximum=max(maximum,adam(d,r,before,moments))
            if arm=='dynamic' or step==13:
                reference=json.loads((FULL24/'R0'/f'block{step}.json').read_text());ref=load_raw(FULL24/'R0'/f'block{step}.pt.gz')
                assert all(state_digest(d[k])==state_digest(ref[k]) for k in ('lm','lce'))
                if arm=='dynamic':
                    assert state_digest(d['updated_master'])==state_digest(ref['updated_master'])
                    for k in ('controller','optimizer_after','runtime_after','inputs_sha256','rng'):assert r[k]==reference[k]
                del ref
            if step in (20,24):
                ck=check_checkpoint(path/f'block{step}-checkpoint',r,d,moments,provenance(plan,arm));moment_max=max(moment_max,ck['moment_max_relative_l2'])
            if step>=21:late.append(median(x['effective_support'] for x in r['images'])<128 or median(x['top11_mass'] for x in r['images'])>.35)
            coefficients.append({'block':step,**r['policy'],'cap':r['controller']['lambda_cap'],'scaled_lce_norm':r['gradient']['scaled_auxiliary_shared_norm']})
            before=d['updated_master'];previous=r;total_updates+=r['real_updates'];del d;gc.collect()
        rows,summary=verify_eval(a.campaign,arm);all_rows[arm]=rows
        state=phenotype(summary,late)
        stable=summary['support_median']>=256 and summary['support_below128']<=8 and summary['top11_above035']<=8 and sum(late)<=1
        report[arm]={'metrics':summary,'concentration':state,'stable':stable,'coefficients':coefficients,'adam_max_abs':maximum,'moment_max_relative_l2':moment_max}
    assert report['dynamic']['concentration']['concentrated'],'INVALID_CONTROL phenotype'
    budget=json.loads((a.campaign/'budget.json').read_text());assert total_updates==36 and budget['counts']['update']<=40 and budget['counts']['eval']<=128
    paired={arm:[{'sample_id':x['sample_id'],'score_delta':x['score']-y['score'],'support_delta':x['effective_support']-y['effective_support'],'lm_loss_delta':x['lm_loss']-y['lm_loss']} for x,y in zip(all_rows[arm],all_rows['dynamic'])] for arm in ARMS[1:]}
    write(a.output,{'decision':'VALID_COMPARISON','arms':report,'paired_vs_dynamic':paired,'budget':budget['counts'],'scope':'shared-parent persistence/reversal only; no onset-prevention, efficacy or universal controller-necessity claim'})
    print(json.dumps({'decision':'VALID_COMPARISON','stable':{k:v['stable'] for k,v in report.items()}},indent=2))
if __name__=='__main__':main()
