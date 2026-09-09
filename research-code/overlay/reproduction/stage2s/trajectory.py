"""Two bounded four-update trajectories and fresh-process no-step resume replay."""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
import torch
from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.rng_state import snapshot_rng_state,restore_rng_state,rng_state_fingerprint
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit,NvidiaSmiMonitor
from reproduction.stage2l.controller import DynamicController,compute_block_gradients
from reproduction.stage2l.checkpoint import save_atomic,load_strict
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES,SOURCE_FILES
from reproduction.stage2m.train_scout import _checkpoint_payload,matched_configuration
from reproduction.stage2p import INITIAL_STATE_SHA256,INITIAL_RNG_SHA256,TRAIN_SHA256,CONFIG_SHA256
from reproduction.stage2q.execution import AttentionExecution,cpu_tree
from reproduction.stage2q.run import canonicalize_dormant,full_state,data_contract,save_raw,load_raw,make_optimizer,update
from reproduction.stage2s.common import BASELINE,Budget,write,setup_flags,runtime_state,restore_runtime,state_digest,device_inventory,runtime_devices
from reproduction.stage2s.resize import ResizeProbe
from reproduction.stage2s.run import collect


def predict_next(masters,opt,applied):
    """Read-only FP32 AdamW algebra on clones; no optimizer.step/model mutation.

    Foreach operation ordering matches installed torch 2.7 noncapturable AdamW.
    Independent float64 arithmetic is checked separately from these predictions.
    """
    group=opt.param_groups[0]
    assert group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
    assert not group['amsgrad'] and not group['maximize'] and not group['capturable']
    ps=[p.detach().clone() for p in masters.values()]
    gs=[applied[n].to(p) for n,p in masters.items()]
    ms=[opt.state[p]['exp_avg'].clone() for p in masters.values()]
    vs=[opt.state[p]['exp_avg_sq'].clone() for p in masters.values()]
    steps=[float(opt.state[p]['step'])+1 for p in masters.values()]
    torch._foreach_lerp_(ms,gs,1-.9)
    torch._foreach_mul_(vs,.999);torch._foreach_addcmul_(vs,gs,gs,1-.999)
    denom=torch._foreach_sqrt(vs)
    torch._foreach_div_(denom,[(1-.999**s)**.5 for s in steps])
    torch._foreach_add_(denom,1e-8)
    torch._foreach_addcdiv_(ps,ms,denom,[-group['lr']/(1-.9**s) for s in steps])
    return {n:p.cpu() for n,p in zip(TRAINABLE_NAMES,ps)}


def run(a):
    gate=json.loads((a.campaign/'candidate-comparison.json').read_text())
    if not gate['bounded_pass']: raise RuntimeError('Single-block gate forbids expansion')
    if (a.mode=='resume') != (a.resume is not None): raise RuntimeError('Resume argument contract')
    a.output.mkdir(exist_ok=False);budget=Budget(a.campaign);setup_flags()
    assert canonical_json_sha256(matched_configuration())==CONFIG_SHA256
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==BASELINE
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True)
    paths=set(SOURCE_FILES)|{str(p.relative_to(repo)) for d in ('stage2q','stage2s') for p in (repo/'reproduction'/d).rglob('*.py')}
    source={p:sha256_file(repo/p) for p in sorted(paths)}
    split,manifest=data_contract(a.split_audit,4)
    provenance={'head':BASELINE,'source_sha256':canonical_json_sha256(source),'train_manifest_sha256':TRAIN_SHA256,
                'setting':'deterministic_bilinear','scope':'route_a_first4_training_blocks_only','schedule_horizon':24}
    write(a.output/'source.json',source);write(a.output/'provenance.json',provenance)
    startblock=3 if a.mode=='resume' else 0
    allowed=[str((Path(split['image_root'])/r['relative_path']).resolve()) for r in manifest[startblock*3:12]]
    audit=MedicalImageOpenAudit(Path(split['image_root']),allowed);audit.install()
    attention=AttentionExecution('canonical_flash_deterministic').install()
    probe=ResizeProbe(deterministic=True).install()
    monitor=NvidiaSmiMonitor(a.output/'nvidia-smi.csv');monitor.start()
    try:
        os.environ['MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION']=split['train_annotation']
        os.environ['MEDIC_AD_STAGE2H_IMAGE_ROOT']=split['image_root']
        from reproduction.stage2h.runtime import load_stage2h_runtime,make_stage2h_dataset
        wrapper,loaded=load_stage2h_runtime(a.base_model,seed=42,training=True,gradient_checkpointing=True)
        model=wrapper.llm;named=dict(model.named_parameters())
        assert canonical_json_sha256(loaded['step0_state'])==INITIAL_STATE_SHA256
        write(a.output/'dormant.json',canonicalize_dormant(model))
        assert canonical_json_sha256(trainable_state_record(model))==INITIAL_STATE_SHA256
        initial_full=full_state(model);write(a.output/'full-initial-state.json',initial_full)
        initial=cpu_tree({n:named[n] for n in TRAINABLE_NAMES})
        _,dataset,collator=make_stage2h_dataset(a.base_model,'medic_ad_stage2h_calibration',shuffle=False)
        assert [(r.get('cabg_sample_id'),r.get('scout_exposure_id')) for r in dataset.list_data_dict]==[(r['sample_id'],r['scout_exposure_id']) for r in manifest]
        initial_rng=snapshot_rng_state();assert rng_state_fingerprint(initial_rng)==INITIAL_RNG_SHA256
        save_raw(a.output/'start.pt.gz',{'trainable':initial,'runtime':runtime_state(model),'rng':initial_rng})
        masters,opt,sched=make_optimizer(initial,wrapper.device);controller=DynamicController();anchor='0'*64
        if a.mode=='resume':
            payload,check=load_strict(a.resume,provenance)
            assert payload['sampler_state']['cursor']==3
            assert payload['sampler_state']['manifest_sha256']==TRAIN_SHA256
            assert payload['sampler_state']['order_sha256']==canonical_json_sha256(manifest)
            sideinfo=json.loads(a.resume.with_name('runtime-state.json').read_text())
            sidepath=a.resume.with_name('runtime-state.pt.gz')
            assert sideinfo['checkpoint_sha256']==check['checkpoint_sha256'] and sideinfo['raw']['sha256']==sha256_file(sidepath)
            side=load_raw(sidepath)
            with torch.no_grad():
                for n in TRAINABLE_NAMES:
                    masters[n].copy_(payload['master_state'][n].to(wrapper.device))
                    named[n].copy_(payload['adapter_state'][n].to(wrapper.device))
            # PyTorch restores moments to each parameter's device and keeps the
            # noncapturable Adam step counters on CPU, as in the fresh process.
            opt.load_state_dict(payload['optimizer_state'])
            sched.load_state_dict(payload['scheduler_state']);controller.load_state_dict(payload['controller_state'])
            restore_runtime(model,side);restore_rng_state(payload['rng_state'])
            anchor=payload['trace_state']['last_record_sha256']
            restored_full=full_state(model)
            exact={'adapter':state_digest(cpu_tree({n:named[n] for n in TRAINABLE_NAMES}))==state_digest(payload['adapter_state']),
                   'master':state_digest(cpu_tree(masters))==state_digest(payload['master_state']),
                   'optimizer':state_digest(cpu_tree(opt.state_dict()))==state_digest(payload['optimizer_state']),
                   'scheduler':sched.state_dict()==payload['scheduler_state'],'controller':controller.state_dict()==payload['controller_state'],
                   'runtime':state_digest(runtime_state(model))==state_digest(side),
                   'optimizer_devices':device_inventory(opt.state_dict())==sideinfo['optimizer_devices'],
                   'runtime_devices':runtime_devices(model)==sideinfo['runtime_devices'],
                   'rng':rng_state_fingerprint(snapshot_rng_state())==check['rng_fingerprint'],
                   'full_parameters_and_buffers':restored_full==json.loads(a.resume.with_name('full-state.json').read_text())}
            if not all(exact.values()):raise RuntimeError(f'Restore mismatch: {exact}')
            write(a.output/'resume-audit.json',{'exact':exact,'checkpoint_sha256':check['checkpoint_sha256'],'sampler':payload['sampler_state']})
        for block in range(startblock,4):
            budget.reserve('block',f'{a.output.name}:block{block+1}');started=time.monotonic()
            before=cpu_tree(masters);optim_before=cpu_tree(opt.state_dict());controller_before=controller.state_dict()
            entry_runtime=runtime_state(model)
            lm,lce,images,inputs,rng=collect(wrapper,dataset,collator,manifest,block,attention,probe)
            applied,details=compute_block_gradients(lm,lce,[r['label'] for r in images],trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=controller)
            lr=opt.param_groups[0]['lr']
            if a.mode=='train':
                budget.reserve('update',f'{a.output.name}:block{block+1}')
                update_info=update(masters,opt,sched,applied)
                with torch.no_grad():
                    for n,p in masters.items():named[n].copy_(p.to(torch.bfloat16))
                after=cpu_tree(masters)
            else:
                after=predict_next(masters,opt,applied)
                assert state_digest(cpu_tree(masters))==state_digest(before)
                assert state_digest(cpu_tree(opt.state_dict()))==state_digest(optim_before)
                update_info={'learning_rate_used':lr,'kind':'cloned_state_algebra_no_optimizer_step'}
            raw={'inputs':inputs,'lm':[{n:v.to(torch.bfloat16) for n,v in g.items()} for g in lm],
                 'lce':[{n:v.to(torch.bfloat16) for n,v in g.items()} for g in lce],
                 'aggregate_lm':details['aggregate_lm'],'aggregate_lce':details['aggregate_lce'],
                 'applied':applied,'master_before':before,'updated_master':after,'optimizer_before':optim_before,
                 'optimizer_after':cpu_tree(opt.state_dict()),'runtime_before':entry_runtime,
                 'runtime_after':runtime_state(model)}
            assert all(torch.equal(g[n],raw[key][i][n].float()) for key,gs in (('lm',lm),('lce',lce)) for i,g in enumerate(gs) for n in g)
            saved=save_raw(a.output/f'block{block+1}.pt.gz',raw)
            record={'block':block+1,'images':images,'rng':rng,'controller_before':controller_before,'controller':details['budget'],
                    'gradient':details['gradient'],'update':update_info,'raw':saved,'seconds':time.monotonic()-started,
                    'real_updates':int(a.mode=='train'),'previous_record_sha256':anchor}
            anchor=canonical_json_sha256(record);record['record_sha256']=anchor
            write(a.output/f'block{block+1}.json',record)
            print(json.dumps({'output':a.output.name,'block':block+1,'lambda':details['budget']['lambda_final'],'seconds':record['seconds'],'real_updates':record['real_updates']}),flush=True)
            del raw,lm,lce,applied,details,before,after,optim_before;gc.collect();torch.cuda.empty_cache()
            if block==2:
                ck=a.output/'block3-checkpoint';ck.mkdir()
                payload=_checkpoint_payload(model,masters,opt,sched,controller,cursor=3,split=split,run_provenance=provenance,last_hash=anchor)
                check=save_atomic(ck/'checkpoint.pt',payload)
                saved=save_raw(ck/'runtime-state.pt.gz',runtime_state(model))
                write(ck/'runtime-state.json',{'checkpoint_sha256':check['checkpoint_sha256'],'raw':saved,
                       'optimizer_devices':device_inventory(opt.state_dict()),'runtime_devices':runtime_devices(model)})
                write(ck/'full-state.json',full_state(model))
                del payload;gc.collect()
        if a.mode=='resume':
            restore_runtime(model,side)
            assert full_state(model)==restored_full
        final_full=full_state(model);write(a.output/'full-final-state.json',final_full)
        frozen=lambda full:[v for v in full['parameters'] if v['name'] not in TRAINABLE_NAMES]
        assert frozen(final_full)==frozen(initial_full)
        file_result=audit.result();assert file_result['status']=='SUCCESS'
        write(a.output/'file-open-audit.json',file_result)
        write(a.output/'attention.json',attention.report());write(a.output/'resize-calls.json',probe.calls)
        write(a.output/'result.json',{'status':'SUCCESS','mode':a.mode,'real_updates':4 if a.mode=='train' else 0,'frozen_state_unchanged':True})
    finally:
        probe.close();attention.close();m=monitor.stop();write(a.output/'monitor-summary.json',m)
        if m['peak_memory_used_mib']>22500:raise RuntimeError('VRAM budget')

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['train','resume'],required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--base-model',type=Path,default=Path('/home/checkpoints/Lingshu-7B'))
    p.add_argument('--split-audit',type=Path,default=Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json'))
    run(p.parse_args())

if __name__=='__main__':main()
