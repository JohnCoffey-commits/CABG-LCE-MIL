"""Continuous fixed LM-only trajectories and exact own-midpoint replays."""
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
from reproduction.stage2q.run import canonicalize_dormant,full_state,data_contract,make_optimizer,update
from reproduction.stage2s.common import write,setup_flags,runtime_state,restore_runtime,state_digest,device_inventory,runtime_devices
from reproduction.stage2s.resize import ResizeProbe
from reproduction.stage2s.run import collect as original_collect
from reproduction.stage2s.trajectory import predict_next
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2z.common import Budget,save_raw,load_raw,source_identity,provenance as make_provenance,read_campaign,require_execution,require_phase_ready,encode_master,disk_guard,PHASES,parent,configuration,read,Inputs,BOUNDARIES,data_metadata,WriteGuard,BASE,SPLIT
from reproduction.stage2u.policy import gradients
from reproduction.stage2t.evaluate import bind_data_registry


def collect(*args):
    maps=[]; method=RawAttentionObserver.finalize
    def capture(self,evidence):
        result=method(self,evidence)
        maps.append(cpu_tree({k:result[k] for k in ("abnormal_raw","normal_raw")}))
        return result
    RawAttentionObserver.finalize=capture
    try: result=original_collect(*args)
    finally: RawAttentionObserver.finalize=method
    assert len(maps)==3
    return (*result,maps)


def run(a):
    require_execution(a)
    assert a.base_model==BASE and a.split_audit==SPLIT
    guard=WriteGuard(a.campaign);ins=Inputs()
    plan,source=read_campaign(a.campaign,a.code_commit)
    require_phase_ready(a.campaign,a.phase)
    cfg=PHASES[a.phase]
    for name,value in cfg.items():setattr(a,name,value)
    a.resume,parent_provenance=parent(a.campaign,a.phase,plan)
    a.output=a.campaign/a.phase
    a.output.mkdir(exist_ok=False);budget=Budget(a.campaign);budget.reserve('model',a.phase);setup_flags()
    split,manifest,_=data_metadata(a.campaign)
    provenance=make_provenance(plan,a.phase)
    write(a.output/'source.json',source);write(a.output/'provenance.json',provenance)
    write(a.output/'configuration.json',configuration(a.seed))
    write(a.output/'environment.json',{**plan['environment'],'cublas':os.environ['CUBLAS_WORKSPACE_CONFIG'],
          'tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_deterministic':torch.backends.cudnn.deterministic,
          'global_deterministic':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),
          'mode':a.mode,'phase':a.phase,'seed':a.seed,'start':a.start,'end':a.end})
    startblock=a.start
    allowed=[str((Path(split['image_root'])/r['relative_path']).resolve()) for r in manifest[startblock*3:a.end*3]]
    audit=MedicalImageOpenAudit(Path(split['image_root']),allowed);audit.install()
    attention=AttentionExecution('canonical_flash_deterministic').install()
    probe=ResizeProbe(deterministic=True).install()
    monitor=NvidiaSmiMonitor(a.output/'nvidia-smi.csv');monitor.start()
    try:
        write(a.output/'data-binding.json',bind_data_registry(split['train_annotation'],split['image_root']))
        from reproduction.stage2h.runtime import load_stage2h_runtime,make_stage2h_dataset
        wrapper,loaded=load_stage2h_runtime(a.base_model,seed=a.seed,training=True,gradient_checkpointing=True)
        model=wrapper.llm;named=dict(model.named_parameters())
        if a.seed==42:assert canonical_json_sha256(loaded['step0_state'])==INITIAL_STATE_SHA256
        write(a.output/'dormant.json',canonicalize_dormant(model))
        initial_trainable=canonical_json_sha256(trainable_state_record(model))
        if a.seed==42:assert initial_trainable==INITIAL_STATE_SHA256
        initial_full=full_state(model);write(a.output/'full-initial-state.json',initial_full)
        initial=cpu_tree({n:named[n] for n in TRAINABLE_NAMES})
        _,dataset,collator=make_stage2h_dataset(a.base_model,'medic_ad_stage2h_calibration',shuffle=False)
        assert [(r.get('cabg_sample_id'),r.get('scout_exposure_id')) for r in dataset.list_data_dict]==[(r['sample_id'],r['scout_exposure_id']) for r in manifest]
        initial_rng=snapshot_rng_state()
        initial_info={'seed':a.seed,'trainable_state_sha256':initial_trainable,
                      'rng_fingerprint':rng_state_fingerprint(initial_rng),'runtime_sha256':state_digest(runtime_state(model))}
        if a.seed==42:assert initial_info['rng_fingerprint']==INITIAL_RNG_SHA256
        assert initial_info==ins.js(f's{a.seed}-prefix/initial-identity.json')
        assert initial_full==ins.js(f's{a.seed}-prefix/full-initial-state.json')
        assert state_digest(initial)==state_digest(ins.raw(f's{a.seed}-prefix/start.pt.gz')['trainable'])
        write(a.output/'initial-identity.json',initial_info)
        if not a.start:
            saved=save_raw(a.output/'start.pt.gz',{'trainable':initial,'runtime':runtime_state(model),'rng':initial_rng})
            write(a.output/'start.json',{'raw':saved,'master_sha256':state_digest({n:v.float() for n,v in initial.items()})})
        if a.resume:
            write(a.output/'parent-lock.json',{'checkpoint':str(a.resume),'checkpoint_sha256':sha256_file(a.resume),'provenance':parent_provenance})
        masters,opt,sched=make_optimizer(initial,wrapper.device);controller=DynamicController();anchor='0'*64
        if not a.start:
            write(a.output/'initial-optimizer.json',cpu_tree(opt.state_dict()))
            write(a.output/'initial-scheduler.json',sched.state_dict())
        if a.resume:
            payload,check=load_strict(a.resume,parent_provenance)
            assert payload['sampler_state']['cursor']==a.start
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
        for block in range(startblock,a.end):
            budget.reserve('block',f'{a.output.name}:block{block+1}');started=time.monotonic()
            before=cpu_tree(masters);optim_before=cpu_tree(opt.state_dict());controller_before=controller.state_dict()
            entry_runtime=runtime_state(model)
            lm,lce,images,inputs,rng,maps=collect(wrapper,dataset,collator,manifest,block,attention,probe)
            applied,details=gradients(lm,lce,[r['label'] for r in images],controller=controller,arm=a.arm,frozen=0.)
            if a.mode=='replay' or block==0:
                if a.mode=='replay':
                    reference_root=a.campaign/f's{a.seed}-lm_only'
                    reference=read(reference_root/f'block{block+1}.json');raw_path=reference_root/f'block{block+1}.pt.gz'
                    assert sha256_file(raw_path)==reference['raw']['sha256'];ref_raw=load_raw(raw_path)
                else:reference,ref_raw=ins.record(f's{a.seed}-prefix',1)
                exact={'raw_lm':all(torch.equal(v,ref_raw['lm'][i][n].float()) for i,g in enumerate(lm) for n,v in g.items()),
                       'raw_lce':all(torch.equal(v,ref_raw['lce'][i][n].float()) for i,g in enumerate(lce) for n,v in g.items()),
                       'raw_maps':state_digest(maps)==state_digest(ref_raw['attention_maps']),
                       'images':images==reference['images'],'master':state_digest(before)==reference['master_before'],
                       'inputs':state_digest(inputs)==reference['inputs_sha256'],'rng':rng==reference['rng'],
                       'optimizer':state_digest(optim_before)==reference['optimizer_before'],
                       'runtime':state_digest(entry_runtime)==reference['runtime_before'],'shadow':details['budget']==reference['controller']}
                if a.mode=='replay':exact['applied']=state_digest(applied)==reference['applied_sha256']
                if not all(exact.values()):raise RuntimeError(f'Actual binding/control gate: {exact}')
                write(a.output/f'block{block+1}-reference-gate.json',exact);del ref_raw
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
            raw={'lm':[{n:v.to(torch.bfloat16) for n,v in g.items()} for g in lm],
                 'lce':[{n:v.to(torch.bfloat16) for n,v in g.items()} for g in lce],
                 'master_xor':encode_master(before,after),'attention_maps':maps}
            assert all(torch.equal(g[n],raw[key][i][n].float()) for key,gs in (('lm',lm),('lce',lce)) for i,g in enumerate(gs) for n in g)
            saved=save_raw(a.output/f'block{block+1}.pt.gz',raw)
            record={'block':block+1,'images':images,'rng':rng,'controller_before':controller_before,'controller':details['budget'],
                    'policy':details['policy'],'applied_sha256':state_digest(applied),
                    'gradient':details['gradient'],'update':update_info,'raw':saved,'seconds':time.monotonic()-started,
                    'inputs_sha256':state_digest(inputs),'runtime_before':state_digest(entry_runtime),
                    'runtime_after':state_digest(runtime_state(model)),
                    'optimizer_before':state_digest(optim_before),'optimizer_after':state_digest(cpu_tree(opt.state_dict())),
                    'optimizer_devices':device_inventory(opt.state_dict()),
                    'aggregate_lm':state_digest(details['aggregate_lm']),'aggregate_lce':state_digest(details['aggregate_lce']),
                    'master_before':state_digest(before),'master_after':state_digest(after),
                    'real_updates':int(a.mode=='train'),'phase':a.phase,'seed':a.seed,
                    'actual_master_after':state_digest(cpu_tree(masters)),'previous_record_sha256':anchor}
            if a.mode=='replay':
                for key in ('master_after','runtime_after','controller','aggregate_lm','aggregate_lce'):
                    if record[key]!=reference[key]:raise RuntimeError('Restored-host reference diverged: '+key)
                assert record['actual_master_after']==record['master_before']
                assert record['optimizer_after']==record['optimizer_before']
            anchor=canonical_json_sha256(record);record['record_sha256']=anchor
            write(a.output/f'block{block+1}.json',record)
            print(json.dumps({'output':a.output.name,'block':block+1,'selected_lambda':details['policy']['selected_lambda'],'seconds':record['seconds'],'real_updates':record['real_updates']}),flush=True)
            del raw,lm,lce,applied,details,before,after,optim_before,inputs,maps,entry_runtime;gc.collect();torch.cuda.empty_cache()
            if a.mode=='train' and block+1 in BOUNDARIES:
                ck=a.output/f'block{block+1}-checkpoint';ck.mkdir();disk_guard(ck)
                payload=_checkpoint_payload(model,masters,opt,sched,controller,cursor=block+1,split=split,run_provenance=provenance,last_hash=anchor)
                check=save_atomic(ck/'checkpoint.pt',payload)
                saved=save_raw(ck/'runtime-state.pt.gz',runtime_state(model))
                write(ck/'runtime-state.json',{'checkpoint_sha256':check['checkpoint_sha256'],'raw':saved,
                       'optimizer_devices':device_inventory(opt.state_dict()),'runtime_devices':runtime_devices(model)})
                write(ck/'full-state.json',full_state(model))
                del payload;gc.collect()
        if a.mode=='replay':
            restore_runtime(model,side);restore_rng_state(payload['rng_state'])
            assert full_state(model)==restored_full
            assert state_digest(cpu_tree(masters))==state_digest(payload['master_state'])
            assert state_digest(cpu_tree(opt.state_dict()))==state_digest(payload['optimizer_state'])
        final_full=full_state(model);write(a.output/'full-final-state.json',final_full)
        frozen=lambda full:[v for v in full['parameters'] if v['name'] not in TRAINABLE_NAMES]
        assert frozen(final_full)==frozen(initial_full)
        file_result=audit.result();assert file_result['status']=='SUCCESS'
        write(a.output/'file-open-audit.json',file_result)
        write(a.output/'attention.json',attention.report());write(a.output/'resize-calls.json',probe.calls)
        assert not guard.violations
        write(a.output/'prior-inputs-checked.json',ins.checked)
        write(a.output/'result.json',{'historical_write_violations':guard.violations,'status':'SUCCESS','mode':a.mode,'real_updates':a.end-a.start if a.mode=='train' else 0,'start':a.start,'end':a.end,'arm':a.arm,'seed':a.seed,'phase':a.phase,'frozen_state_unchanged':True})
    finally:
        probe.close();attention.close();m=monitor.stop();write(a.output/'monitor-summary.json',m)
        if m['peak_memory_used_mib']>22500:raise RuntimeError('VRAM budget')

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--code-commit',required=True)
    p.add_argument('--base-model',type=Path,default=Path('/home/checkpoints/Lingshu-7B'))
    p.add_argument('--split-audit',type=Path,default=Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json'))
    run(p.parse_args())
if __name__=='__main__':main()
