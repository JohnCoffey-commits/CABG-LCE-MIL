"""One fresh original-runtime process per fixed anchor; forward measurements only."""
import argparse,gc,time
from contextlib import ExitStack
from unittest.mock import patch
from reproduction.stage2x.common import *
from reproduction.stage2x.math import response
from reproduction.stage2i.rng_state import snapshot_rng_state,restore_rng_state
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit,NvidiaSmiMonitor
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2q.execution import AttentionExecution,cpu_tree
from reproduction.stage2q.run import canonicalize_dormant,full_state
from reproduction.stage2s.common import setup_flags,runtime_state,restore_runtime,runtime_devices
from reproduction.stage2s.resize import ResizeProbe
from reproduction.stage2t.evaluate import bind_data_registry
from reproduction.stage2v.common import environment

def trainable(model):return cpu_tree({n:dict(model.named_parameters())[n] for n in TRAINABLE_NAMES})
def install(model,weights,runtime,rng):
    named=dict(model.named_parameters())
    with torch.no_grad():
        for n in TRAINABLE_NAMES:named[n].copy_(weights[n].to(named[n]))
    restore_runtime(model,runtime);restore_rng_state(rng)
    assert state_digest(trainable(model))==state_digest(weights)
    assert state_digest(runtime_state(model))==state_digest(runtime)
    assert rng_state_fingerprint(snapshot_rng_state())==rng_state_fingerprint(rng)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--anchor',choices=ANCHORS,required=True);p.add_argument('--attempt',default='attempt1');p.add_argument('--execute',action='store_true');a=p.parse_args()
    assert a.execute
    out=a.output/a.anchor/a.attempt;out.mkdir(parents=True,exist_ok=False);guard=WriteGuard(a.output)
    plan=read(a.output/'plan.json');ins=Inputs();source=source_identity(a.code_commit,ins)
    assert plan['code_commit']==a.code_commit and source==read(a.output/'source.json') and plan['environment']==environment()
    assert plan['protocol_sha256']==sha256_file(Path(__file__).with_name('PROTOCOL.md'))
    setup_flags();budget=Budget(a.output);budget.reserve('process',str(out));started=time.time()
    cfg,r,original_maps,before,runtime,rng,expected_full=state(ins,a.anchor)
    weights_before={n:v.bfloat16() for n,v in before.items()};registered=plan['anchors'][a.anchor]
    reps,aliases=variants(ins.wjs('corners/'+a.anchor+'.json'));assert reps==registered['representatives'] and aliases==registered['aliases']
    split=read(a.output/'split.json');manifest=read(a.output/'train-manifest.json');indices=registered['indices']
    allowed=[str((Path(split['image_root'])/manifest[i]['relative_path']).resolve()) for i in indices]
    audit=MedicalImageOpenAudit(Path(split['image_root']),allowed);audit.install()
    attention=AttentionExecution('canonical_flash_deterministic').install();probe=ResizeProbe(deterministic=True).install();probe.capture=False
    monitor=NvidiaSmiMonitor(out/'nvidia-smi.csv');monitor.start()
    try:
        write(out/'data-binding.json',bind_data_registry(split['train_annotation'],split['image_root']))
        from reproduction.stage2h.runtime import load_stage2h_runtime,make_stage2h_dataset,move_batch_to_device
        wrapper,loaded=load_stage2h_runtime(BASE,seed=cfg['seed'],training=True,gradient_checkpointing=True);model=wrapper.llm
        write(out/'dormant.json',canonicalize_dormant(model));initial=full_state(model)
        assert initial==ins.js(f"s{cfg['seed']}-prefix/full-initial-state.json")
        _,dataset,collator=make_stage2h_dataset(BASE,'medic_ad_stage2h_calibration',shuffle=False)
        assert [(x.get('cabg_sample_id'),x.get('scout_exposure_id')) for x in dataset.list_data_dict]==[(x['sample_id'],x['scout_exposure_id']) for x in manifest]
        install(model,weights_before,runtime,rng);assert full_state(model)==expected_full
        write(out/'entry.json',dict(status='PASS',initial_full_exact=True,anchor_full_exact=True,master_sha256=state_digest(before),
            runtime_sha256=state_digest(runtime),rng_sha256=rng_state_fingerprint(rng),runtime_devices=runtime_devices(model),record_sha256=r['record_sha256'],
            source_sha256=canonical_json_sha256(source),code_commit=a.code_commit,environment=environment(),seed=cfg['seed'],step=cfg['step'],aliases=aliases))
        raw_by_variant={};rows_by_variant={};variant_info={}
        with ExitStack() as stack:
            for obj,name in ((torch.autograd,'grad'),(torch.autograd,'backward'),(torch.Tensor,'backward'),(torch.optim.AdamW,'step')):stack.enter_context(patch.object(obj,name,forbidden))
            for key in ['baseline',*reps]:
                weights=weights_before if key=='baseline' else candidate(ins,a.anchor,key,before)
                install(model,weights,runtime,rng);entry=state_digest(trainable(model));runtime_entry=state_digest(runtime_state(model))
                rows=[];raw_rows=[];inputs=[];input_trees=[]
                for j,index in enumerate(indices):
                    row=manifest[index];batch=move_batch_to_device(collator([dataset[index]]),wrapper.device);input_trees.append(cpu_tree(batch));inputs.append(state_digest(input_trees[-1]))
                    label=batch.pop('anomaly_labels');assert int(label.item())==int(row['scout_label']=='abnormal')
                    batch.update(tune_mode='default',return_anomaly_evidence=True);attention.scope=probe.scope='forward'
                    budget.reserve('forward',f'{out}:{key}:{j}')
                    with RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:result=model(**batch)
                    maps=observer.finalize(result.anomaly_evidence);e=compute_lce(maps['abnormal_raw'],maps['normal_raw'],[row['scout_label']])
                    shifted=batch['labels'][:,1:];valid=shifted!=-100
                    raw=dict(maps=cpu_tree({k:maps[k] for k in ('abnormal_raw','normal_raw')}),lm_logits=cpu_tree(result.logits[:,:-1,:][valid]),targets=cpu_tree(shifted[valid]))
                    original=dict(sample_id=row['sample_id'],sha256=row['sha256'],label=row['scout_label'],exposure_id=row['scout_exposure_id'],
                        lm_loss=float(result.loss.detach()),lce_loss=float(e['per_image_loss'][0].detach()),lce_score=float(e['score'][0].detach()),
                        effective_support=float(e['effective_support'][0].detach()),top11_mass=float(e['top11_mass'][0].detach()))
                    analysis=response(raw['maps'],row['scout_label']);analysis['lm_loss']=float(torch.nn.functional.cross_entropy(raw['lm_logits'].double(),raw['targets']))
                    saved=save_raw(out/f'{key}-image{j}.pt.gz',raw)
                    rec=dict(index=index,input_sha256=inputs[-1],maps_sha256=state_digest(raw['maps']),original=original,analysis=analysis,raw=saved)
                    write(out/f'{key}-image{j}.json',rec);rows.append(rec);raw_rows.append(raw['maps'])
                    assert not any(p.grad is not None for p in model.parameters())
                    del result,maps,e,batch,raw;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
                input_list_sha=state_digest(input_trees);assert input_list_sha==r['inputs_sha256']
                after_rng=rng_state_fingerprint(snapshot_rng_state());assert after_rng==r['rng']['after_block']
                if key=='baseline':
                    gates=dict(images=[x['original'] for x in rows]==r['images'],maps=state_digest(raw_rows)==state_digest(original_maps),rng=after_rng==r['rng']['after_block'],inputs=input_list_sha==r['inputs_sha256'])
                    write(out/'baseline-gate.json',dict(status='PASS' if all(gates.values()) else 'FAIL',gates=gates))
                    assert all(gates.values()),gates
                else:assert inputs==[x['input_sha256'] for x in rows_by_variant['baseline']]
                assert state_digest(trainable(model))==entry
                variant_info[key]=dict(trainable_sha256=entry,runtime_entry_sha256=runtime_entry,runtime_exit_sha256=state_digest(runtime_state(model)),rng_before=rng_state_fingerprint(rng),rng_after=after_rng,inputs=inputs,input_list_sha256=input_list_sha)
                raw_by_variant[key]=raw_rows;rows_by_variant[key]=rows
                write(out/f'{key}-state.json',variant_info[key]);print(json.dumps(dict(anchor=a.anchor,variant=key,forwards=3)),flush=True)
                del weights,input_trees;gc.collect();torch.cuda.empty_cache()
        install(model,weights_before,runtime,rng);assert full_state(model)==expected_full
        ar=audit.result();assert ar['status']=='SUCCESS';write(out/'file-open-audit.json',ar)
        write(out/'result.json',dict(status='PASS',anchor=a.anchor,attempt=a.attempt,code_commit=a.code_commit,aliases=aliases,variants=variant_info,
            rows=rows_by_variant,forwards=3*(1+len(reps)),real_updates=0,backward=0,complete_anchor_state_restored=True,
            prior_inputs_checked=ins.checked,candidate_inputs_checked=ins.wchecked,historical_write_violations=guard.violations,
            seconds=time.time()-started,finished_epoch=time.time()))
    finally:
        probe.close();attention.close();m=monitor.stop();write(out/'monitor-summary.json',m);assert m['peak_memory_used_mib']<=22500
if __name__=='__main__':main()
