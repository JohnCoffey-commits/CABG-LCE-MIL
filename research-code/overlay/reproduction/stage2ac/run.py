"""One fresh-process, fixed-membership diagnostic phase. Never trains CABG."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from reproduction.stage2ac.contracts import QUESTION, MAX_NEW_TOKENS, parse_answer, require_time, write
from reproduction.stage2aa.common import Inputs
from reproduction.stage2aa.evaluate import bind_data_registry, lm_terms
from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, tensor_sha256
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit, NvidiaSmiMonitor
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2m.constants import TRAINABLE_NAMES
from reproduction.stage2q.execution import AttentionExecution, cpu_tree
from reproduction.stage2q.run import canonicalize_dormant, full_state, save_raw
from reproduction.stage2s.common import setup_flags, runtime_state, restore_runtime, state_digest
from reproduction.stage2s.resize import ResizeProbe
from reproduction.stage2x.common import WriteGuard, forbidden


def reserve(root, plan, kind):
    require_time(plan)
    if shutil.disk_usage(root).free < 16_000_000_000:
        raise RuntimeError('Free-space floor reached')
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) >= 8_000_000_000:
        raise RuntimeError('New-artifact storage ceiling reached')
    limits = {'load':8, 'generation':32, 'bridge':32, 'feature':100}
    with (root / 'budget.json').open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        budget = json.load(f)
        if budget['counts'][kind] >= limits[kind]:
            raise RuntimeError('Diagnostic operation budget exhausted: ' + kind)
        if budget['model_wall_seconds'] >= 5400:
            raise RuntimeError('Diagnostic 90-minute model tranche exhausted')
        budget['counts'][kind] += 1
        budget['events'].append({'kind':kind, 'epoch':time.time()})
        f.seek(0); json.dump(budget,f,indent=2); f.truncate(); f.flush(); os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--phase',required=True)
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    if not a.execute:raise RuntimeError('--execute required')
    root=a.campaign.resolve();plan=json.loads((root/'plan.json').read_text())
    if a.phase not in plan['phases']:raise RuntimeError('Unregistered phase')
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==plan['code_commit']
    for name,h in plan['sources'].items():assert sha256_file(repo/name)==h,name
    out=root/a.phase;out.mkdir(exist_ok=False)
    guard=WriteGuard(root);setup_flags();reserve(root,plan,'load')
    start=time.time()
    remaining=min(5400-json.loads((root/'budget.json').read_text())['model_wall_seconds'],plan['deadline_epoch']-5400-time.time())
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic time ceiling')))
    signal.alarm(max(1,int(remaining)))
    torch.autograd.backward=forbidden
    torch.optim.AdamW.step=forbidden
    cfg=plan['phases'][a.phase];seed=cfg['seed'];ins=Inputs()
    rows=plan['feature_rows'] if cfg['mode']=='features' else plan['development_check']
    split=plan['split'];image_root=Path(split['image_root'])
    allowed=[str((image_root/r['relative_path']).resolve()) for r in rows]
    audit=MedicalImageOpenAudit(image_root,allowed);audit.install()
    for row in rows:assert sha256_file(image_root/row['relative_path'])==row['sha256']
    attention=AttentionExecution('canonical_flash_deterministic').install()
    probe=ResizeProbe(deterministic=True).install()
    monitor=NvidiaSmiMonitor(out/'nvidia-smi.csv');monitor.start()
    try:
        bind_data_registry(split['eval_annotation'],image_root)
        from reproduction.stage2h.runtime import load_stage2h_runtime,make_stage2h_dataset,move_batch_to_device
        base=Path('/home/checkpoints/Lingshu-7B')
        wrapper,loaded=load_stage2h_runtime(base,seed=seed,training=False,gradient_checkpointing=False)
        model=wrapper.llm;write(out/'dormant.json',canonicalize_dormant(model))
        initial=full_state(model)
        assert initial['parameters']==ins.js(f's{seed}-prefix/full-initial-state.json')['parameters']
        assert canonical_json_sha256(trainable_state_record(model))==ins.js(f's{seed}-prefix/initial-identity.json')['trainable_state_sha256']
        named=dict(model.named_parameters())
        if cfg['mode']=='answers':
            arm=cfg['arm'];phase=f's{seed}-'+{'none':'lm_only','full':'dynamic','early':'lce_off'}[arm]
            getpath,getjs,getraw=(ins.zpath,ins.zjs,ins.zraw) if arm=='none' else (ins.path,ins.js,ins.raw)
            payload,manifest=load_strict(getpath(phase+'/block24-checkpoint/checkpoint.pt'),getjs(phase+'/provenance.json'))
            assert payload['sampler_state']['cursor']==24
            with torch.no_grad():
                for name in TRAINABLE_NAMES:named[name].copy_(payload['adapter_state'][name].to(named[name]))
            write(out/'checkpoint.json',{'path':str(getpath(phase+'/block24-checkpoint/checkpoint.pt')),'sha256':manifest['checkpoint_sha256'],'provenance':payload['provenance']})
            del payload;gc.collect()
            historical=f"eval-s{seed}-"+{'none':'lm_only24','full':'dynamic24','early':'lce_off24'}[arm]
            old=getjs(historical+'/predictions.json')
            _,dataset,collator=make_stage2h_dataset(base,'medic_ad_stage2h_calibration',shuffle=False)
            rank={x['cabg_sample_id']:i for i,x in enumerate(dataset.list_data_dict)}
        else:
            assert all(torch.count_nonzero(p)==0 for p in model.visual.deep_prompt_embeddings)
            write(out/'representation.json',{'mode':'canonical_initial_zero_vpt_original_merged_visual_tokens_mean','no_lce_or_lm_adapters_loaded':True,'initial_trainable':trainable_state_record(model)})
        model.eval();before=full_state(model);runtime=runtime_state(model)
        write(out/'parameters-before.json',before)
        observations=[];features=[]
        for row in rows:
            restore_runtime(model,runtime)
            if cfg['mode']=='answers':
                reserve(root,plan,'bridge')
                index=rank[row['sample_id']]
                assert old[index]['sha256']==row['sha256']
                batch=move_batch_to_device(collator([dataset[index]]),wrapper.device)
                assert int(batch.pop('anomaly_labels').item())==int(row['scout_label']=='abnormal')
                batch.update(tune_mode='default',return_anomaly_evidence=True,use_cache=False)
                with torch.inference_mode(),RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
                    result=model(**batch)
                maps=observer.finalize(result.anomaly_evidence)
                raw={'maps':cpu_tree({k:maps[k] for k in ('abnormal_raw','normal_raw')}),'lm_terms':lm_terms(result.logits,batch['labels'])}
                oldraw=getraw(historical+f'/image{index:02}.pt.gz')
                save_raw(out/f'bridge-{index:02}.pt.gz',raw)
                assert state_digest(raw)==state_digest(oldraw),'Historical numerical bridge failed'
                bridge_pixels=tensor_sha256(batch['pixel_values'])
                del result,maps,oldraw,raw,batch
                restore_runtime(model,runtime)
            with Image.open(image_root/row['relative_path']) as image:
                inputs=wrapper.process_messages({'prompt':QUESTION,'image':image.convert('RGB')})
            assert 'labels' not in inputs and 'anomaly_labels' not in inputs
            input_ids=inputs.input_ids[0].tolist()
            context=wrapper.processor.tokenizer.decode(input_ids,skip_special_tokens=False)
            record={'sample_id':row['sample_id'],'sha256':row['sha256'],'target':int(row['scout_label']=='abnormal'),
                    'input_ids':input_ids,'context':context,'pixel_sha256':tensor_sha256(inputs['pixel_values']),
                    'image_grid_thw':inputs['image_grid_thw'].cpu().tolist()}
            require_time(plan);t=time.monotonic()
            with torch.inference_mode():
                if cfg['mode']=='answers':
                    reserve(root,plan,'generation')
                    generated=model.generate(**inputs,do_sample=False,temperature=0.,top_p=1.,repetition_penalty=1.,
                                             max_new_tokens=MAX_NEW_TOKENS,use_cache=True,tune_mode='default',diff_mode=False)
                    tail=generated[0,len(input_ids):].cpu().tolist()
                    text=wrapper.processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    record.update(output_ids=tail,text=text,parsed=parse_answer(text),bridge_index=index,bridge_exact=True,
                                  bridge_pixel_sha256=bridge_pixels,generation_pixels_equal_bridge=bridge_pixels==record['pixel_sha256'])
                    del generated
                else:
                    reserve(root,plan,'feature')
                    visual=model.visual(inputs['pixel_values'].to(model.visual.dtype),grid_thw=inputs['image_grid_thw'])
                    feature=visual[0].float().mean(0).cpu().numpy()
                    assert feature.ndim==1 and np.isfinite(feature).all()
                    features.append(feature);record.update(role=row['role'],feature_dimension=len(feature))
                    del visual
            record['elapsed_seconds']=time.monotonic()-t
            observations.append(record);write(out/f'record-{len(observations):03}.json',record)
            print(json.dumps({'phase':a.phase,'done':len(observations),'total':len(rows)}),flush=True)
            del inputs;gc.collect();torch.cuda.empty_cache()
        if features:
            with (out/'features.npy').open('xb') as f:np.save(f,np.stack(features),allow_pickle=False)
        restore_runtime(model,runtime)
        after=full_state(model);assert after['parameters']==before['parameters']
        assert all(p.grad is None for p in model.parameters())
        write(out/'parameters-after.json',after);write(out/'observations.json',observations)
        ar=audit.result();write(out/'file-open-audit.json',ar);assert ar['status']=='SUCCESS'
        assert not guard.violations
        write(out/'result.json',{'status':'PASS','phase':a.phase,'count':len(rows),'seed':seed,'config':cfg,
                                'optimizer_updates':0,'no_backward':True,'parameter_identity_preserved':True,
                                'protected_internal_test_opened':0,'code_commit':plan['code_commit'],
                                'prior_checked':dict(stage2v=ins.checked,stage2z=ins.zchecked)})
    finally:
        signal.alarm(0);probe.close();attention.close()
        measured=monitor.stop();write(out/'monitor-summary.json',measured)
        with (root/'budget.json').open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);b['model_wall_seconds']+=time.time()-start
            f.seek(0);json.dump(b,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())
        if measured['peak_memory_used_mib']>22500:raise RuntimeError('VRAM ceiling exceeded')


if __name__=='__main__':main()
