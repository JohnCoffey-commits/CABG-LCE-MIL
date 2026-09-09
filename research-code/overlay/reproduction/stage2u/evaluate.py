"""Fixed post-training development observations; no calibration or selection."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import torch
from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit,NvidiaSmiMonitor
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2m.constants import TRAINABLE_NAMES
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2p import INITIAL_STATE_SHA256
from reproduction.stage2q.execution import AttentionExecution,cpu_tree
from reproduction.stage2q.run import canonicalize_dormant,full_state
from reproduction.stage2s.common import setup_flags,write
from reproduction.stage2s.resize import ResizeProbe
from reproduction.stage2u.common import Budget,save_raw,source_identity,provenance,read_plan,require_execution,ARMS
from reproduction.stage2t.preflight import eval_contract

def bind_data_registry(annotation,image_root):
    """Bind before import; fail before model loading if a stale registry exists."""
    os.environ['MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION']=str(annotation)
    os.environ['MEDIC_AD_STAGE2H_IMAGE_ROOT']=str(image_root)
    registry=importlib.import_module('qwenvl.data')
    actual=registry.MEDIC_AD_STAGE2H_CALIBRATION
    if actual['annotation_path']!=str(annotation) or actual['data_path']!=str(image_root):
        raise RuntimeError('Data registry was imported before the requested binding; use a fresh process')
    return dict(actual)

def lm_terms(logits,labels):
    shifted=logits[:,:-1,:].float();targets=labels[:,1:];mask=targets!=-100
    selected=shifted[mask];target=targets[mask]
    assert target.numel()>0
    return {'logsumexp':torch.logsumexp(selected,dim=-1).cpu(),
            'target_logit':selected.gather(-1,target[:,None]).squeeze(-1).cpu(),
            'valid_tokens':int(target.numel())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--code-commit',required=True);a=p.parse_args()
    require_execution(a);a.cursor=24;plan,source=read_plan(a.plan,a.code_commit)
    # Both registered trajectories must finish before any dev image is loaded.
    for phase in ARMS:
        r=json.loads((a.campaign/phase/'result.json').read_text());assert r['end']==24 and r['status']=='SUCCESS'
    a.output.mkdir(exist_ok=False);setup_flags();budget=Budget(a.campaign)
    prov=provenance(plan,a.arm)
    split=json.loads(Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json').read_text());rows=eval_contract(split)
    payload,check=load_strict(a.checkpoint,prov);assert payload['sampler_state']['cursor']==a.cursor
    write(a.output/'provenance.json',prov);write(a.output/'source.json',source)
    allowed=[str((Path(split['image_root'])/r['relative_path']).resolve()) for r in rows]
    audit=MedicalImageOpenAudit(Path(split['image_root']),allowed);audit.install()
    attention=AttentionExecution('canonical_flash_deterministic').install();probe=ResizeProbe(deterministic=True).install()
    monitor=NvidiaSmiMonitor(a.output/'nvidia-smi.csv');monitor.start()
    try:
        write(a.output/'data-binding.json',bind_data_registry(split['eval_annotation'],split['image_root']))
        from reproduction.stage2h.runtime import load_stage2h_runtime,make_stage2h_dataset,move_batch_to_device
        base=Path('/home/checkpoints/Lingshu-7B')
        wrapper,loaded=load_stage2h_runtime(base,seed=42,training=False,gradient_checkpointing=False)
        model=wrapper.llm;assert canonical_json_sha256(loaded['step0_state'])==INITIAL_STATE_SHA256
        write(a.output/'dormant.json',canonicalize_dormant(model));named=dict(model.named_parameters())
        with torch.no_grad():
            for n in TRAINABLE_NAMES:named[n].copy_(payload['adapter_state'][n].to(wrapper.device))
        model.eval();initial=full_state(model);write(a.output/'full-initial-state.json',initial)
        _,dataset,collator=make_stage2h_dataset(base,'medic_ad_stage2h_calibration',shuffle=False)
        assert [(r.get('cabg_sample_id'),r.get('image_sha256'),r.get('scout_eval_rank')) for r in dataset.list_data_dict]==[(r['sample_id'],r['sha256'],r['scout_eval_rank']) for r in rows]
        predictions=[]
        for index,row in enumerate(rows):
            budget.reserve('eval',f'{a.output.name}:{index}')
            batch=move_batch_to_device(collator([dataset[index]]),wrapper.device)
            assert int(batch.pop('anomaly_labels').item())==int(row['scout_label']=='abnormal')
            batch.update(tune_mode='default',return_anomaly_evidence=True,use_cache=False)
            with torch.inference_mode(),RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
                result=model(**batch)
            maps=observer.finalize(result.anomaly_evidence)
            e=compute_lce(maps['abnormal_raw'],maps['normal_raw'],[row['scout_label']])
            with torch.inference_mode():terms=lm_terms(result.logits,batch['labels'])
            saved=save_raw(a.output/f'image{index:02}.pt.gz',{'maps':cpu_tree({k:maps[k] for k in ('abnormal_raw','normal_raw')}),'lm_terms':terms})
            rec={'sample_id':row['sample_id'],'sha256':row['sha256'],'label':row['scout_label'],
                 'score':float(e['score'][0]),'lce_loss':float(e['per_image_loss'][0]),
                 'effective_support':float(e['effective_support'][0]),'top11_mass':float(e['top11_mass'][0]),
                 'lm_loss':float(result.loss),'valid_tokens':terms['valid_tokens'],'raw':saved}
            assert all(torch.isfinite(torch.tensor(rec[k])) for k in ('score','lce_loss','effective_support','top11_mass','lm_loss'))
            predictions.append(rec);write(a.output/f'image{index:02}.json',rec)
            print(json.dumps({'output':a.output.name,'image':index,'support':rec['effective_support']}),flush=True)
            del result,maps,e,batch,terms;gc.collect();torch.cuda.empty_cache()
        final=full_state(model);assert final['parameters']==initial['parameters']
        write(a.output/'full-final-state.json',final);write(a.output/'predictions.json',predictions)
        write(a.output/'metrics.json',summarize_metrics(predictions))
        ar=audit.result();assert ar['status']=='SUCCESS';write(a.output/'file-open-audit.json',ar)
        write(a.output/'attention.json',{'counts':dict(attention.counts),'no_backward_expected':True})
        write(a.output/'result.json',{'status':'SUCCESS','arm':a.arm,'cursor':a.cursor,'checkpoint_sha256':check['checkpoint_sha256'],
              'adapter_sha256':canonical_json_sha256(trainable_state_record(model)),'count':32,'updates':0,'protected_internal_test_opened':0})
    finally:
        probe.close();attention.close();m=monitor.stop();write(a.output/'monitor-summary.json',m)
        if m['peak_memory_used_mib']>22500:raise RuntimeError('VRAM ceiling')
if __name__=='__main__':main()
