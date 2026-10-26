"""Label-free scores and original-path greedy answers with identical cached inputs."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from reproduction.stage2af.runtime import Runtime,write,sha_file
from reproduction.stage2af.core import ARMS
from reproduction.stage2af.inputs import load_prepared,DECODE
from reproduction.stage2af.guard import InputGuard
from reproduction.stage2af.budget import Budget
from reproduction.stage2af.frozen import require_frozen
from reproduction.stage2af.metrics import parse_answer,score_maps,summary
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2l.constants import TRAINABLE_NAMES
from reproduction.stage2q.run import full_state
from reproduction.stage2s.common import restore_runtime

def load_adapter(runtime,path,protocol,arm,coefficient):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    assert sha_file(path)==meta['sha256'] and path.stat().st_size==meta['bytes']
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert (cp['seed'],cp['arm'],cp['coefficient'],cp['protocol_sha256'])==(runtime.seed,arm,coefficient,protocol)
    assert cp['initial_identity']==runtime.initial_identity
    named=dict(runtime.model.named_parameters())
    with torch.no_grad():
        for n in TRAINABLE_NAMES:named[n].copy_(cp['adapter'][n].to(named[n]))
    return meta

def observe(runtime,row,out,index):
    model=runtime.model;inputs=load_prepared(row,runtime.wrapper.device);prefix=inputs['input_ids'].shape[1]
    record={k:row[k] for k in ['sample_id','target','sha256','group_id','split','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']}
    restore_runtime(model,runtime.clean_runtime);t=time.time()
    with torch.inference_mode(),RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
        result=model(**inputs,tune_mode='default',return_anomaly_evidence=True,use_cache=False,diff_mode=False)
    maps=observer.finalize(result.anomaly_evidence);record.update(score_maps(maps['abnormal_raw'],maps['normal_raw']))
    # Both fixed answer alternatives are single tokenizer tokens; their raw
    # prompt-position logit gap is explanatory, not generated correctness.
    yes=runtime.contract['answer_suffix_tokens']['Yes'][0];no=runtime.contract['answer_suffix_tokens']['No'][0]
    logits=result.logits[0,-1].float();record['yes_no_logit_gap']=float(logits[yes]-logits[no])
    path=Path(out)/f'raw-{index:04}.npz'
    with path.open('xb') as f:np.savez_compressed(f,abnormal=maps['abnormal_raw'].float().cpu().numpy(),normal=maps['normal_raw'].float().cpu().numpy(),yes_no_logits=logits[[yes,no]].cpu().numpy())
    record.update(raw_file=path.name,raw_sha256=sha_file(path),score_seconds=time.time()-t)
    del result,maps,logits;restore_runtime(model,runtime.clean_runtime);t=time.time()
    with torch.inference_mode():generated=model.generate(**inputs,**DECODE)
    tokens=generated[0,prefix:].cpu().tolist();text=runtime.wrapper.processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    record.update(output_ids=tokens,text=text,parsed=parse_answer(text),generation_seconds=time.time()-t)
    del generated,inputs;gc.collect()
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[43,44],required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--step',type=int,choices=[24,100,300,1000],required=True);p.add_argument('--split',choices=['development','final'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--attempt',default='v1');a=p.parse_args()
    root=a.root;plan,protocol=require_frozen(root)
    if a.split=='final':assert a.step==1000 and (root/'final-evaluation-release.json').is_file(),'Final gate not released'
    name=f's{a.seed}-{a.arm}-step{a.step:04}-{a.split}';out=root/'evaluation'/name/a.attempt;out.mkdir(parents=True,exist_ok=False)
    budget=Budget(root);budget.check(space=500_000_000);r=Runtime(root,a.seed,training=False);rows=[x for x in r.rows if x['split']==a.split]
    guard=InputGuard(root,rows).install();started=time.time()
    try:
        coefficient=plan['fixed_coefficients'][str(a.seed)] if a.arm=='fixed_lce' else None
        meta=load_adapter(r,a.checkpoint,protocol,a.arm,coefficient);assert meta['step']==a.step
        write(out/'checkpoint.json',meta);before=full_state(r.model);write(out/'parameters-before.json',before);records=[]
        for i,row in enumerate(rows):
            budget.check();rec=observe(r,row,out,i);write(out/f'record-{i:04}.json',rec);records.append(rec)
            if (i+1)%25==0:print(json.dumps({'endpoint':name,'evaluated':i+1,'total':len(rows)}),flush=True)
        restore_runtime(r.model,r.clean_runtime);after=full_state(r.model);write(out/'parameters-after.json',after);assert after==before
        assert all(p.grad is None for p in r.model.parameters());write(out/'file-open-audit.json',guard.record());assert not guard.violations
        write(out/'observations.json',records)
        write(out/'summary.json',{'status':'PASS','endpoint':name,'metrics':summary(records),'protocol_sha256':protocol,
             'model_wall_seconds':time.time()-started,'parameter_and_buffer_identity_preserved':True,'checkpoint_sha256':meta['sha256']})
    finally:r.close()
if __name__=='__main__':main()
