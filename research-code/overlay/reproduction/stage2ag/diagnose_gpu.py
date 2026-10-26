"""Server-only, zero-update, fixed-checkpoint development interventions."""
import argparse,hashlib,json,os,platform,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[43,44],required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    assert platform.system()=='Linux' and Path('/home/Medic-AD').is_dir(),'SERVER ONLY; no CPU/MPS fallback'
    import numpy as np
    import torch
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0),'Authorized L4 required'
    from reproduction.stage2af.runtime import Runtime,write,sha_file
    from reproduction.stage2af.inputs import load_prepared,DECODE,tensor_sha,read_rows
    from reproduction.stage2af.evaluate import load_adapter
    from reproduction.stage2af.metrics import parse_answer,score_maps
    from reproduction.stage2af.guard import InputGuard
    from reproduction.stage2af.frozen import require_frozen
    from reproduction.stage2h.state_audit import trainable_state_record
    from reproduction.stage2s.common import restore_runtime
    from reproduction.stage2ag.interventions import Intervention,CONDITIONS
    from reproduction.stage2ag.budget import Budget
    plan,protocol=require_frozen(a.old_root)
    rows=[x for x in read_rows(a.old_root/'data/prepared-v1/manifest.jsonl') if x['split']=='development']
    assert len(rows)==128 and sum(x['target'] for x in rows)==64
    guard=InputGuard(a.old_root,rows).install()
    out=a.root/'diagnostics'/f'seed{a.seed}'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    budget=Budget(a.root);budget.check();started=time.time()
    r=Runtime(a.old_root,a.seed,training=False)
    checkpoint=a.old_root/'formal'/f's{a.seed}-fixed_lce'/('attempt-v1' if a.seed==43 else 'attempt-v2')/'checkpoint-step1000.pt'
    cp=load_adapter(r,checkpoint,protocol,'fixed_lce',plan['fixed_coefficients'][str(a.seed)])
    before=trainable_state_record(r.model);versions={n:p._version for n,p in r.model.named_parameters()}
    write(out/'identity.json',dict(checkpoint=cp,initial_loaded=before,seed=a.seed,input_manifest_sha256=sha_file(a.old_root/'data/prepared-v1/manifest.jsonl'),
          torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),host=platform.node(),pid=os.getpid()))
    prior_root=a.old_root/'evaluation'/f's{a.seed}-fixed_lce-step1000-development'/'v1'
    prior={x['sample_id']:x for x in json.loads((prior_root/'observations.json').read_text())}
    ordered=sorted([x['sample_id'] for x in rows],key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    donors={sid:ordered[(i+1)%len(ordered)] for i,sid in enumerate(ordered)}
    write(out/'donors.json',donors)
    probe=Intervention(r.model).install();native_tokens={};native_records={};native_trace={};all_records=[];noops=[]
    def observe(row,condition,index):
        budget.check();probe.condition=condition;probe.donor=native_tokens.get(donors[row['sample_id']]);probe.trace={};probe.capture=True
        restore_runtime(r.model,r.clean_runtime);inputs=load_prepared(row,r.wrapper.device)
        assert 'labels' not in inputs and 'anomaly_labels' not in inputs
        t=time.time()
        with torch.inference_mode():res=r.model(**inputs,tune_mode='default',return_anomaly_evidence=True,use_cache=False,diff_mode=False)
        trace=probe.trace
        yes=r.contract['answer_suffix_tokens']['Yes'][0];no=r.contract['answer_suffix_tokens']['No'][0]
        yn=res.logits[0,-1,[yes,no]].detach().float().cpu().numpy()
        scores=score_maps(torch.from_numpy(trace['a']).squeeze(2),torch.from_numpy(trace['n']).squeeze(2))
        rec={k:row[k] for k in ['sample_id','target','split','sha256','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']}
        rec.update(condition=condition,index=index,seed=a.seed,yes_no_logits=yn.tolist(),yes_no_logit_gap=float(yn[0]-yn[1]),
                   fed_direct_sha256=tensor_sha(probe.fed_direct),fed_ano_sha256=hashlib.sha256(trace['fed_ano'].tobytes()).hexdigest(),**scores)
        if condition=='native':
            native_tokens[row['sample_id']]=probe.last_combined.clone()
            native_trace[row['sample_id']]={k:v.copy() for k,v in trace.items() if k in ['replacement_e','query','attention','pre_ffn','post_ffn','fed_ano']}
            old=prior[row['sample_id']]
            with np.load(prior_root/old['raw_file'],allow_pickle=False) as z:
                assert np.array_equal(trace['a'].squeeze(2),z['abnormal'])
                assert np.array_equal(trace['n'].squeeze(2),z['normal'])
                assert np.array_equal(yn,z['yes_no_logits'])
        elif condition in ('ano_swap','visual_swap'):rec['donor_sample_id']=donors[row['sample_id']]
        if condition!='native':
            orig=native_trace[row['sample_id']]
            rec['relative_change']={k:float(np.linalg.norm(trace[k].astype('float64')-orig[k])/max(np.linalg.norm(orig[k]),1e-30)) for k in orig}
        del res
        probe.capture=False;restore_runtime(r.model,r.clean_runtime)
        with torch.inference_mode():generated=r.model.generate(**inputs,**DECODE)
        tokens=generated[0,inputs['input_ids'].shape[1]:].cpu().tolist()
        text=r.wrapper.processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        rec.update(output_ids=tokens,text=text,parsed=parse_answer(text),seconds=time.time()-t)
        if condition=='native':
            assert tokens==prior[row['sample_id']]['output_ids'] and rec['parsed']==prior[row['sample_id']]['parsed']
            native_records[row['sample_id']]=rec
        if condition=='clone':
            old=native_records[row['sample_id']]
            assert tokens==old['output_ids'] and rec['yes_no_logits']==old['yes_no_logits']
            assert all(v==0 for v in rec['relative_change'].values())
            return rec
        folder=out/condition;folder.mkdir(exist_ok=True)
        raw=folder/f'trace-{index:04}.npz'
        with raw.open('xb') as f:np.savez_compressed(f,**trace)
        rec.update(trace_file=str(raw.relative_to(out)),trace_sha256=sha_file(raw))
        write(folder/f'record-{index:04}.json',rec)
        return rec
    try:
        for condition in CONDITIONS:
            for i,row in enumerate(rows):
                rec=observe(row,condition,i);all_records.append(rec)
                if condition=='native' and row['sample_id'] in ordered[:4]:noops.append(observe(row,'clone',i))
                if (i+1)%32==0:print(json.dumps(dict(seed=a.seed,condition=condition,done=i+1,total=128,resources=budget.snapshot())),flush=True)
        after=trainable_state_record(r.model)
        assert after==before and versions=={n:p._version for n,p in r.model.named_parameters()}
        assert all(p.grad is None for p in r.model.parameters())
        write(out/'observations.json',all_records);write(out/'no-op-checks.json',noops)
        write(out/'file-open-audit.json',guard.record());assert not guard.violations
        write(out/'summary.json',dict(status='PRODUCER_COMPLETE_REQUIRES_INDEPENDENT_REDUCTION',records=len(all_records),native_exact=128,noops_exact=len(noops),optimizer_updates=0,
              parameters_preserved=True,seconds=time.time()-started,resources=budget.snapshot()))
    finally:probe.close();r.close()

if __name__=='__main__':main()
