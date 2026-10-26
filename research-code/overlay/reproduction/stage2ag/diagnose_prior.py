"""Server-only, fixed prior-odds diagnostic with actual unrestricted generation."""
import argparse,math,os,platform,time,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[43,44],required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    assert platform.system()=='Linux' and Path('/home/Medic-AD').is_dir(),'SERVER ONLY'
    import torch
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
    from transformers import LogitsProcessor,LogitsProcessorList
    from reproduction.stage2af.runtime import Runtime,write,sha_file
    from reproduction.stage2af.inputs import load_prepared,DECODE,read_rows
    from reproduction.stage2af.evaluate import load_adapter
    from reproduction.stage2af.metrics import parse_answer,summary,paired
    from reproduction.stage2af.guard import InputGuard
    from reproduction.stage2af.frozen import require_frozen
    from reproduction.stage2h.state_audit import trainable_state_record
    from reproduction.stage2s.common import restore_runtime
    from reproduction.stage2ag.budget import Budget
    plan,protocol=require_frozen(a.old_root)
    rows=[x for x in read_rows(a.old_root/'data/prepared-v1/manifest.jsonl') if x['split']=='development']
    assert len(rows)==128 and sum(x['target'] for x in rows)==64
    guard=InputGuard(a.old_root,rows).install();budget=Budget(a.root);budget.check()
    out=a.root/'diagnostics'/f'seed{a.seed}'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    r=Runtime(a.old_root,a.seed,training=False)
    checkpoint=a.old_root/'formal'/f's{a.seed}-fixed_lce'/('attempt-v1' if a.seed==43 else 'attempt-v2')/'checkpoint-step1000.pt'
    cp=load_adapter(r,checkpoint,protocol,'fixed_lce',plan['fixed_coefficients'][str(a.seed)])
    before=trainable_state_record(r.model);versions={n:p._version for n,p in r.model.named_parameters()}
    write(out/'identity.json',dict(seed=a.seed,checkpoint=cp,initial_loaded=before,host=platform.node(),pid=os.getpid(),preregistration_sha256=sha_file(a.root/'plan/follow-up-preregistration.md')))
    prior=json.loads((a.root/'diagnostics'/f'seed{a.seed}'/f'diagnostic-s{a.seed}-v1/observations.json').read_text())
    prior={x['sample_id']:x for x in prior if x['condition']=='native'}
    yes=r.contract['answer_suffix_tokens']['Yes'][0];no=r.contract['answer_suffix_tokens']['No'][0]
    class Prior(LogitsProcessor):
        def __init__(self,length,offset):self.length=length;self.offset=offset;self.calls=[]
        def __call__(self,ids,scores):
            step=ids.shape[1]-self.length;before=scores.clone()
            if step==0:scores=scores.clone();scores[:,yes]-=self.offset
            changed=torch.nonzero(scores!=before).cpu().tolist()
            if step==0:
                expected=[] if self.offset==0 else [[0,yes]]
                assert changed==expected and abs(float(before[0,yes]-scores[0,yes])-self.offset)<2e-6
                values=dict(before_yes_no=before[0,[yes,no]].float().cpu().tolist(),after_yes_no=scores[0,[yes,no]].float().cpu().tolist())
            else:
                assert changed==[];values={}
            assert torch.isfinite(scores).all()
            self.calls.append(dict(step=step,changed_indices=changed,**values));return scores
    # A real score-vector contract test exercises first/later positions and zero offset.
    trial=torch.zeros((1,r.model.config.vocab_size),device=r.wrapper.device,dtype=torch.float32)
    ids=torch.zeros((1,10),device=r.wrapper.device,dtype=torch.long)
    for offset in [0.,math.log(2)]:
        f=Prior(10,offset);g=f(ids,trial);assert torch.equal(f(torch.zeros((1,11),device=ids.device,dtype=ids.dtype),g),g)
    result=[];started=time.time()
    try:
        for i,row in enumerate(rows):
            budget.check();inputs=load_prepared(row,r.wrapper.device)
            assert 'labels' not in inputs and 'anomaly_labels' not in inputs
            for condition,offset in [('native',0.),('prior_odds',math.log(2))]:
                restore_runtime(r.model,r.clean_runtime);processor=Prior(inputs['input_ids'].shape[1],offset);t=time.time()
                with torch.inference_mode():generated=r.model.generate(**inputs,**DECODE,logits_processor=LogitsProcessorList([processor]))
                tokens=generated[0,inputs['input_ids'].shape[1]:].cpu().tolist()
                text=r.wrapper.processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                rec={k:row[k] for k in ['sample_id','target','split','sha256','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']}
                b=prior[row['sample_id']]
                rec.update(seed=a.seed,condition=condition,output_ids=tokens,text=text,parsed=parse_answer(text),seconds=time.time()-t,
                           score=b['score'],effective_support=b['effective_support'],top11_mass=b['top11_mass'],processor_calls=processor.calls,offset=offset,yes_token_id=yes,no_token_id=no)
                assert len(processor.calls)==len(tokens) and processor.calls[0]['step']==0
                if condition=='native':assert tokens==b['output_ids'] and rec['parsed']==b['parsed']
                result.append(rec)
                write(out/f'record-{i:04}-{condition}.json',rec)
            if (i+1)%32==0:print(json.dumps(dict(seed=a.seed,done=i+1,total=128,resources=budget.snapshot())),flush=True)
        assert trainable_state_record(r.model)==before and versions=={n:p._version for n,p in r.model.named_parameters()}
        assert all(p.grad is None for p in r.model.parameters())
        native=[x for x in result if x['condition']=='native'];changed=[x for x in result if x['condition']=='prior_odds']
        write(out/'observations.json',result);write(out/'file-open-audit.json',guard.record());assert not guard.violations
        write(out/'summary.json',dict(status='PRODUCER_COMPLETE_REQUIRES_INDEPENDENT_REDUCTION',records=len(result),native_exact=128,optimizer_updates=0,
              parameters_preserved=True,seconds=time.time()-started,endpoints={c:summary([x for x in result if x['condition']==c]) for c in ['native','prior_odds']},comparison=paired(changed,native,bootstrap_seed=a.seed)))
    finally:r.close()

if __name__=='__main__':main()
