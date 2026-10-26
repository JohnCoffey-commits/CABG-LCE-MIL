"""Three actual debug updates: continuous 1→2 and fresh-process 1→2 replay."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from reproduction.stage2af.runtime import Runtime,write,sha_file,save_gradients
from reproduction.stage2af.core import compose,apply_update
from reproduction.stage2af.budget import Budget
from reproduction.stage2af.guard import InputGuard
from reproduction.stage2s.common import state_digest,runtime_state,restore_runtime
from reproduction.stage2af.evaluate import observe
from reproduction.stage2af.exact_state import equal_state
from reproduction.stage2l.controller import DynamicController,compute_block_gradients
from reproduction.stage2l.constants import TRAINABLE_NAMES,S9_NAMES

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['continuous','resume'],required=True);a=p.parse_args()
    root=a.root;out=root/'debug'/('preflight-'+a.mode+'-v1');out.mkdir(exist_ok=False)
    protocol=sha_file(root/'engineering-protocol-v1.md');budget=Budget(root);budget.check(space=2_000_000_000)
    r=Runtime(root,43);guard=InputGuard(root,[x for x in r.rows if x['split']=='train']).install();started=time.time();times=[]
    try:
        write(out/'initial-state.json',r.initial_full)
        if a.mode=='resume':
            assert r.load(root/'debug/preflight-continuous-v1/step0001.pt',protocol,'cabg_lce',None)==1
            steps=[2]
        else:steps=[1,2]
        for step in steps:
            t=time.time();budget.check(space=800_000_000);lm,lce,images,inputs,rng=r.collect(step)
            applied,details=compose(lm,lce,[x['label'] for x in images],'cabg_lce',r.controller)
            if step==1:
                original,_=compute_block_gradients(lm,lce,[x['label'] for x in images],trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=DynamicController())
                assert all(torch.equal(applied[n],original[n]) for n in applied)
                for batch,row in zip(inputs,r.train_sequence[:3]):
                    prefix=len(row['input_ids'])
                    assert batch['input_ids'][0,:prefix].tolist()==row['input_ids']
                    assert batch['labels'][0,:prefix].eq(-100).all()
            raw=save_gradients(out/f'raw-step{step:04}.pt.gz',lm,lce,images)
            budget.record('update_reserved',trajectory='preflight-'+a.mode,category='debug',step=step)
            lr=apply_update(r.masters,r.optimizer,r.scheduler,applied,r.model)
            budget.record('update_completed',trajectory='preflight-'+a.mode,category='debug',step=step)
            checkpoint=r.save(out/f'step{step:04}.pt',step,'cabg_lce',None,protocol)
            rec={'step':step,'images':images,'rng':rng,'applied_digest':state_digest(applied),
                 'controller':details['budget'],'lr':lr,'raw':raw,'checkpoint':checkpoint,'seconds':time.time()-t}
            write(out/f'step{step:04}-record.json',rec);times.append(rec['seconds'])
            print(json.dumps({'debug_mode':a.mode,'step':step,'seconds':rec['seconds']}),flush=True)
            del lm,lce,inputs,applied,details;gc.collect()
        if a.mode=='continuous':
            old_runtime=r.clean_runtime;r.model.eval();r.clean_runtime=runtime_state(r.model)
            for i,row in enumerate(r.train_sequence[:2]):write(out/f'input-readout-{i}.json',observe(r,row,out,i))
            r.clean_runtime=old_runtime;restore_runtime(r.model,old_runtime)
        frozen=r.verify_frozen();write(out/'file-open-audit.json',guard.record())
        if a.mode=='resume':
            ref=root/'debug/preflight-continuous-v1'
            left=json.loads((ref/'step0002-record.json').read_text());right=json.loads((out/'step0002-record.json').read_text())
            for key in ['images','rng','applied_digest','controller','lr']:assert left[key]==right[key],key
            x=torch.load(ref/'step0002.pt',weights_only=False,map_location='cpu');y=torch.load(out/'step0002.pt',weights_only=False,map_location='cpu')
            equal_state(x,y)
        write(out/'summary.json',{'status':'PASS','mode':a.mode,'actual_updates':len(steps),'update_seconds':times,
             'wall_seconds':time.time()-started,'resume_exact':a.mode=='resume','frozen':frozen,'attention':r.attention.report()})
    finally:r.close()

if __name__=='__main__':main()
