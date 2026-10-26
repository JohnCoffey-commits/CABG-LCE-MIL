"""Eight no-update training blocks per retained initialization; no held-out use."""
import argparse
import json
import gc
from pathlib import Path
from reproduction.stage2af.runtime import Runtime,write,save_gradients
from reproduction.stage2af.core import calibrate_initial_coefficient,fixed_coefficient
from reproduction.stage2af.budget import Budget
from reproduction.stage2af.guard import InputGuard

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[43,44],required=True);a=p.parse_args()
    out=a.root/'debug'/f'calibration-s{a.seed}-v1';out.mkdir(exist_ok=False);budget=Budget(a.root);budget.check(space=3_000_000_000)
    runtime=Runtime(a.root,a.seed);records=[]
    guard=InputGuard(a.root,[r for r in runtime.rows if r['split']=='train']).install()
    try:
        write(out/'initial-identity.json',runtime.initial_identity)
        for step in range(1,9):
            budget.check(space=300_000_000);runtime.restore_initial()
            lm,lce,images,inputs,rng=runtime.collect(step)
            record=calibrate_initial_coefficient(lm,lce,[r['label'] for r in images])
            record.update(step=step,images=images,raw=save_gradients(out/f'block{step}.pt.gz',lm,lce,images))
            write(out/f'block{step}.json',record);records.append(record)
            print(json.dumps({'seed':a.seed,'calibration_block':step,'coefficient':record['coefficient']}),flush=True)
            del lm,lce,inputs;gc.collect()
        runtime.restore_initial();frozen=runtime.verify_frozen();write(out/'file-open-audit.json',guard.record())
        write(out/'summary.json',{'status':'PASS','seed':a.seed,'rule':'median of8 initial-block min(rho*classRMSratio,rho_max*aggregateNormRatio)','coefficient':fixed_coefficient(records),'blocks':8,'actual_updates':0,'training_only':True,'frozen_check':frozen,'coefficients':[r['coefficient'] for r in records]})
    finally:runtime.close()

if __name__=='__main__':main()
