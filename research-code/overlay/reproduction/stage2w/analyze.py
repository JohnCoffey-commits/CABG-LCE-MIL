"""All72 gradient records, no model/data reads or GPU work."""
import argparse,gc,json,time
from statistics import median
from reproduction.stage2w.common import *
from reproduction.stage2w.math import aggregates,geometry,corners

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    started=time.time();ins,guard=setup(a.output,a.code_commit);folder=a.output/'context';folder.mkdir()
    summary={};manifest=ins.js('train-manifest.json')
    for phase,steps in PHASES.items():
        summaries=[];previous=None
        for step in steps:
            r,d=ins.record(phase,step);lm,aux,actual=aggregates(d,r)
            assert r['seed'] in SEEDS and r['phase']==phase and r['real_updates']==1
            expected=manifest[(step-1)*3:step*3]
            assert [(x['sample_id'],x['sha256'],x['label']) for x in r['images']]==[(x['sample_id'],x['sha256'],x['scout_label']) for x in expected]
            if previous:assert r['previous_record_sha256']==previous['record_sha256'] and r['master_before']==previous['master_after'] and r['optimizer_before']==previous['optimizer_after']
            grads,clip=corners(lm,aux,r['policy']['shadow_lambda']);native='00' if phase.endswith('lce_off') else '11'
            assert state_digest(grads[native])==r['applied_sha256']==state_digest(actual)
            out=dict(phase=phase,step=step,raw_sha256=r['raw']['sha256'],record_sha256=r['record_sha256'],
                inputs_sha256=r['inputs_sha256'],geometry=geometry(lm,aux,r['policy']['shadow_lambda']),clip_off=clip[0],clip_dynamic=clip[1],
                policy=r['policy'],controller=r['controller'],images=r['images'],support_median=median(x['effective_support'] for x in r['images']),
                top11_median=median(x['top11_mass'] for x in r['images']))
            write(folder/f'{phase}-b{step}.json',out);summaries.append(dict(step=step,support=out['support_median'],top11=out['top11_median']))
            previous=r;del d,lm,aux,actual,grads;gc.collect();disk(a.output)
        summary[phase]=summaries;print(json.dumps(dict(phase=phase,records=len(summaries))),flush=True)
    write(a.output/'context-result.json',dict(status='PASS',records=72,phases=summary,checked_inputs=ins.checked,access=guard.report(),seconds=time.time()-started,model_loads=0,real_updates=0,image_forwards=0))
if __name__=='__main__':main()
