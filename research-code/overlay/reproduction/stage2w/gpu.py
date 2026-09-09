"""Two fresh processes: observed reference kernels, then four-corner clones."""
import argparse,copy,gc,json,math,time
from reproduction.stage2i.run_gate_d3 import NvidiaSmiMonitor
from reproduction.stage2q.run import save_raw
from reproduction.stage2w.common import *
from reproduction.stage2w.math import aggregates,corners,native,attribution,memory_attribution,decision

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);p.add_argument('--phase',choices=('reference','corners'),required=True);a=p.parse_args()
    if not a.execute:raise RuntimeError('Explicit --execute required before any arithmetic execution')
    ins,guard=open_existing(a.output,a.code_commit);assert read(a.output/'context-result.json')['status']=='PASS'
    assert torch.__version__=='2.7.0+cu126' and torch.version.cuda=='12.6'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()=='580.126.20'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    if a.phase=='corners':assert read(a.output/'reference-result.json')['status']=='PASS'
    folder=a.output/a.phase;folder.mkdir();budget=GpuBudget(a.output);started=time.time()
    monitor=NvidiaSmiMonitor(folder/'nvidia-smi.csv');monitor.start();results={}
    def forbidden(*_,**__):raise RuntimeError('No real optimizer.step permitted')
    torch.optim.AdamW.step=forbidden
    try:
        for name in ANCHORS:
            cfg,r,d,before,opt,refs=anchor(ins,name);lm,aux,actual=aggregates(d,r)
            gradients,scales=corners(lm,aux,r['policy']['shadow_lambda']);native_keys=tuple(refs)
            values={};records={};state_before=state_digest((before,opt))
            todo=native_keys if a.phase=='reference' else ('00','01','10','11')
            for key in todo:
                budget.count(a.phase,name,key)
                after,states=native(before,opt,gradients[key],'cuda')
                assert all(torch.isfinite(v).all() for v in after.values())
                record=dict(gradient_sha256=state_digest(gradients[key]),master_sha256=state_digest(after),bf16_sha256=state_digest({n:v.bfloat16() for n,v in after.items()}))
                if key in refs:
                    rr=refs[key]['record'];assert record['gradient_sha256']==rr['applied_sha256']
                    assert record['master_sha256']==rr['master_after'] and all(torch.equal(after[n],refs[key]['after'][n]) for n in TRAINABLE_NAMES)
                    nextopt=copy.deepcopy(opt);nextopt['state']=states;nextopt['param_groups'][0]['lr']=1e-4*.5*(1+math.cos(math.pi*cfg['step']/24))
                    assert state_digest(nextopt)==rr['optimizer_after'],'Native optimizer state mismatch'
                    record['native_master_and_optimizer_exact']=True
                if a.phase=='corners':
                    disk(a.output)
                    raw=save_raw(folder/f'{name}-{key}.pt.gz',dict(master_xor=encode_master(before,after)))
                    record['raw']=raw
                    if key in refs:
                        pilot=read(a.output/'reference'/f'{name}.json')['corners'][key]
                        assert all(record[k]==pilot[k] for k in ('gradient_sha256','master_sha256','bf16_sha256'))
                    values[key]=after
                records[key]=record
                del states;gc.collect();torch.cuda.empty_cache()
            assert state_digest((before,opt))==state_before,'Observed input state mutated'
            report=dict(anchor=name,configuration=cfg,input_state_sha256=state_before,record_sha256=r['record_sha256'],
                native_references=list(refs),clip_off=scales[0],clip_dynamic=scales[1],coefficient=r['policy']['shadow_lambda'],corners=records)
            if a.phase=='corners':
                report['attribution']=attribution(values)
                key='00' if cfg['phase'].endswith('lce_off') else '11'
                report['memory']=memory_attribution(before,opt,gradients[key],cfg['step'])
            write(folder/f'{name}.json',report);results[name]=report
            del values,gradients,actual,lm,aux,d,before,opt,refs;gc.collect();torch.cuda.empty_cache();disk(a.output)
            print(json.dumps(dict(phase=a.phase,anchor=name,clones=len(todo),elapsed=time.time()-started)),flush=True)
    finally:
        monitored=monitor.stop();write(folder/'monitor-summary.json',monitored)
    assert monitored['peak_memory_used_mib']<=8192 and monitored['status']=='SUCCESS'
    write(a.output/f'{a.phase}-result.json',dict(status='PASS',anchors=len(results),arithmetic_clones=12 if a.phase=='reference' else 40,
        reports=results,decision=decision(results) if a.phase=='corners' else None,checked_inputs=ins.checked,access=guard.report(),seconds=time.time()-started,
        real_updates=0,model_loads=0,image_forwards=0,monitor=monitored,finished_epoch=time.time()))
if __name__=='__main__':main()
