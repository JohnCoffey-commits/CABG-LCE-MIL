"""Six frozen long-horizon trajectories; scalar logs and sparse raw gradients."""
from __future__ import annotations
import argparse,gc,json,signal,time,traceback
from pathlib import Path
import torch
from reproduction.stage2af.runtime import Runtime,write,sha_file,save_gradients
from reproduction.stage2af.core import ARMS,compose,apply_update
from reproduction.stage2af.budget import Budget,BudgetExceeded
from reproduction.stage2af.guard import InputGuard
from reproduction.stage2af.frozen import require_frozen
from reproduction.stage2af.execution_amendment import load_execution_amendment,expected_frozen_parameters
from reproduction.stage2s.common import state_digest
from reproduction.stage2i.cabg_math import CABGContractError

class RequestedStop(RuntimeError):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[43,44],required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--attempt',default='v2');p.add_argument('--resume',type=Path);a=p.parse_args()
    root=a.root;plan,protocol,amendment,amendment_identity=load_execution_amendment(root);assert a.seed==44;trajectory=f's{a.seed}-{a.arm}'
    out=root/'formal'/trajectory/('attempt-'+a.attempt);out.mkdir(parents=True,exist_ok=False)
    write(out/'execution-amendment.json',{'sha256':amendment_identity,'original_protocol_sha256':protocol})
    budget=Budget(root);budget.check(space=6_000_000_000)
    coefficient=plan['fixed_coefficients'][str(a.seed)] if a.arm=='fixed_lce' else None
    r=None;last_step=0;records=[];started=time.time();last_checkpoint=None;reserved_now=False
    def stop(*_):raise RequestedStop('Supervisor requested budget stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        r=Runtime(root,a.seed);guard=InputGuard(root,[x for x in r.rows if x['split']=='train']).install()
        assert r.initial_identity==plan['initial_identities'][str(a.seed)]
        initial_parameters={x['name']:x for x in r.initial_full['parameters']}
        for name,expected in expected_frozen_parameters(plan,amendment,a.seed).items():assert initial_parameters[name]==expected,('Frozen base/init differs',name)
        assert r.initial_full['buffers']==plan['initial_buffers']
        write(out/'initial-state.json',r.initial_full)
        write(out/'sequence.json',r.train_sequence)
        if a.resume:last_step=r.load(a.resume,protocol,a.arm,coefficient)
        prior_reserved={e['step'] for e in budget.records() if e['kind']=='update_reserved' and e.get('trajectory')==trajectory}
        for step in range(last_step+1,1001):
            reserved_now=False;t=time.time();budget.check(space=800_000_000)
            lm,lce,images,inputs,rng=r.collect(step)
            labels=[x['label'] for x in images]
            applied,details=compose(lm,lce,labels,a.arm,r.controller,coefficient)
            raw=None
            if step in plan['sparse_gradient_steps']:
                raw=save_gradients(out/f'gradients-step{step:04}.pt.gz',lm,lce,images)
            category='retry' if step in prior_reserved else 'formal'
            before_state=r.controller.state_dict()
            budget.record('update_reserved',trajectory=trajectory,category=category,step=step,attempt=a.attempt);reserved_now=True
            lr=apply_update(r.masters,r.optimizer,r.scheduler,applied,r.model)
            budget.record('update_completed',trajectory=trajectory,category=category,step=step,attempt=a.attempt);reserved_now=False;last_step=step
            rec={'trajectory':trajectory,'step':step,'category':category,'images':images,'rng':rng,'lr':lr,
                 'lm_rms':details['lm_rms'],'lce_rms':details['lce_rms'],'controller':details['budget'],
                 'applied_coefficient':details['applied_coefficient'],'applied_dynamic_cap':a.arm=='cabg_lce','gradient':details['applied_gradient'],
                 'raw':raw,'applied_digest':state_digest(applied) if raw else None,
                 'seconds_before_checkpoint':time.time()-t}
            # Each complete step is its own immutable scalar file, enabling
            # duplicate/replayed steps to be reconciled without dropping evidence.
            write(out/f'step{step:04}.json',rec);records.append(rec)
            if step%50==0 or step==24:
                last_checkpoint=r.save(out/f'checkpoint-step{step:04}.pt',step,a.arm,coefficient,protocol)
            print(json.dumps({'trajectory':trajectory,'step':step,'lambda':rec['applied_coefficient'],
                   'lm_mean':sum(x['lm_loss'] for x in images)/3,'seconds':time.time()-t}),flush=True)
            del lm,lce,inputs,applied,details;gc.collect()
        frozen=r.verify_frozen();write(out/'file-open-audit.json',guard.record())
        assert not guard.violations
        write(out/'summary.json',{'status':'COMPLETE','trajectory':trajectory,'last_step':last_step,
             'checkpoint':last_checkpoint,'execution_amendment_sha256':amendment_identity,'wall_seconds':time.time()-started,'protocol_sha256':protocol,
             'frozen':frozen,'attention':r.attention.report(),'resources':budget.snapshot()})
    except BaseException as exc:
        if isinstance(exc,(BudgetExceeded,RequestedStop)):kind='RESOURCE_STOP'
        elif isinstance(exc,FloatingPointError) or (isinstance(exc,(CABGContractError,RuntimeError)) and any(x in str(exc).lower() for x in ['nonfinite','non-finite','numerics','rms must be finite','support is zero'])):kind='NUMERICAL_FAILURE'
        else:kind='ENGINEERING_FAILURE'
        write(out/'failure.json',{'status':kind,'trajectory':trajectory,'last_completed_step':last_step,
             'update_reserved_when_failed':reserved_now,'last_checkpoint':last_checkpoint,'error':repr(exc),
             'traceback':traceback.format_exc(),'resources':budget.snapshot()})
        raise
    finally:
        if r is not None:r.close()

if __name__=='__main__':main()
