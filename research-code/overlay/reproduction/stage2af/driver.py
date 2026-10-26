"""Serial finite campaign coordinator; engineering failures require a fresh diagnosis."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from reproduction.stage2af.frozen import require_frozen
from reproduction.stage2af.budget import Budget

def write(path,value):
    if Path(path).exists():
        assert json.loads(Path(path).read_text())==value,'Existing evidence differs: '+str(path)
        return
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def run_job(root,name,module,args,gpu=True):
    out=root/'control'/name
    if out.exists():
        if not (out/'exit.json').is_file():raise RuntimeError('Existing job has no terminal evidence: '+name)
        return json.loads((out/'exit.json').read_text())['exit_code']
    cmd=[sys.executable,'-B','-m','reproduction.stage2af.job','--root',str(root),'--name',name]+(['--gpu'] if gpu else [])+['--',sys.executable,'-B','-m',module,'--root',str(root)]+args
    with (root/'logs'/(name+'.log')).open('x') as f:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'control/driver-events.jsonl').open('a') as log:log.write(json.dumps({'epoch':time.time(),'event':'job_launched','name':name,'wrapper_pid':child.pid})+'\n')
        return child.wait()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',default='driver-v1');a=p.parse_args();root=a.root
    out=root/'control'/a.name;out.mkdir(exist_ok=False);started=time.time();code=1
    with (root/'control/driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/'process.json',{'pid':os.getpid(),'proc_start_ticks':Path('/proc/self/stat').read_text().split()[21],'start_epoch':started,'name':a.name})
        plan,protocol=require_frozen(root);budget=Budget(root);states={};accepted={}
        try:
            for seed in plan['seeds']:
                for arm in plan['arms']:
                    budget.check();trajectory=f's{seed}-{arm}';name='formal-'+trajectory+'-v1'
                    ret=run_job(root,name,'reproduction.stage2af.train',['--seed',str(seed),'--arm',arm])
                    d=root/'formal'/trajectory/'attempt-v1'
                    if ret==0:
                        s=json.loads((d/'summary.json').read_text());assert s['status']=='COMPLETE' and s['last_step']==1000;states[trajectory]=s
                    else:
                        failure=json.loads((d/'failure.json').read_text()) if (d/'failure.json').exists() else {}
                        if failure.get('status')=='NUMERICAL_FAILURE':states[trajectory]=failure
                        else:raise RuntimeError('Engineering/resource interruption; retain remaining work: '+trajectory)
            write(root/'training-terminal-status-v1.json',states)
            # Evaluate fixed dev checkpoints even for a trajectory with a later
            # numerical failure, provided the checkpoint was complete beforehand.
            for seed in plan['seeds']:
                for arm in plan['arms']:
                    for step in plan['evaluation_nodes']:
                        cp=root/'formal'/f's{seed}-{arm}'/'attempt-v1'/f'checkpoint-step{step:04}.pt'
                        if not cp.with_suffix('.json').exists():continue
                        endpoint=f's{seed}-{arm}-step{step:04}-development';name='eval-'+endpoint+'-v1'
                        ret=run_job(root,name,'reproduction.stage2af.evaluate',['--seed',str(seed),'--arm',arm,'--step',str(step),'--split','development','--checkpoint',str(cp)])
                        if ret:raise RuntimeError('Evaluation implementation failure: '+endpoint)
                        accepted[endpoint]=str(root/'evaluation'/endpoint/'v1')
            ret=run_job(root,'linear-fit-v1','reproduction.stage2af.linear_head',['--phase','fit'])
            if ret:raise RuntimeError('Fixed-head fit/extraction failure')
            accepted['linear-head-fit']=str(root/'evaluation/linear-head/fit/v1')
            # Final is released only after the full fixed training/dev/head phase,
            # without using their numerical values to choose a method/checkpoint.
            release=root/'final-evaluation-release.json'
            if not release.exists():write(release,{'epoch':time.time(),'protocol_sha256':protocol,'training_status_file':'training-terminal-status-v1.json','final_selection_rule':'step1000 for every completed trajectory; no metric-based selection','methods_unchanged':True})
            for seed in plan['seeds']:
                for arm in plan['arms']:
                    trajectory=f's{seed}-{arm}'
                    if states[trajectory].get('status')!='COMPLETE':continue
                    cp=root/'formal'/trajectory/'attempt-v1/checkpoint-step1000.pt';endpoint=trajectory+'-step1000-final'
                    ret=run_job(root,'eval-'+endpoint+'-v1','reproduction.stage2af.evaluate',['--seed',str(seed),'--arm',arm,'--step','1000','--split','final','--checkpoint',str(cp)])
                    if ret:raise RuntimeError('Final implementation failure; final already used: '+endpoint)
                    accepted[endpoint]=str(root/'evaluation'/endpoint/'v1')
            ret=run_job(root,'linear-final-v1','reproduction.stage2af.linear_head',['--phase','final'])
            if ret:raise RuntimeError('Final fixed-head extraction failure')
            accepted['linear-head-final']=str(root/'evaluation/linear-head/final/v1')
            write(root/'accepted-endpoints.json',accepted)
            for name,module in [('analysis-v1','analyze'),('result-verification-v1','verify_results')]:
                if run_job(root,name,'reproduction.stage2af.'+module,[],gpu=False):raise RuntimeError('Closure implementation failure: '+name)
            write(out/'summary.json',{'status':'EXECUTION_AND_METRIC_VERIFICATION_DONE_PENDING_COMPLETION_AUDIT','protocol_sha256':protocol,'states':states,'accepted_endpoints':accepted,'resources':budget.snapshot()});code=0
        except BaseException as exc:
            write(out/'interruption.json',{'status':'NEEDS_REVIEW','error':repr(exc),'traceback':traceback.format_exc(),'completed_states':states,'accepted_so_far':accepted,'resources':budget.snapshot()});raise
        finally:write(out/'exit.json',{'exit_code':code,'end_epoch':time.time(),'wall_seconds':time.time()-started})
if __name__=='__main__':main()
