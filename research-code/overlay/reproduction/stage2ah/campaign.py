"""Server-owned sequential frozen campaign; stops on any failed child."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from reproduction.stage2ah.budget import Budget


def write(path, value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    assert os.uname().sysname=='Linux' and Path('/home/Medic-AD').is_dir()
    old='/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1'
    budget=Budget(root);budget.check(storage=True)
    forecast=json.loads((root/'short-trial-forecast.json').read_text())
    assert forecast['status']=='FEASIBLE' and forecast['both_prefixes_pass']
    jobs=[]
    def run(name,module,arguments,limit):
        budget.check(storage=True)
        out=root/'control'/(name+'-launcher.json')
        assert not out.exists(),'Never duplicate a registered job'
        with (root/'logs'/(name+'.log')).open('x') as log:
            cmd=[sys.executable,'-m','reproduction.stage2ah.supervise','--root',str(root),'--name',name,'--timeout',str(limit),'--',sys.executable,'-m',module,*arguments]
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out,dict(pid=proc.pid,epoch=time.time(),command=cmd,campaign_pid=os.getpid()))
            jobs.append(name);print(json.dumps(dict(started=name,pid=proc.pid)),flush=True)
            code=proc.wait()
        exit_record=json.loads((root/'control'/name/'exit.json').read_text())
        assert code==0 and exit_record['exit_code']==0 and exit_record['reason'] is None,(name,code,exit_record)
        print(json.dumps(dict(completed=name,seconds=exit_record['seconds'])),flush=True)
    try:
        for seed in [43,44]:
            for arm in ['aligned','shuffled']:
                args=['--root',str(root),'--old-root',old,'--seed',str(seed),'--arm',arm]
                trainargs=list(args)
                if seed==43:trainargs+=['--resume',str(root/'formal'/f's{seed}-{arm}'/'attempt-v1/checkpoint-step0024.pt')]
                run(f's{seed}-{arm}-formal-v1','reproduction.stage2ah.train',trainargs,7000)
                run(f's{seed}-{arm}-development-v1','reproduction.stage2ah.evaluate',args+['--split','development'],3600)
        complete={}
        for seed in [43,44]:
            for arm in ['aligned','shuffled']:
                name=f's{seed}-{arm}';p=root/'formal'/name/'attempt-v1/summary.json'
                value=json.loads(p.read_text());assert value['status']=='COMPLETE' and value['last_step']==1000
                complete[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        write(root/'final-evaluation-release.json',dict(epoch=time.time(),protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),complete_trajectories=complete,
              scope='One unified reused-test follow-up; no tuning or second readout',all_method_and_evaluation_rules_frozen_before_training=True))
        for seed in [43,44]:
            for arm in ['aligned','shuffled']:
                args=['--root',str(root),'--old-root',old,'--seed',str(seed),'--arm',arm,'--split','final']
                run(f's{seed}-{arm}-final-v1','reproduction.stage2ah.evaluate',args,3600)
        run('analysis-v1','reproduction.stage2ah.analyze',['--root',str(root)],1800)
        run('training-verification-v1','reproduction.stage2ah.verify_training',['--root',str(root)],3600)
        run('results-verification-v1','reproduction.stage2ah.verify_results',['--root',str(root)],3600)
        write(root/'campaign-complete.json',dict(status='ALL_SCHEDULED_JOBS_COMPLETE',epoch=time.time(),jobs=jobs,resources=budget.snapshot()))
    except BaseException as exc:
        write(root/'campaign-failure.json',dict(epoch=time.time(),error=repr(exc),traceback=traceback.format_exc(),jobs=jobs,resources=budget.snapshot()))
        raise


if __name__=='__main__':main()
