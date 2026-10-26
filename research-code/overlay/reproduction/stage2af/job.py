"""Exclusive stage-owned process supervisor with persistent cumulative limits."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from reproduction.stage2af.budget import Budget,BudgetExceeded

def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--gpu',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.command[:1]==['--']:a.command=a.command[1:]
    out=a.root/'control'/a.name;out.mkdir(exist_ok=False);budget=Budget(a.root)
    with (a.root/'control/active-job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);budget.check()
        if a.gpu:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if active:raise RuntimeError('Existing GPU task: '+active)
        # Preserve the exact stage sources seen by this process before launch.
        source_root=a.root/'code/runtime/reproduction/stage2af';snap=out/'source';snap.mkdir()
        import hashlib
        identities={}
        for source in source_root.glob('*.py'):
            body=source.read_bytes();(snap/source.name).write_bytes(body);identities[source.name]=hashlib.sha256(body).hexdigest()
        write(out/'source-identity.json',identities)
        start=time.time();child=subprocess.Popen(a.command,stdin=subprocess.DEVNULL,start_new_session=True)
        record={'name':a.name,'wrapper_pid':os.getpid(),'child_pid':child.pid,
                'proc_start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                'start_epoch':start,'command':a.command,'gpu':a.gpu}
        write(out/'process.json',record)
        if a.gpu:budget.record('gpu_start',id=a.name,pid=child.pid,proc_start_ticks=record['proc_start_ticks'])
        reason=None;last_monitor=0;peak=0
        try:
            while child.poll() is None:
                try:budget.check()
                except BudgetExceeded as exc:
                    reason=str(exc);os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                if a.gpu and time.time()-last_monitor>=15:
                    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
                    if r.returncode==0:
                        memory,util=(int(x.strip()) for x in r.stdout.split(','));peak=max(peak,memory)
                        with (out/'gpu-monitor.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'memory_MiB':memory,'utilization_percent':util})+'\n')
                    last_monitor=time.time()
                time.sleep(2)
            code=child.wait()
        except BaseException as exc:
            reason='Supervisor failure: '+repr(exc)
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            write(out/'supervisor-failure.json',{'error':repr(exc),'child_stopped':True})
            raise
        finally:
            if child.poll() is not None:
                end=time.time()
                if a.gpu:budget.record('gpu_end',id=a.name,pid=child.pid,exit_code=child.returncode)
                write(out/'exit.json',{'exit_code':child.returncode,'end_epoch':end,'wall_seconds':end-start,
                    'budget_stop':reason,'peak_memory_MiB':peak,'resources':budget.snapshot()})
        raise SystemExit(code)

if __name__=='__main__':main()
