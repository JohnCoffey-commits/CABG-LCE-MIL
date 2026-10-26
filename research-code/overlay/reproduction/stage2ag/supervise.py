"""Own one uniquely named remote child, retain logs and enforce cumulative caps."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
from reproduction.stage2ag.budget import Budget

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--timeout',type=int,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert os.uname().sysname=='Linux' and Path('/home/Medic-AD').is_dir(),'SERVER ONLY'
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    out=a.root/'control'/a.name;out.mkdir(exist_ok=False);budget=Budget(a.root)
    with (a.root/'control/active-job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);budget.check(storage=True)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('Another GPU process is active: '+active)
        sources={str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.root/'code/runtime/reproduction/stage2ag').glob('*.py')}
        snap=out/'source';snap.mkdir()
        for f in (a.root/'code/runtime/reproduction/stage2ag').glob('*.py'):shutil.copy2(f,snap/f.name)
        write(out/'source.json',sources)
        start=time.time();child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,start_new_session=True)
        ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]
        write(out/'process.json',dict(id=a.name,supervisor_pid=os.getpid(),pid=child.pid,start_ticks=ticks,start_epoch=start,command=cmd))
        budget.event('gpu_start',id=a.name,pid=child.pid,start_ticks=ticks,epoch=start)
        reason=None;peak=0;last=0
        try:
            while child.poll() is None:
                budget.check()
                if time.time()-start>a.timeout:raise RuntimeError('Per-job timeout')
                if time.time()-last>=15:
                    budget.check(storage=True)
                    r=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                    mem,util=map(int,r.strip().split(','));peak=max(peak,mem)
                    with (out/'gpu.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),memory_MiB=mem,utilization=util))+'\n')
                    last=time.time()
                time.sleep(2)
        except BaseException as exc:
            reason=repr(exc)
            if child.poll() is None:
                # Child is still owned, unreaped, and its PID cannot be recycled.
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:
            code=child.wait();budget.event('gpu_end',id=a.name,pid=child.pid,exit_code=code)
            write(out/'exit.json',dict(exit_code=code,reason=reason,end_epoch=time.time(),seconds=time.time()-start,peak_MiB=peak,resources=budget.snapshot(),storage_bytes=budget.storage()))
        raise SystemExit(code if reason is None else 1)

if __name__=='__main__':main()
