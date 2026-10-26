"""Persistent stage2ag cumulative limits; no historical ledger reuse."""
import fcntl
import json
import os
import shutil
import time
from pathlib import Path


class Budget:
    def __init__(self, root): self.root=Path(root)
    def records(self):
        p=self.root/'events.jsonl'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    def event(self,kind,**data):
        with (self.root/'control/budget.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with (self.root/'events.jsonl').open('a') as f:
                f.write(json.dumps({'epoch':time.time(),'kind':kind,**data})+'\n');f.flush();os.fsync(f.fileno())
    def snapshot(self):
        now=time.time();events=self.records()
        starts={e['id']:e for e in events if e['kind']=='gpu_start'}
        ends={e['id']:e for e in events if e['kind']=='gpu_end'}
        assert len(starts)==sum(e['kind']=='gpu_start' for e in events)
        seconds=sum(ends.get(k,{'epoch':now})['epoch']-v['epoch'] for k,v in starts.items())
        first=min((e['epoch'] for e in starts.values()),default=None)
        return dict(epoch=now,gpu_process_seconds=seconds,first_gpu_epoch=first,
                    research_window_seconds=now-first if first else 0,
                    updates_reserved=sum(e.get('count',1) for e in events if e['kind']=='update_reserved'),
                    updates_completed=sum(e.get('count',1) for e in events if e['kind']=='update_completed'),
                    free_bytes=shutil.disk_usage(self.root).free,
                    open_gpu_jobs=sorted(set(starts)-set(ends)))
    def storage(self):
        total=0
        for root,dirs,files in os.walk(self.root,followlinks=False):
            dirs[:]=[d for d in dirs if not Path(root,d).is_symlink()]
            for name in files:
                p=Path(root,name)
                if not p.is_symlink(): total+=p.stat().st_size
        return total
    def check(self,storage=False):
        s=self.snapshot()
        if s['gpu_process_seconds']>=57600:raise RuntimeError('GPU process time cap')
        if s['research_window_seconds']>=129600:raise RuntimeError('Research window cap')
        if s['updates_reserved']>9000:raise RuntimeError('Update cap')
        if s['free_bytes']<50000000000:raise RuntimeError('Disk free floor')
        if storage and self.storage()>110000000000:raise RuntimeError('New storage cap')
        return s
