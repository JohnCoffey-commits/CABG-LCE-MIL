"""Persistent cumulative limits, including idle instance time and all retries."""
from __future__ import annotations
import fcntl
import json
import math
import os
import shutil
import time
from pathlib import Path

class BudgetExceeded(RuntimeError):pass

class Budget:
    def __init__(self,root):
        self.root=Path(root);self.base=json.loads((self.root/'resource-ledger.json').read_text())
        self.events=self.root/'control/events.jsonl';self.lock=self.root/'control/budget.lock'

    def records(self):
        return [json.loads(s) for s in self.events.read_text().splitlines()] if self.events.exists() else []

    def snapshot(self):
        now=time.time();events=self.records()
        process={e['id']:e for e in events if e['kind']=='gpu_start'}
        ends={e['id']:e for e in events if e['kind']=='gpu_end'}
        gpu_seconds=sum(max(0,ends.get(k,{'epoch':now})['epoch']-v['epoch']) for k,v in process.items())
        # An unclosed GPU process is conservatively counted through now even if
        # it disappeared; reconcile with authoritative exit evidence, never reset.
        reserved=[e for e in events if e['kind']=='update_reserved']
        completed=[e for e in events if e['kind']=='update_completed']
        downloads=self.root/'control/downloads.jsonl'
        download_bytes=sum(json.loads(s).get('bytes',0) for s in downloads.read_text().splitlines()) if downloads.exists() else 0
        elapsed=max(0,now-self.base['start_epoch'])
        billed_hours=math.ceil(elapsed/60)/60
        return {'epoch':now,'instance_seconds':elapsed,'estimated_cost_usd':billed_hours*self.base['pricing']['estimated_total_usd_per_hour'],
                'gpu_process_seconds':gpu_seconds,'updates_reserved':len(reserved),'updates_completed':len(completed),
                'extra_updates_reserved':sum(e['category']!='formal' for e in reserved),
                'download_bytes':download_bytes,'free_bytes':shutil.disk_usage(self.root).free}

    def check(self,space=0):
        s=self.snapshot()
        for bad,reason in [(s['estimated_cost_usd']>=49.95,'USD ceiling with closeout margin'),
                           (s['gpu_process_seconds']>=216000,'GPU process ceiling'),
                           (s['free_bytes']-space<50_000_000_000,'free-space floor'),
                           (s['download_bytes']>50_000_000_000,'download ceiling')]:
            if bad:raise BudgetExceeded(reason)
        return s

    def record(self,kind,**fields):
        with self.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if kind=='update_reserved':
                s=self.check()
                if s['updates_reserved']>=6100:raise BudgetExceeded('Total actual/reserved update ceiling')
                if fields['category']!='formal' and s['extra_updates_reserved']>=100:raise BudgetExceeded('Extra update ceiling')
                previous=[e for e in self.records() if e['kind']=='update_reserved' and e.get('trajectory')==fields.get('trajectory') and e.get('category')=='formal']
                if fields['category']=='formal' and len(previous)>=1000:raise BudgetExceeded('Per-trajectory formal update ceiling')
            event={'kind':kind,'epoch':time.time(),**fields}
            with self.events.open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
        return event
