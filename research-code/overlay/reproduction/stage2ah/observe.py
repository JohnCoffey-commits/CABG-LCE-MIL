"""Read-only status via the already authorized SSH master; no fallback."""
import json
from reproduction.stage2ah.ops import remote, ROOT


def main():
    print(remote(f'''from pathlib import Path
import json,time,subprocess,shutil
root=Path({ROOT!r})
events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
starts={{x['id']:x for x in events if x['kind']=='gpu_start'}}
ends={{x['id']:x for x in events if x['kind']=='gpu_end'}}
active=sorted(set(starts)-set(ends));progress={{}}
for name in ['s43-aligned','s43-shuffled','s44-aligned','s44-shuffled']:
 d=root/'formal'/name/'attempt-v1';steps=sorted(d.glob('step[0-9][0-9][0-9][0-9].json'))
 progress[name]=int(steps[-1].stem[4:]) if steps else 0
seconds=1199.4530730247498+sum(ends.get(k,{{'epoch':time.time()}})['epoch']-v['epoch'] for k,v in starts.items())
answer=dict(epoch=time.time(),active_jobs=active,steps=progress,updates=sum(x['kind']=='update_completed' for x in events),cumulative_seconds=seconds,
 free_GB=shutil.disk_usage(root).free/1e9,completed=(root/'campaign-complete.json').exists(),failure=(root/'campaign-failure.json').exists())
evaluation=list((root/'evaluation').glob('*/v1'))
answer['new_generated_records']=sum(len(list(p.glob('record-[0-9][0-9][0-9][0-9].json'))) for p in evaluation)
answer['new_completed_endpoints']=sum((p/'summary.json').exists() for p in evaluation)
if answer['failure']:answer['failure_detail']=json.loads((root/'campaign-failure.json').read_text())
if active:
 p=root/'logs'/(active[0]+'.log')
 if p.exists():answer['tail']=p.read_text()[-450:]
print(json.dumps(answer))
''').strip())


if __name__=='__main__':main()
