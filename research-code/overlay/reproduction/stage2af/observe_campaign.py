"""Compact read-only observation of the sealed stage2af coordinator and jobs."""
import argparse
import json
from reproduction.stage2af.remote import run_script


SCRIPT = r'''
import json,sys
from pathlib import Path
root=Path('/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1')
sys.path.insert(0,str(root/'code/runtime'))
from reproduction.stage2af.budget import Budget
def read(p):return json.loads(p.read_text())
def process(folder):
 p=folder/'process.json'
 if not p.exists():return {'registered':False}
 r=read(p);pid=r.get('child_pid',r.get('pid'));stat=Path('/proc')/str(pid)/'stat'
 live=False;state=None
 try:
  fields=stat.read_text().rsplit(')',1)[1].split();state=fields[0]
  live=fields[19]==str(r['proc_start_ticks']) and state!='Z'
 except FileNotFoundError:pass
 result={'pid':pid,'start_ticks':r['proc_start_ticks'],'same_process_live':live,'state':state,'start_epoch':r['start_epoch']}
 if (folder/'exit.json').exists():
  e=read(folder/'exit.json');result['exit']={k:e.get(k) for k in ['exit_code','wall_seconds','budget_stop','peak_memory_MiB']}
 return result
driver=process(root/'control'/DRIVER)
events=root/'control/driver-events.jsonl'
launches=[json.loads(s) for s in events.read_text().splitlines()] if events.exists() else []
job=launches[-1]['name'] if launches else None
current={'name':job,**process(root/'control'/job)} if job else None
if job:
 if job.startswith('eval-') and '-v' in job:
  endpoint,attempt=job[5:].rsplit('-v',1)
  folder=root/'evaluation'/endpoint/('v'+attempt)
  current['written_prediction_records']=len(list(folder.glob('record-????.json')))
  current['expected_prediction_records']=128 if endpoint.endswith('development') else 542
 log=root/'logs'/(job+'.log')
 if log.exists():
  with log.open('rb') as f:f.seek(max(0,log.stat().st_size-3000));tail=f.read().decode(errors='replace')
  lines=[s for s in tail.splitlines() if s.startswith('{')]
  if lines:
   try:current['latest_progress']=json.loads(lines[-1])
   except json.JSONDecodeError:pass
  if current.get('exit',{}).get('exit_code',0):current['failure_tail']=tail[-1600:]
attemptfile=root/'accepted-training-attempts.json'
attempts=read(attemptfile) if attemptfile.exists() else {}
trajectories={}
for name,attempt in attempts.items():
 d=root/'formal'/name/attempt
 if not d.exists():trajectories[name]={'status':'NOT_STARTED','attempt':attempt};continue
 steps=sorted(d.glob('step????.json'))
 row={'attempt':attempt,'completed_steps':len(steps),'checkpoint_count':len(list(d.glob('checkpoint-step????.pt')))}
 if (d/'summary.json').exists():row['status']=read(d/'summary.json')['status']
 elif (d/'failure.json').exists():row['status']=read(d/'failure.json')['status']
 else:row['status']='PROCESSING' if current and current['same_process_live'] and name in job else 'NO_LIVE_PROCESS_CONFIRMED'
 trajectories[name]=row
evaluation=root/'evaluation'
readouts=list(evaluation.glob('*/v*/summary.json'))+list(evaluation.glob('linear-head/*/v*/summary.json'))
print(json.dumps({'driver':driver,'current_job':current,'training':trajectories,'evaluation_readouts_complete':len(readouts),'final_released':(root/'final-evaluation-release.json').exists(),'resources':Budget(root).snapshot()}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', default='driver-v2')
    args = parser.parse_args()
    result = run_script(SCRIPT.replace('DRIVER', repr(args.driver)), 30)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    print(json.dumps(json.loads(result.stdout)), flush=True)


if __name__ == '__main__':
    main()
