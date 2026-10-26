"""Local transport for this stage only; all remote execution uses the existing socket."""
import argparse
import json
from pathlib import Path
from reproduction.stage2af.remote import run_script
ROOT='/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sync','launch','campaign','poll']);p.add_argument('--name');p.add_argument('--gpu',action='store_true');p.add_argument('--module');a,extra_args=p.parse_known_args();a.extra=extra_args
    prefix="import pathlib,json,os,subprocess,hashlib\nroot=pathlib.Path("+repr(ROOT)+")\n"
    if a.action=='sync':
        files={x.name:x.read_text() for x in Path(__file__).parent.glob('*.py')}
        script=prefix+'''files=FILES
for name,body in files.items():
 p=root/'code/runtime/reproduction/stage2af'/name;p.parent.mkdir(parents=True,exist_ok=True)
 protocol=root/'protocol.json'
 if protocol.exists():
  protected=json.loads(protocol.read_text())['sources']
  if str(p) in protected and hashlib.sha256(body.encode()).hexdigest()!=protected[str(p)]:raise RuntimeError('Frozen producer source cannot be changed by sync: '+name)
 amendment=root/'engineering-amendment-02.json'
 if amendment.exists():
  protected=json.loads(amendment.read_text())['source_identities']
  if str(p) in protected and hashlib.sha256(body.encode()).hexdigest()!=protected[str(p)]:raise RuntimeError('Sealed amendment source cannot be changed by sync: '+name)
 if p.exists() and p.read_text()==body:continue
 if p.exists() and p.read_text()!=body:
  old=p.read_bytes();snapshot=root/'code/revisions'/hashlib.sha256(old).hexdigest()/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
  if not snapshot.exists():snapshot.write_bytes(old)
 p.write_text(body)
print(json.dumps({'synced':len(files)}))
'''.replace('FILES',repr(files))
    elif a.action in ['launch','campaign']:
        if a.action=='campaign':a.module=a.module or 'reproduction.stage2af.driver'
        assert a.name and a.module
        extra=a.extra[1:] if a.extra[:1]==['--'] else a.extra
        script=prefix+'''name=NAME;module=MODULE;gpu=GPU;extra=EXTRA;coordinator=COORDINATOR
if (root/'control'/name).exists():raise FileExistsError('Job already registered')
env=os.environ.copy()
for sub in ['tmp','hf-cache','torch-cache','xdg-cache','mpl-cache']:(root/'environment'/sub).mkdir(exist_ok=True)
env.update(PYTHONPATH=str(root/'code/runtime')+':'+str(root/'code/runtime/qwen-vl-finetune'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(root/'environment/hf-cache'),TORCH_HOME=str(root/'environment/torch-cache'),XDG_CACHE_HOME=str(root/'environment/xdg-cache'),MPLCONFIGDIR=str(root/'environment/mpl-cache'),TMPDIR=str(root/'environment/tmp'),TOKENIZERS_PARALLELISM='false')
python=str(root/'environment/venv/bin/python')
command=[python,'-B','-m','reproduction.stage2af.job','--root',str(root),'--name',name]+(['--gpu'] if gpu else [])+['--',python,'-B','-m',module,'--root',str(root)]+extra
if coordinator:command=[python,'-B','-m',module,'--root',str(root),'--name',name]
log=root/'logs'/(name+'.log')
with log.open('x') as stream:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=root/'code/runtime',start_new_session=True)
print(json.dumps({'wrapper_pid':child.pid,'log':str(log),'name':name}))
'''.replace('NAME',repr(a.name)).replace('MODULE',repr(a.module)).replace('GPU',repr(a.gpu)).replace('EXTRA',repr(extra)).replace('COORDINATOR',repr(a.action=='campaign'))
    else:
        assert a.name
        script=prefix+'''out=root/'control'/NAME
for name in ['process.json','exit.json']:
 p=out/name
 if p.exists():
  d=json.loads(p.read_text())
  if name=='process.json':
   proc=pathlib.Path('/proc')/str(d['child_pid']);d['same_process_live']=proc.exists() and (proc/'stat').read_text().split()[21]==d['proc_start_ticks']
  print(json.dumps({name:d}))
log=root/'logs'/(NAME+'.log')
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2200));print(f.read().decode(errors='replace'))
'''.replace('NAME',repr(a.name))
    r=run_script(script,45);print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
if __name__=='__main__':main()
