"""Local orchestration only; every computational job is launched on the existing SSH master."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path

WORKSPACE=Path(__file__).resolve().parents[3]
LOCAL=WORKSPACE/'document/cabg_mil_v1_2/evidence-transport-20260919-v1'
ROOT='/home/Medic-AD/outputs/cabg-mil-v1.2/evidence-transport-20260919-v1'
OLD='/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1'
PY=OLD+'/environment/venv/bin/python'
SSH=['ssh','-F','/dev/null','-S','/path/to/your/existing-ssh-control.sock','-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-T','USER@YOUR_EXECUTION_HOST']

def remote(script,timeout=50):
    r=subprocess.run(SSH+['python3 -'],input=script,text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'Existing multiplexed connection/remote command failed ({r.returncode}); no fallback.\n{r.stdout}\n{r.stderr}')
    return r.stdout

def deploy():
    files={f'code/runtime/reproduction/stage2ah/{p.name}':base64.b64encode(p.read_bytes()).decode() for p in Path(__file__).parent.glob('*.py')}
    for name in ['plan/protocol.md','protocol.json','protocol-seal.json','carryover.json']:
        files[name]=base64.b64encode((LOCAL/name).read_bytes()).decode()
    script=f'''from pathlib import Path
import base64,shutil,json,hashlib,time
root=Path({ROOT!r});old=Path({OLD!r})
if not root.exists():
 root.mkdir()
 for d in ['control','logs','formal','evaluation','plan']: (root/d).mkdir()
 shutil.copytree(old/'code/runtime',root/'code/runtime',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (root/'creation.json').write_text(json.dumps(dict(epoch=time.time(),run_id=root.name,source_runtime=str(old/'code/runtime'),scope='source-only copy, no model or data duplication'))+'\\n')
files={files!r}
for name,body in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(body)
 if p.exists() and p.read_bytes()!=data:
  backup=root/'code-revisions'/hashlib.sha256(p.read_bytes()).hexdigest()/p.name
  backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists():backup.write_bytes(p.read_bytes())
 p.write_bytes(data)
print(json.dumps({{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}))
'''
    result=remote(script,timeout=60)
    (LOCAL/'deployment-records').mkdir(exist_ok=True)
    path=LOCAL/'deployment-records'/(hashlib.sha256(result.encode()).hexdigest()+'.json')
    if not path.exists():path.write_text(result)
    print(result)

def launch(name,command,timeout):
    script=f'''from pathlib import Path
import json,os,subprocess,time
root=Path({ROOT!r});name={name!r};python={PY!r}
env=os.environ.copy();env.update(PYTHONPATH=str(root/'code/runtime')+':'+str(root/'code/runtime/qwen-vl-finetune'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
record=root/'control'/(name+'-launcher.json')
if record.exists():raise RuntimeError('Job already registered; inspect, never relaunch')
with (root/'logs'/(name+'.log')).open('x') as log:
 cmd=[python,'-m','reproduction.stage2ah.supervise','--root',str(root),'--name',name,'--timeout',str({timeout}),'--',python,*{command!r}]
 proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root/'code/runtime',env=env)
 rec=dict(pid=proc.pid,epoch=time.time(),command=cmd);record.write_text(json.dumps(rec,indent=2)+'\\n');print(json.dumps(rec))
'''
    print(remote(script))

def status(name):
    print(remote(f'''from pathlib import Path
import json
root=Path({ROOT!r});name={name!r}
for p in [root/'control'/(name+'-launcher.json'),root/'control'/name/'process.json',root/'control'/name/'exit.json']:
 print(str(p),p.read_text() if p.exists() else 'NOT_PRESENT')
p=root/'logs'/(name+'.log')
if p.exists():print('LOG_TAIL',p.read_text()[-5500:])
'''))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','launch-diagnostic','status']);p.add_argument('--seed',type=int,choices=[43,44]);p.add_argument('--name');a=p.parse_args()
    if a.action=='deploy':deploy()
    elif a.action=='status':status(a.name)
    else:
        assert a.seed and a.name
        launch(a.name,['-m','reproduction.stage2ah.diagnose_gpu','--root',ROOT,'--old-root',OLD,'--seed',str(a.seed),'--attempt',a.name],3600)
