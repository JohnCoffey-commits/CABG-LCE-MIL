"""Fixed phases, identities and cumulative resource limits; no model math."""
import fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2m.train_scout import matched_configuration
from reproduction.stage2p import CONFIG_SHA256
from reproduction.stage2q.run import save_raw as underlying_save
from reproduction.stage2s.common import write
from reproduction.stage2u.common import source_identity as u_identity,encode_master,decode_master,tensor_bytes,load_raw,FULL24

SEEDS=(43,44)
LIMITS={'update':80,'block':84,'eval':256}
SECONDS=10800;STORAGE=30_000_000_000;FLOOR=16_000_000_000
HISTORICAL=FULL24.with_name('controller-lce-persistence-v1')
HISTORICAL_CONTROL=FULL24.with_name('controller-lce-persistence-execution-control-v1')
PHASES={'bridge-b13':dict(seed=42,start=12,end=13,arm='dynamic',mode='replay'),
        'bridge-b24':dict(seed=42,start=23,end=24,arm='dynamic',mode='replay')}
for seed in SEEDS:
    PHASES[f's{seed}-prefix']=dict(seed=seed,start=0,end=12,arm='dynamic',mode='train')
    for arm in ('dynamic','lce_off'):
        PHASES[f's{seed}-{arm}']=dict(seed=seed,start=12,end=24,arm=arm,mode='train')
EVALUATIONS={'bridge-eval-b24':dict(seed=42,phase='R0',cursor=24)}
for seed in SEEDS:
    EVALUATIONS[f'eval-s{seed}-parent12']=dict(seed=seed,phase=f's{seed}-prefix',cursor=12)
    for arm in ('dynamic','lce_off'):
        EVALUATIONS[f'eval-s{seed}-{arm}24']=dict(seed=seed,phase=f's{seed}-{arm}',cursor=24)

def read(path):return json.loads(Path(path).read_text())

def configuration(seed):
    original=matched_configuration();assert canonical_json_sha256(original)==CONFIG_SHA256
    if seed not in (42,*SEEDS):raise ValueError('Unregistered seed')
    return {**original,'seed':seed}

def source_identity(commit):
    result=u_identity(commit);repo=Path(__file__).resolve().parents[2]
    for name in ('models/Qwen2_5_VL/Qwen2_5_VL_hf.py','qwen-vl-finetune/qwenvl/train/train_qwen.py',
                 'qwen-vl-finetune/qwenvl/data/data_qwen.py','qwen-vl-finetune/qwenvl/data/__init__.py'):
        subprocess.check_call(['git','ls-files','--error-unmatch',name],cwd=repo,stdout=subprocess.DEVNULL)
        result[name]=sha256_file(repo/name)
    for p in sorted(Path(__file__).parent.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.sh','.md'):
            name=str(p.relative_to(repo))
            subprocess.check_call(['git','ls-files','--error-unmatch',name],cwd=repo,stdout=subprocess.DEVNULL)
            result[name]=sha256_file(p)
    return result

def environment():
    return dict(torch=torch.__version__,cuda=torch.version.cuda,
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())

def read_campaign(root,commit):
    plan=read(Path(root)/'preflight.json');source=source_identity(commit)
    assert plan['status']=='PASS' and plan['code_commit']==commit
    assert plan['source_sha256']==canonical_json_sha256(source)
    assert plan['protocol_sha256']==sha256_file(Path(__file__).with_name('PROTOCOL.md'))
    assert plan['environment']==environment()
    return plan,source

def require_execution(a,evaluation=False):
    if not a.execute:raise RuntimeError('Explicit authorization and --execute required before model work')
    if a.phase not in (EVALUATIONS if evaluation else PHASES):raise ValueError('Unregistered phase')

def require_phase_ready(root,phase,evaluation=False):
    root=Path(root)
    if phase.startswith('bridge'):
        if evaluation:assert read(root/'bridge-train-verification.json')['status']=='PASS'
        return
    assert read(root/'bridge-verification.json')['status']=='PASS'
    if evaluation:
        for name,cfg in PHASES.items():
            if cfg['mode']=='train':assert read(root/name/'result.json')['status']=='SUCCESS'
    elif PHASES[phase]['start']:
        seed=PHASES[phase]['seed']
        assert read(root/f's{seed}-prefix/result.json')['status']=='SUCCESS'
        if PHASES[phase]['arm']=='lce_off':assert read(root/f's{seed}-dynamic/result.json')['status']=='SUCCESS'

def provenance(plan,phase):
    cfg=PHASES[phase]
    return dict(head=plan['code_commit'],source_sha256=plan['source_sha256'],phase=phase,
        seed=cfg['seed'],applied_policy=cfg['arm'],schedule_horizon=24,
        configuration_sha256=canonical_json_sha256(configuration(cfg['seed'])),
        protocol_sha256=plan['protocol_sha256'],train_manifest_sha256=plan['train_manifest_sha256'],
        eval_manifest_sha256=plan['eval_manifest_sha256'])

def parent(root,phase,plan):
    cfg=PHASES[phase]
    if not cfg['start']:return None,None
    if cfg['mode']=='replay':
        return FULL24/'R0'/f"block{cfg['start']}-checkpoint/checkpoint.pt",read(FULL24/'R0/provenance.json')
    name=f"s{cfg['seed']}-prefix"
    return Path(root)/name/'block12-checkpoint/checkpoint.pt',provenance(plan,name)

def eval_checkpoint(root,phase,plan):
    cfg=EVALUATIONS[phase]
    if phase.startswith('bridge'):
        return FULL24/'R0/block24-checkpoint/checkpoint.pt',read(FULL24/'R0/provenance.json')
    return Path(root)/cfg['phase']/f"block{cfg['cursor']}-checkpoint/checkpoint.pt",provenance(plan,cfg['phase'])

def artifact_bytes(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def disk_guard(path,needed=500_000_000):
    if needed<0:raise ValueError('Invalid write reservation')
    if shutil.disk_usage(path).free<FLOOR+needed:raise RuntimeError('Stage2v free-space/write ceiling')
    roots=[p for p in (Path(path),*Path(path).parents) if (p/'budget.json').is_file()]
    if roots and artifact_bytes(roots[0])+needed>STORAGE:raise RuntimeError('Stage2v artifact/write ceiling')

def save_raw(path,value):
    disk_guard(Path(path).parent,tensor_bytes(value)+16*1024**2)
    return underlying_save(path,value)

class Budget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'budget.json'
        if not self.path.exists():write(self.path,dict(started_epoch=time.time(),counts=dict.fromkeys(LIMITS,0),events=[]))
        remaining=SECONDS-(time.time()-read(self.path)['started_epoch'])
        if remaining<=0:raise RuntimeError('Stage2v time ceiling')
        def timeout(*_):raise TimeoutError('Stage2v three-hour ceiling')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(remaining)))
    def reserve(self,kind,context):
        if kind not in LIMITS:raise ValueError(kind)
        disk_guard(self.root)
        with self.path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=json.load(f)
            if s['counts'][kind]>=LIMITS[kind] or time.time()-s['started_epoch']>=SECONDS:raise RuntimeError('Stage2v operation/time ceiling')
            s['counts'][kind]+=1;s['events'].append(dict(kind=kind,context=context,epoch=time.time()))
            f.seek(0);json.dump(s,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())
