"""Locked two-seed phases, original evidence and cumulative budgets."""
import fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2p import TRAIN_SHA256,EVAL_SHA256
from reproduction.stage2q.run import save_raw as underlying_save,data_contract
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2u.common import encode_master,decode_master,tensor_bytes,load_raw
from reproduction.stage2v.common import configuration,environment
from reproduction.stage2z.common import Inputs as TrainingInputs,SOURCE
from reproduction.stage2x.common import WriteGuard
LATEST=SOURCE.with_name('lm-only-baseline-v1')
LATEST_CONTROL=SOURCE.with_name('lm-only-baseline-control-v1')
ZVERIFY='3e936f276fee2d68a42cee9e3892f2c458cc4a59b584e2f7529e4112ee387519'
ZINVENTORY='0b79305794b3cb57c945671387ef13eb6b0e73cbc490bb1e9854ca535df27012'
ZPRODUCER='02c52ecd6bbb644c559ecf91b9f060c6636d89ef'
ZCHECKER='25afe222dcb7849362e71ae23c33ec4e194762cd'
SPLIT=Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json')
BASE=Path('/home/checkpoints/Lingshu-7B')
SEEDS=(43,44);BOUNDARIES=(20,24)
LIMITS=dict(update=36,block=40,eval=192,model=12)
PLANNED=dict(update=24,block=26,eval=128,model=8)
SECONDS=5400;STORAGE=18_000_000_000;FLOOR=16_000_000_000
PHASES={f's{s}-late_only':dict(seed=s,start=12,end=24,arm='dynamic',mode='train') for s in SEEDS}
PHASES.update({f'replay-s{s}-b21':dict(seed=s,start=20,end=21,arm='dynamic',mode='replay') for s in SEEDS})
EVALUATIONS={f'eval-ref-s{s}-lm_only24':dict(seed=s,phase=f's{s}-lm_only',cursor=24,reference=True) for s in SEEDS}
EVALUATIONS.update({f'eval-s{s}-late_only24':dict(seed=s,phase=f's{s}-late_only',cursor=24,reference=False) for s in SEEDS})
HISTORIES=('none','early','late','full')
CONTRASTS={
    'early_without_late':{'early':1,'none':-1},
    'early_with_late':{'full':1,'late':-1},
    'late_without_early':{'late':1,'none':-1},
    'late_with_early':{'full':1,'early':-1},
    'late_vs_early':{'late':1,'early':-1},
    'interaction':{'full':1,'early':-1,'late':-1,'none':1}}

def read(path):return json.loads(Path(path).read_text())
class Inputs(TrainingInputs):
    def __init__(self):
        super().__init__();p=LATEST_CONTROL/'artifact-inventory-v1.json';assert sha256_file(p)==ZINVENTORY
        inv=read(p);assert inv['status']=='PASS'
        self.zentries={e['path']:e for e in inv['entries'] if e['root']==LATEST.name}
        self.zcontrol={e['path']:e for e in inv['entries'] if e['root']==LATEST_CONTROL.name};self.zchecked={}
        assert sha256_file(LATEST/'verification.json')==ZVERIFY
        v=self.zjs('verification.json');assert v['status']=='PASS'
        assert v['decision']==dict(dynamic='DESCRIPTIVE_LCE_GAIN_2_OF_2',early_only='DESCRIPTIVE_LCE_GAIN_2_OF_2')
        assert v['code_commit']==ZPRODUCER and v['checker_identity']['commit']==ZCHECKER
    def zpath(self,name,control=False):
        e=(self.zcontrol if control else self.zentries)[name];p=(LATEST_CONTROL if control else LATEST)/name
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha256_file(p)==e['sha256']
        self.zchecked[('control/' if control else '')+name]=e['sha256'];return p
    def zjs(self,name,control=False):return read(self.zpath(name,control))
    def zraw(self,name):return load_raw(self.zpath(name))
    def zrecord(self,phase,step):
        r=self.zjs(f'{phase}/block{step}.json');copy=dict(r);h=copy.pop('record_sha256');assert canonical_json_sha256(copy)==h
        d=self.zraw(f'{phase}/block{step}.pt.gz');assert r['raw']['sha256']==self.zentries[f'{phase}/block{step}.pt.gz']['sha256']
        return r,d

def source_identity(commit):
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).strip()
    prior=Inputs().zjs('checker-source-v2.json',True);assert len(prior)==135;result={}
    for n,h in prior.items():assert sha256_file(repo/n)==h;result[n]=h
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix in ('.py','.sh','.md'):
            n=str(p.relative_to(repo));subprocess.check_call(['git','ls-files','--error-unmatch',n],cwd=repo,stdout=subprocess.DEVNULL);result[n]=sha256_file(p)
    return result

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
    if evaluation:
        assert read(root/'training-verification.json')['status']=='PASS'
        if not EVALUATIONS[phase]['reference']:assert read(root/'reference-verification.json')['status']=='PASS'
    elif PHASES[phase]['mode']=='replay':
        for seed in SEEDS:assert read(root/f's{seed}-late_only/result.json')['status']=='SUCCESS'

def provenance(plan,phase):
    cfg=PHASES[phase]
    return dict(head=plan['code_commit'],source_sha256=plan['source_sha256'],phase=phase,
        seed=cfg['seed'],applied_policy=cfg['arm'],schedule_horizon=24,
        configuration_sha256=canonical_json_sha256(configuration(cfg['seed'])),
        protocol_sha256=plan['protocol_sha256'],train_manifest_sha256=plan['train_manifest_sha256'],eval_manifest_sha256=plan['eval_manifest_sha256'])

def parent(root,phase,plan):
    cfg=PHASES[phase]
    if cfg['mode']=='train':
        assert cfg['start']==12;ins=Inputs();name=f"s{cfg['seed']}-lm_only"
        return ins.zpath(name+'/block12-checkpoint/checkpoint.pt'),ins.zjs(name+'/provenance.json')
    assert cfg['mode']=='replay' and cfg['start']==20
    name=f"s{cfg['seed']}-late_only"
    return Path(root)/name/'block20-checkpoint/checkpoint.pt',provenance(plan,name)

def eval_checkpoint(root,phase,plan):
    cfg=EVALUATIONS[phase]
    if cfg['reference']:
        ins=Inputs();name=cfg['phase']
        return ins.zpath(name+'/block24-checkpoint/checkpoint.pt'),ins.zjs(name+'/provenance.json')
    return Path(root)/cfg['phase']/'block24-checkpoint/checkpoint.pt',provenance(plan,cfg['phase'])

def original_phase(seed,step):
    assert seed in SEEDS and 1<=step<=24
    return f's{seed}-prefix' if step<=12 else f's{seed}-dynamic'

def data_metadata(root):
    split,train=data_contract(SPLIT,0)
    assert split==read(Path(root)/'split.json') and train==read(Path(root)/'train-manifest.json')
    assert sha256_file(Path(split['eval_manifest']))==EVAL_SHA256 and sha256_file(Path(split['eval_annotation']))==split['eval_annotation_sha256']
    ev=[json.loads(s) for s in Path(split['eval_manifest']).read_text().splitlines()]
    assert ev==read(Path(root)/'eval-manifest.json')
    return split,train,ev

def artifact_bytes(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def disk_guard(path,needed=500_000_000):
    if needed<0:raise ValueError('Invalid write reservation')
    if shutil.disk_usage(path).free<FLOOR+needed:raise RuntimeError('Stage2aa free-space/write ceiling')
    roots=[p for p in (Path(path),*Path(path).parents) if (p/'budget.json').is_file()]
    if roots and artifact_bytes(roots[0])+needed>STORAGE:raise RuntimeError('Stage2aa artifact/write ceiling')

def save_raw(path,value):
    disk_guard(Path(path).parent,tensor_bytes(value)+16*1024**2)
    return underlying_save(path,value)

class Budget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'budget.json'
        if not self.path.exists():write(self.path,dict(started_epoch=time.time(),counts=dict.fromkeys(LIMITS,0),events=[]))
        remaining=SECONDS-(time.time()-read(self.path)['started_epoch'])
        if remaining<=0:raise RuntimeError('Stage2aa time ceiling')
        def timeout(*_):raise TimeoutError('Stage2aa ninety-minute ceiling')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(remaining)))
    def reserve(self,kind,context):
        if kind not in LIMITS:raise ValueError(kind)
        disk_guard(self.root)
        with self.path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=json.load(f)
            if s['counts'][kind]>=LIMITS[kind] or time.time()-s['started_epoch']>=SECONDS:raise RuntimeError('Stage2aa operation/time ceiling')
            s['counts'][kind]+=1;s['events'].append(dict(kind=kind,context=context,epoch=time.time()))
            f.seek(0);json.dump(s,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())
