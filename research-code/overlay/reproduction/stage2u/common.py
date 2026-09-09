"""Explicit execution gate, parent/source locks, bounded resources and lossless codec."""
import fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2m.constants import TRAINABLE_NAMES
from reproduction.stage2q.run import save_raw as underlying_save
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2t.common import load_raw,source_identity as t_identity
ARMS=('dynamic','frozen_ratio','lce_off')
PARENT_SHA='770c029177ea377789bdf5682396a022b2c3990c6f18b253b19b1a261fd46bb0'
FULL24=Path('/home/Medic-AD/outputs/cabg-mil-v1.2/full24-reproducibility-v1')
LIMITS={'update':40,'block':40,'eval':128};FLOOR=4_000_000_000;STORAGE=12_000_000_000;SECONDS=7200

def artifact_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def disk_guard(path,needed=500_000_000):
    if needed<0:raise ValueError('Invalid write reservation')
    if shutil.disk_usage(path).free<FLOOR+needed:raise RuntimeError('Causal free-space/write headroom')
    roots=[p for p in (Path(path),*Path(path).parents) if (p/'budget.json').is_file()]
    if roots and artifact_bytes(roots[0])+needed>STORAGE:
        raise RuntimeError('Causal artifact/write headroom')

def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
    return 0

def source_identity(commit):
    result=t_identity(commit);repo=Path(__file__).resolve().parents[2]
    for p in sorted((repo/'reproduction/stage2u').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.sh','.md'):
            name=str(p.relative_to(repo));subprocess.check_call(['git','ls-files','--error-unmatch',name],cwd=repo,stdout=subprocess.DEVNULL)
            result[name]=sha256_file(p)
    return result

def require_execution(a):
    if not a.execute:raise RuntimeError('Preparation only: explicit user GPU approval and --execute are required')
    if a.arm not in ARMS:raise ValueError('Unknown causal arm')

def encode_master(before,after):
    if set(before)!=set(after):raise ValueError('Master schema')
    result={n:torch.bitwise_xor(before[n].contiguous().view(torch.int32),after[n].contiguous().view(torch.int32)) for n in before}
    if state_digest(decode_master(before,result))!=state_digest(after):raise RuntimeError('Lossless master codec')
    return result

def decode_master(before,delta):
    return {n:torch.bitwise_xor(before[n].contiguous().view(torch.int32),delta[n]).view(torch.float32) for n in before}

def save_raw(path,value):
    # Uncompressed tensor bytes plus16MiB exceed the bounded ZIP/gzip metadata overhead.
    disk_guard(Path(path).parent,needed=tensor_bytes(value)+16*1024**2)
    return underlying_save(path,value)

class Budget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'budget.json'
        if not self.path.exists():write(self.path,{'started_epoch':time.time(),'counts':dict.fromkeys(LIMITS,0),'events':[]})
        s=json.loads(self.path.read_text());remaining=SECONDS-(time.time()-s['started_epoch'])
        if remaining<=0:raise RuntimeError('Causal time budget')
        def timeout(*_):raise TimeoutError('Causal two-hour ceiling')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(remaining)))
    def reserve(self,kind,context):
        if kind not in LIMITS:raise ValueError(kind)
        disk_guard(self.root)
        if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())>=STORAGE:raise RuntimeError('Causal 12GB artifact ceiling')
        with self.path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=json.load(f)
            if s['counts'][kind]>=LIMITS[kind] or time.time()-s['started_epoch']>=SECONDS:raise RuntimeError('Causal operation/time ceiling')
            s['counts'][kind]+=1;s['events'].append({'kind':kind,'context':context,'epoch':time.time()})
            f.seek(0);json.dump(s,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())

def read_plan(path,commit):
    plan=json.loads(Path(path).read_text());source=source_identity(commit)
    assert plan['status']=='CPU_READY_GPU_NOT_AUTHORIZED' and plan['code_commit']==commit
    assert plan['source_sha256']==canonical_json_sha256(source)
    assert sha256_file(Path(plan['parent_checkpoint']))==plan['parent_checkpoint_sha256']==PARENT_SHA
    assert sha256_file(Path(plan['full24_verification']))==plan['full24_verification_sha256']
    assert plan['protocol_sha256']==sha256_file(Path(__file__).with_name('PROTOCOL.md'))
    assert Path(plan['parent_checkpoint']).resolve()==(FULL24/'R0/block12-checkpoint/checkpoint.pt').resolve()
    from reproduction.stage2l.checkpoint import load_strict
    payload,_=load_strict(Path(plan['parent_checkpoint']),plan['parent_provenance'])
    record=json.loads((FULL24/'R0/block12.json').read_text());copy=dict(record);claimed=copy.pop('record_sha256')
    assert canonical_json_sha256(copy)==claimed==payload['trace_state']['last_record_sha256']
    assert plan['frozen_ratio']==record['controller']['lambda_raw']
    assert plan['runtime_sha256']==sha256_file(Path(plan['parent_checkpoint']).with_name('runtime-state.pt.gz'))
    return plan,source

def provenance(plan,arm):
    return {'head':plan['code_commit'],'source_sha256':plan['source_sha256'],'arm':arm,
            'parent_checkpoint_sha256':PARENT_SHA,'protocol_sha256':plan['protocol_sha256'],
            'train_manifest_sha256':plan['train_manifest_sha256'],'eval_manifest_sha256':plan['eval_manifest_sha256'],
            'schedule_horizon':24,'parent_cursor':12,'applied_policy':arm}
