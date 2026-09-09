"""Immutable evidence/state bindings and cumulative forward-only budgets."""
import fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import torch
from reproduction.stage2i.fingerprint import sha256_file,canonical_json_sha256
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2t.common import load_raw
from reproduction.stage2q.run import save_raw
from reproduction.stage2u.common import decode_master
from reproduction.stage2w.common import ANCHORS,TRAINABLE_NAMES,SOURCE,Inputs as OriginalInputs

WEIGHTS=SOURCE.with_name('optimizer-mechanism-v1')
WCONTROL=SOURCE.with_name('optimizer-mechanism-control-v1')
WVERIFY='5a52eaf64f4cd49b884f8014361c0d676d9f92510f45f4f9bb84bdadce175285'
WINVENTORY='37b6967f6635048a380e0bd26a479440780773690b7d475de99c1c03f4cd9323'
SPLIT=Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json')
BASE=Path('/home/checkpoints/Lingshu-7B')
LIMITS=dict(process=14,forward=156);SECONDS=3600

def read(p):return json.loads(Path(p).read_text())
class Inputs(OriginalInputs):
    def __init__(self):
        super().__init__();p=WCONTROL/'artifact-inventory-v1.json';assert sha256_file(p)==WINVENTORY
        self.wentries={e['path']:e for e in read(p)['entries'] if e['root']==WEIGHTS.name};self.wchecked={}
        assert sha256_file(WEIGHTS/'verification-v2/verification.json')==WVERIFY
        assert self.wjs('verification-v2/verification.json')['status']=='PASS'
    def wpath(self,name):
        e=self.wentries[name];p=WEIGHTS/name
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha256_file(p)==e['sha256']
        self.wchecked[name]=e['sha256'];return p
    def wjs(self,name):return read(self.wpath(name))
    def wraw(self,name):return load_raw(self.wpath(name))

def source_identity(commit,ins):
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).strip()
    prior=ins.wjs('verification-v2/checker-source.json');out={}
    assert len(prior)==104
    for n,h in prior.items():assert sha256_file(repo/n)==h;out[n]=h
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix in ('.py','.md','.sh'):
            n=str(p.relative_to(repo));subprocess.check_call(['git','ls-files','--error-unmatch',n],cwd=repo,stdout=subprocess.DEVNULL);out[n]=sha256_file(p)
    return out

def variants(report):
    representatives={};aliases={}
    for key in ('00','01','10','11'):
        h=report['corners'][key]['bf16_sha256']
        if h not in representatives:representatives[h]=key
        aliases[key]=representatives[h]
    return list(representatives.values()),aliases

def state(ins,name):
    cfg=ANCHORS[name];r,d=ins.record(cfg['phase'],cfg['step'])
    if cfg['parent']:
        ck=cfg['parent'];payload=torch.load(ins.path(ck+'/checkpoint.pt'),map_location='cpu',weights_only=False)
        assert ins.js(ck+'/checkpoint.pt.manifest.json')['checkpoint_sha256']==ins.entries[ck+'/checkpoint.pt']['sha256']
        assert payload['sampler_state']['cursor']==cfg['step']-1
        before=payload['master_state'];runtime=ins.raw(ck+'/runtime-state.pt.gz');rng=payload['rng_state']
        assert state_digest(payload['optimizer_state'])==r['optimizer_before']
        assert all(torch.equal(payload['adapter_state'][n],before[n].bfloat16()) for n in TRAINABLE_NAMES)
        expected_full=ins.js(ck+'/full-state.json')
        del payload
    else:
        p=ins.raw(cfg['phase']+'/start.pt.gz');before={n:v.float() for n,v in p['trainable'].items()};runtime=p['runtime'];rng=p['rng']
        expected_full=ins.js(cfg['phase']+'/full-initial-state.json')
    assert state_digest(before)==r['master_before'] and state_digest(runtime)==r['runtime_before']
    assert rng_state_fingerprint(rng)==r['rng']['before_block']
    return cfg,r,d['attention_maps'],before,runtime,rng,expected_full

def candidate(ins,name,key,before):
    report=ins.wjs('corners/'+name+'.json');data=ins.wraw(f'corners/{name}-{key}.pt.gz')
    after=decode_master(before,data['master_xor']);cast={n:v.bfloat16() for n,v in after.items()}
    assert state_digest(after)==report['corners'][key]['master_sha256']
    assert state_digest(cast)==report['corners'][key]['bf16_sha256']
    return cast

def disk(root):
    assert shutil.disk_usage(root).free>=16_000_000_000
    assert sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())<4_000_000_000

class Budget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'budget.json'
        if not self.path.exists():write(self.path,dict(started_epoch=time.time(),counts=dict.fromkeys(LIMITS,0),events=[],real_updates=0,backward=0))
        left=SECONDS-(time.time()-read(self.path)['started_epoch']);assert left>0
        def stop(*_):raise TimeoutError('Forward study one-hour budget')
        signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(left)))
    def reserve(self,kind,context):
        assert kind in LIMITS
        with self.path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=json.load(f)
            assert s['counts'][kind]<LIMITS[kind] and time.time()-s['started_epoch']<SECONDS
            s['counts'][kind]+=1;s['events'].append(dict(kind=kind,context=context,epoch=time.time()))
            f.seek(0);json.dump(s,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())
        disk(self.root)

class WriteGuard:
    def __init__(self,root):
        root=Path(root).resolve();self.violations=[]
        def hook(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes)):return
            p=Path(os.fsdecode(args[0])).resolve();s=str(p)
            if s.startswith('/home/Medic-AD/outputs/') and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) and not p.is_relative_to(root):
                self.violations.append(s);raise RuntimeError('Historical artifact write forbidden')
        sys.addaudithook(hook)

def forbidden(*_,**__):raise RuntimeError('Training update/backward forbidden in forward-response study')
