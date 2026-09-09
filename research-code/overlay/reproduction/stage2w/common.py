"""Fixed artifact identities, directly measured anchors and bounded IO."""
import copy,gc,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import torch
from reproduction.stage2i.fingerprint import sha256_file,canonical_json_sha256
from reproduction.stage2s.common import state_digest,write
from reproduction.stage2t.common import load_raw
from reproduction.stage2u.common import decode_master,encode_master
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES

SOURCE=Path('/home/Medic-AD/outputs/cabg-mil-v1.2/initialization-lce-persistence-v1')
CONTROL=SOURCE.with_name('initialization-lce-persistence-control-v1')
INVENTORY_SHA='25e5f03ab27ac4febc60cd27b16438a209b5567df4244a039423cfc14e6f2e90'
VERIFICATION_SHA='0906f21a1963e238e3efd7c31da56ceefcf4589e3debdc3269197bc2aff9cf69'
SEEDS=(43,44)
PHASES={f's{s}-{a}':range(1,13) if a=='prefix' else range(13,25) for s in SEEDS for a in ('prefix','dynamic','lce_off')}
ANCHORS={}
for s in SEEDS:
    for label,phase,step,parent,refs in (
        ('initial',f's{s}-prefix',1,None,('11',)),
        ('prefix8',f's{s}-prefix',9,f's{s}-prefix/block8-checkpoint',('11',)),
        ('parent12',f's{s}-dynamic',13,f's{s}-prefix/block12-checkpoint',('00','11')),
        ('dynamic20',f's{s}-dynamic',21,f's{s}-dynamic/block20-checkpoint',('11',)),
        ('off20',f's{s}-lce_off',21,f's{s}-lce_off/block20-checkpoint',('00',))):
        ANCHORS[f's{s}-{label}']=dict(seed=s,phase=phase,step=step,parent=parent,references=refs)
GROUPS={'T21':tuple(TRAINABLE_NAMES),'S9':tuple(S9_NAMES),
    'VPT':tuple(n for n in S9_NAMES if 'deep_prompt_embeddings' in n),
    'shared_non_VPT':tuple(n for n in S9_NAMES if 'deep_prompt_embeddings' not in n),
    'T12':tuple(n for n in TRAINABLE_NAMES if n not in S9_NAMES)}

def read(p):return json.loads(Path(p).read_text())
def identity(commit):
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).strip()
    prior=read(SOURCE/'source.json');result={}
    for n,h in prior.items():assert sha256_file(repo/n)==h;result[n]=h
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix in ('.py','.md','.sh'):
            n=str(p.relative_to(repo));subprocess.check_call(['git','ls-files','--error-unmatch',n],cwd=repo,stdout=subprocess.DEVNULL);result[n]=sha256_file(p)
    return result

class Inputs:
    def __init__(self):
        p=CONTROL/'artifact-inventory-v1.json';assert sha256_file(p)==INVENTORY_SHA
        inv=read(p);assert inv['status']=='PASS'
        self.entries={e['path']:e for e in inv['entries'] if e['root']==SOURCE.name};self.checked={}
        assert sha256_file(SOURCE/'verification.json')==VERIFICATION_SHA
        v=self.js('verification.json');assert v['status']=='PASS' and v['replicated']=={'43':True,'44':False}
    def path(self,name):
        assert name in self.entries,name
        p=SOURCE/name;e=self.entries[name]
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha256_file(p)==e['sha256'],name
        self.checked[name]=e['sha256'];return p
    def js(self,name):return read(self.path(name))
    def raw(self,name):return load_raw(self.path(name))
    def record(self,phase,step):
        r=self.js(f'{phase}/block{step}.json');q=dict(r);h=q.pop('record_sha256');assert canonical_json_sha256(q)==h
        d=self.raw(f'{phase}/block{step}.pt.gz');assert r['raw']['sha256']==self.entries[f'{phase}/block{step}.pt.gz']['sha256']
        return r,d

class AccessGuard:
    def __init__(self,output):
        self.output=Path(output).resolve();self.reads=set();self.violations=[]
        def hook(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes)):return
            p=Path(os.fsdecode(args[0])).resolve();s=str(p)
            if s.startswith(('/home/data/','/home/checkpoints/')):
                self.violations.append(s);raise RuntimeError('Image/base/protected/data access forbidden')
            if s.startswith('/home/Medic-AD/outputs/'):
                writing=bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                allowed_write=p.is_relative_to(self.output)
                if writing and not allowed_write:raise RuntimeError('Historical artifact write forbidden')
                if not writing:
                    if not (p.is_relative_to(SOURCE) or p.is_relative_to(CONTROL) or allowed_write):raise RuntimeError('Unregistered historical output read')
                    self.reads.add(s)
        sys.addaudithook(hook)
    def report(self):return dict(status='PASS',data_or_model_opens=0,violations=self.violations,artifact_read_paths=sorted(self.reads))

def disk(root):
    assert shutil.disk_usage(root).free>=16_000_000_000
    assert sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())<10_000_000_000

def cpu_deadline():
    def stop(*_):raise TimeoutError('CPU pass two-hour ceiling')
    signal.signal(signal.SIGALRM,stop);signal.alarm(7200);torch.set_num_threads(4)

def setup(root,commit):
    root=Path(root);root.mkdir(exist_ok=False,parents=True);guard=AccessGuard(root);cpu_deadline()
    ins=Inputs();source=identity(commit);disk(root)
    write(root/'source.json',source);write(root/'protocol-lock.json',dict(code_commit=commit,protocol_sha256=sha256_file(Path(__file__).with_name('PROTOCOL.md')),source_sha256=canonical_json_sha256(source),inventory_sha256=INVENTORY_SHA,source_verification_sha256=VERIFICATION_SHA))
    return ins,guard

def open_existing(root,commit):
    root=Path(root);guard=AccessGuard(root);cpu_deadline();ins=Inputs();source=identity(commit)
    assert source==read(root/'source.json') and read(root/'protocol-lock.json')['code_commit']==commit
    assert read(root/'protocol-lock.json')['protocol_sha256']==sha256_file(Path(__file__).with_name('PROTOCOL.md'))
    return ins,guard

def anchor(ins,name):
    cfg=ANCHORS[name];r,d=ins.record(cfg['phase'],cfg['step'])
    if cfg['parent']:
        p=cfg['parent'];payload=torch.load(ins.path(p+'/checkpoint.pt'),map_location='cpu',weights_only=False)
        manifest=ins.js(p+'/checkpoint.pt.manifest.json');assert manifest['checkpoint_sha256']==ins.entries[p+'/checkpoint.pt']['sha256']
        before=payload['master_state'];opt=payload['optimizer_state']
        assert payload['sampler_state']['cursor']==cfg['step']-1
        assert all(torch.equal(payload['adapter_state'][n],before[n].bfloat16()) for n in TRAINABLE_NAMES)
    else:
        initial=ins.raw(f"{cfg['phase']}/start.pt.gz");before={n:t.float() for n,t in initial['trainable'].items()};opt=ins.js(f"{cfg['phase']}/initial-optimizer.json")
    assert state_digest(before)==r['master_before'] and state_digest(opt)==r['optimizer_before']
    group=opt['param_groups'][0];assert group['params']==list(range(21)) and list(group['betas'])==[.9,.999] and group['eps']==1e-8 and group['weight_decay']==0.
    assert group['lr']==r['update']['learning_rate_used']
    references={}
    for corner in cfg['references']:
        phase=f"s{cfg['seed']}-lce_off" if cfg['step']==13 and corner=='00' else cfg['phase']
        rr,dd=(r,d) if phase==cfg['phase'] else ins.record(phase,cfg['step'])
        for key in ('master_before','optimizer_before','inputs_sha256','rng','controller_before','aggregate_lm','aggregate_lce'):assert rr[key]==r[key]
        after=decode_master(before,dd['master_xor']);assert state_digest(after)==rr['master_after']
        references[corner]=dict(record=rr,after=after)
    return cfg,r,d,before,opt,references

class GpuBudget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'gpu-budget.json'
        if not self.path.exists():write(self.path,dict(started_epoch=time.time(),arithmetic_clones=0,events=[],real_updates=0,model_forwards=0))
        left=1800-(time.time()-read(self.path)['started_epoch']);assert left>0
        def stop(*_):raise TimeoutError('GPU arithmetic thirty-minute ceiling')
        signal.signal(signal.SIGALRM,stop);signal.alarm(int(left))
    def count(self,phase,name,corner):
        d=read(self.path);assert d['arithmetic_clones']<64 and time.time()-d['started_epoch']<1800
        d['arithmetic_clones']+=1;d['events'].append(dict(phase=phase,anchor=name,corner=corner,time=time.time()))
        # Budget is the only mutable live control file; immutable records remain exclusive.
        self.path.write_text(json.dumps(d,indent=2)+'\n');disk(self.root)
