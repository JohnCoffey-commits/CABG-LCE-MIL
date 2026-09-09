"""Immutable stage2x bridge bindings and bounded full cross-input design."""
from reproduction.stage2x.common import *
from reproduction.stage2x.common import Inputs as ForwardInputs
FORWARD=SOURCE.with_name('forward-mechanism-v1')
XCONTROL=SOURCE.with_name('forward-mechanism-control-v1')
XVERIFY='456a892b60c09958832c28a78f504c77866178a80264376dc299671be168ecaa'
XINVENTORY='95b13bf740abbe28b90ab4516643c6c671bd767192e6b37c7abdf11d9bfdf2ef'
BLOCKS=(1,9,13,21)
LIMITS=dict(process=14,forward=528);SECONDS=5400
class Inputs(ForwardInputs):
    def __init__(self):
        super().__init__();p=XCONTROL/'artifact-inventory-v1.json';assert sha256_file(p)==XINVENTORY
        self.xentries={e['path']:e for e in read(p)['entries'] if e['root']==FORWARD.name};self.xchecked={}
        assert sha256_file(FORWARD/'verification.json')==XVERIFY and self.xjs('verification.json')['status']=='PASS'
    def xpath(self,name):
        e=self.xentries[name];p=FORWARD/name
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha256_file(p)==e['sha256']
        self.xchecked[name]=e['sha256'];return p
    def xjs(self,name):return read(self.xpath(name))
    def xraw(self,name):return load_raw(self.xpath(name))

def source_identity(commit,ins):
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).strip()
    prior=ins.xjs('source.json');out={};assert len(prior)==114
    for n,h in prior.items():assert sha256_file(repo/n)==h;out[n]=h
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix in ('.py','.md','.sh'):
            n=str(p.relative_to(repo));subprocess.check_call(['git','ls-files','--error-unmatch',n],cwd=repo,stdout=subprocess.DEVNULL);out[n]=sha256_file(p)
    return out

def queries(step):
    assert step in BLOCKS
    return [step,*[b for b in BLOCKS if b!=step]]

def indices_for(block):
    assert block in BLOCKS
    return list(range((block-1)*3,block*3))

def disk(root):
    assert shutil.disk_usage(root).free>=16_000_000_000
    assert sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())<4_000_000_000

class Budget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'budget.json'
        if not self.path.exists():write(self.path,dict(started_epoch=time.time(),counts=dict.fromkeys(LIMITS,0),events=[],real_updates=0,backward=0))
        left=SECONDS-(time.time()-read(self.path)['started_epoch']);assert left>0
        def stop(*_):raise TimeoutError('Forward study ninety-minute budget')
        signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(left)))
    def reserve(self,kind,context):
        assert kind in LIMITS
        with self.path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=json.load(f)
            assert s['counts'][kind]<LIMITS[kind] and time.time()-s['started_epoch']<SECONDS
            s['counts'][kind]+=1;s['events'].append(dict(kind=kind,context=context,epoch=time.time()))
            f.seek(0);json.dump(s,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())
        disk(self.root)
