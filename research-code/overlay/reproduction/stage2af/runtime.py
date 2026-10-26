"""Verified model/gradient primitives with fixed, cached official-wrapper inputs."""
from __future__ import annotations
import gc
import gzip
import hashlib
import json
import random
import os
from pathlib import Path
import torch
from reproduction.stage2h.runtime import load_stage2h_runtime
from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.rng_state import snapshot_rng_state,restore_rng_state,rng_state_fingerprint
from reproduction.stage2i.fingerprint import canonical_json_sha256
from reproduction.stage2l.controller import DynamicController
from reproduction.stage2l.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2q.execution import AttentionExecution,cpu_tree
from reproduction.stage2q.run import canonicalize_dormant,full_state
from reproduction.stage2s.common import setup_flags,runtime_state,restore_runtime
from reproduction.stage2s.resize import ResizeProbe
from reproduction.stage2s.run import collect as verified_collect
from reproduction.stage2af.inputs import read_rows,training_batch
from reproduction.stage2af.core import make_optimizer

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()

def sequence(rows,seed,steps=1000):
    rng=random.Random(seed);by_class={y:sorted([r for r in rows if r['target']==y],key=lambda r:r['sample_id']) for y in (0,1)}
    queues={0:[],1:[]};cursor={0:0,1:0};out=[]
    for step in range(steps):
        for position,y in enumerate((0,1,1)):
            if cursor[y]>=len(queues[y]):
                queues[y]=list(by_class[y]);rng.shuffle(queues[y]);cursor[y]=0
            row=dict(queues[y][cursor[y]]);cursor[y]+=1
            row.update(scout_label='abnormal' if y else 'normal',scout_exposure_id=f's{seed}-step{step+1:04}-position{position}',exposure_step=step+1)
            out.append(row)
    return out

class PreparedDataset:
    def __init__(self,wrapper,rows,contract,reset_state):self.wrapper,self.rows,self.contract,self.reset_state=wrapper,rows,contract,reset_state
    def __getitem__(self,index):
        restore_runtime(self.wrapper.llm,self.reset_state)
        return training_batch(self.rows[index],self.wrapper.device,self.contract)

def identity_collator(rows):
    assert len(rows)==1
    return rows[0]

class Runtime:
    def __init__(self,root,seed,training=True):
        self.root=Path(root);self.seed=seed;setup_flags()
        self.attention=AttentionExecution('canonical_flash_deterministic').install()
        self.probe=ResizeProbe(deterministic=True).install()
        self.wrapper,self.loaded=load_stage2h_runtime(Path('/home/checkpoints/Lingshu-7B'),seed=seed,training=training,gradient_checkpointing=training)
        self.model=self.wrapper.llm;self.dormant=canonicalize_dormant(self.model)
        self.initial_identity=trainable_state_record(self.model)
        self.initial_full=full_state(self.model)
        self.clean_runtime=runtime_state(self.model)
        self.initial_rng=snapshot_rng_state()
        named=dict(self.model.named_parameters())
        self.initial=cpu_tree({n:named[n] for n in TRAINABLE_NAMES})
        self.masters,self.optimizer,self.scheduler=make_optimizer(self.initial,self.wrapper.device) if training else (None,None,None)
        self.controller=DynamicController()
        self.contract=json.loads((self.root/'data/prepared-v1/input-contract.json').read_text())
        self.rows=read_rows(self.root/'data/prepared-v1/manifest.jsonl')
        self.train_sequence=sequence([r for r in self.rows if r['split']=='train'],seed)
        self.dataset=PreparedDataset(self.wrapper,self.train_sequence,self.contract,self.clean_runtime)

    def collect(self,step):
        return verified_collect(self.wrapper,self.dataset,identity_collator,self.train_sequence,step-1,self.attention,self.probe)

    def restore_initial(self):
        named=dict(self.model.named_parameters())
        with torch.no_grad():
            for n,v in self.initial.items():named[n].copy_(v.to(named[n]))
        restore_runtime(self.model,self.clean_runtime);restore_rng_state(self.initial_rng)

    def save(self,path,step,arm,coefficient,protocol_sha256):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={'schema':'stage2af-checkpoint-v1','seed':self.seed,'step':step,'arm':arm,'coefficient':coefficient,
                 'protocol_sha256':protocol_sha256,'initial_identity':self.initial_identity,
                 'adapter':cpu_tree({n:dict(self.model.named_parameters())[n] for n in TRAINABLE_NAMES}),
                 'masters':cpu_tree(self.masters),'optimizer':cpu_tree(self.optimizer.state_dict()),
                 'scheduler':self.scheduler.state_dict(),'controller':self.controller.state_dict(),
                 'rng':snapshot_rng_state(),'runtime':cpu_tree(self.clean_runtime),
                 'sequence_sha256':canonical_json_sha256(self.train_sequence)}
        with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        record={'path':str(path),'bytes':path.stat().st_size,'sha256':sha_file(path),'step':step,'arm':arm,'seed':self.seed,'protocol_sha256':protocol_sha256}
        write(path.with_suffix('.json'),record);return record

    def load(self,path,protocol_sha256,arm,coefficient):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
        assert path.stat().st_size==meta['bytes'] and sha_file(path)==meta['sha256']
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['schema']=='stage2af-checkpoint-v1' and p['protocol_sha256']==protocol_sha256
        assert (p['seed'],p['arm'],p['coefficient'])==(self.seed,arm,coefficient)
        assert p['initial_identity']==self.initial_identity
        assert p['sequence_sha256']==canonical_json_sha256(self.train_sequence)
        named=dict(self.model.named_parameters())
        with torch.no_grad():
            for n in TRAINABLE_NAMES:
                named[n].copy_(p['adapter'][n].to(named[n]));self.masters[n].copy_(p['masters'][n].to(self.masters[n]))
        self.optimizer.load_state_dict(p['optimizer']);self.scheduler.load_state_dict(p['scheduler']);self.controller.load_state_dict(p['controller'])
        for state in self.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor) and k!='step':state[k]=v.to(self.wrapper.device)
        self.clean_runtime=p['runtime'];self.dataset.reset_state=self.clean_runtime
        restore_runtime(self.model,self.clean_runtime);restore_rng_state(p['rng'])
        return p['step']

    def verify_frozen(self):
        restore_runtime(self.model,self.clean_runtime)
        now=full_state(self.model)
        initial={r['name']:r for r in self.initial_full['parameters']}
        current={r['name']:r for r in now['parameters']}
        assert set(initial)==set(current)
        for n,value in initial.items():
            if n not in TRAINABLE_NAMES:assert current[n]==value,('Frozen parameter changed',n)
        assert now['buffers']==self.initial_full['buffers'],'Registered buffers changed'
        return {'frozen_parameters_unchanged':True,'initial_trainable_identity':self.initial_identity}

    def close(self):self.probe.close();self.attention.close()

def save_gradients(path,lm,lce,images):
    raw={'lm':[{n:v.to(torch.bfloat16) for n,v in row.items()} for row in lm],
         'lce':[{n:v.to(torch.bfloat16) for n,v in row.items()} for row in lce],'images':images}
    for key,original in [('lm',lm),('lce',lce)]:
        for i,row in enumerate(original):
            for n,value in row.items():assert torch.equal(value,raw[key][i][n].float())
    with Path(path).open('xb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=1,mtime=0) as f:torch.save(raw,f)
    return {'path':str(path),'bytes':Path(path).stat().st_size,'sha256':sha_file(path)}
