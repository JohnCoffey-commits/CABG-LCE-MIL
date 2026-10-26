"""Enforce the producer source/input/dependency identities bound before training."""
import hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def require_frozen(root):
    root=Path(root);p=root/'protocol.json';plan=json.loads(p.read_text())
    assert plan['status']=='FROZEN_FOR_FORMAL_EXECUTION'
    seal=json.loads((root/'protocol-seal.json').read_text());assert seal['sha256']==sha(p)
    for path,h in plan['sources'].items():assert sha(path)==h,'Producer source changed: '+path
    for path,h in plan['input_files'].items():assert sha(path)==h,'Input/manifest changed: '+path
    assert plan['horizon']==1000 and plan['seeds']==[43,44]
    return plan,seal['sha256']
