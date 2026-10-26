"""Mirror only small text evidence; never transfer weights, caches, images or checkpoints."""
import base64,gzip,hashlib,json
from pathlib import Path
from reproduction.stage2ah.ops import remote,ROOT,LOCAL

def main():
    script=f'''from pathlib import Path
import base64,gzip,hashlib,json
root=Path({ROOT!r});items=[]
for p in root.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(root)
 if rel.parts[0] not in ['control','logs','formal','evaluation','results-v1','plan']:continue
 if p.suffix not in ['.json','.jsonl','.csv','.md','.log','.py']:continue
 if p.stat().st_size>12000000:raise RuntimeError('Unexpected large text evidence: '+str(rel))
 b=p.read_bytes();items.append(dict(path=str(rel),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),body=base64.b64encode(b).decode()))
for name in ['creation.json','protocol.json','protocol-seal.json','carryover.json','events.jsonl','short-trial-forecast.json','campaign-launcher.json','campaign-source-seal.json','campaign-complete.json','campaign-failure.json','final-evaluation-release.json','accepted-endpoints.json','training-independent-verification-v1.json','verification-amendment-01.json','evaluation-instrumentation-amendment-02.json','copied-runtime-source-verification.json','posthoc-trace-description-plan.json']:
 p=root/name
 if p.exists():
  b=p.read_bytes();items.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),body=base64.b64encode(b).decode()))
assert sum(x['bytes'] for x in items)<150000000
payload=json.dumps(items).encode()
print(base64.b64encode(gzip.compress(payload,compresslevel=3)).decode())
'''
    wire=remote(script,timeout=60)
    items=json.loads(gzip.decompress(base64.b64decode(wire)));target=LOCAL/'verified-execution-v1';target.mkdir(exist_ok=True)
    manifest=[]
    for item in items:
        rel=Path(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
        data=base64.b64decode(item.pop('body'));assert hashlib.sha256(data).hexdigest()==item['sha256']
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=data:
            # Mutable logs/control are snapshotted; immutable results must not change.
            assert rel.parts[0] in ['control','logs'] or rel.name=='events.jsonl',str(rel)
            old=p.read_bytes();archive=target/'previous-snapshots'/hashlib.sha256(old).hexdigest()/p.name
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():archive.write_bytes(old)
        p.write_bytes(data);manifest.append(item)
    serialized=json.dumps(manifest,indent=2)+'\n'
    m=LOCAL/'mirror-manifests';m.mkdir(exist_ok=True);(m/(hashlib.sha256(serialized.encode()).hexdigest()+'.json')).write_text(serialized)
    print(json.dumps(dict(files=len(items),bytes=sum(x['bytes'] for x in items),destination=str(target))))

if __name__=='__main__':main()
