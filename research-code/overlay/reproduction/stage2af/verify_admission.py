"""Separate graph reconstruction and fixed-selection verification; no model calls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SALT='cabg-stage2af-chestx-20260918-v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;d=root/'data/admission-v1'
    result=json.loads((d/'admission.json').read_text());assert result['status']=='PASS_IMAGE_LEVEL_ADMISSION'
    metadata=json.loads((d/'all-image-metadata.json').read_text());pairs=json.loads((d/'duplicate-pairs.json').read_text())
    groups=json.loads((d/'duplicate-groups.json').read_text());exclusions=json.loads((d/'exclusions.json').read_text())
    byid={r['sample_id']:r for r in metadata};assert len(byid)==len(metadata)
    graph={i:set() for i in range(len(metadata))}
    for pair in pairs:
        i,j=pair['left'],pair['right'];assert 0<=i<j<len(metadata)
        if pair['kind']=='perceptual':
            left,right=pair['orientation'];distance=(int(metadata[i]['phash'][left],16)^int(metadata[j]['phash'][right],16)).bit_count()
            assert distance==pair['distance'] and distance<=8
            assert pair['linked']==(distance<=4 or pair['correlation']>=.98)
        else:assert metadata[i][pair['kind']]==metadata[j][pair['kind']]
        if pair['linked']:graph[i].add(j);graph[j].add(i)
    # Independent graph traversal, not producer union-find.
    seen=set();components=[]
    for start in graph:
        if start in seen:continue
        stack=[start];part=set()
        while stack:
            x=stack.pop()
            if x in part:continue
            part.add(x);stack.extend(graph[x]-part)
        seen|=part;components.append(sorted(metadata[i]['sample_id'] for i in part))
    assert sorted(components)==sorted(g['members'] for g in groups)
    # Independently demand every exact-identity group is closed in graph.
    membership={sid:i for i,g in enumerate(components) for sid in g}
    for key in ['sha256','pixel_sha256']:
        known={}
        for row in metadata:
            value=row[key];group=membership[row['sample_id']]
            if value in known:assert known[value]==group
            known[value]=group
    eligible={'train':[],'test':[]}
    for part in components:
        rows=[byid[s] for s in part]
        if len({r['target'] for r in rows})>1:continue
        source='test' if any(r['source_split']=='test' for r in rows) else 'train'
        eligible[source].append(sorted((r for r in rows if r['source_split']==source),key=lambda r:r['sample_id'])[0])
    expected={'train':[],'development':[],'final':eligible['test']}
    for target,train_count in [(0,334),(1,666)]:
        ordered=sorted((r for r in eligible['train'] if r['target']==target),key=lambda r:hashlib.sha256((SALT+'|'+r['sample_id']).encode()).hexdigest())
        expected['development']+=ordered[:64];expected['train']+=ordered[64:64+train_count]
    forbidden=set(json.loads((root/'data/historical-excluded-sha256.json').read_text())['sha256']);selected=[];counts={}
    for split,wanted in expected.items():
        rows=[json.loads(s) for s in (d/f'{split}-manifest.jsonl').read_text().splitlines()]
        assert sorted(r['sample_id'] for r in rows)==sorted(r['sample_id'] for r in wanted)
        counts[split]=dict(collections.Counter(r['target'] for r in rows))
        for r in rows:
            assert r['sha256'] not in forbidden and r['patient_id'] is None
            p=Path(r['path']);assert p.is_relative_to(root/'data')
            assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        selected+=rows
    for key in ['sample_id','sha256','pixel_sha256','group_id']:assert len({r[key] for r in selected})==len(selected)
    annotations={s:{r['file_name']:r for r in json.loads((root/'sources'/f'chestx-ChestX_Det_{s}.json').read_text())} for s in ['train','test']}
    for r in selected:assert r['target']==int(bool(annotations[r['source_split']][r['filename']]['syms']))
    archives={s:{'sha256':sha(root/'data'/f'{s}_data.zip'),'bytes':(root/'data'/f'{s}_data.zip').stat().st_size} for s in ['train','test']}
    report={'status':'PASS','scope':'separate metadata/graph/selection implementation and full selected-source byte hashes; producer image decoding/pHash/correlation not independent image reconstruction',
            'selected':len(selected),'counts':counts,'archive_identities':archives,'pair_candidates':len(pairs),'groups':len(groups),
            'image_modes':dict(collections.Counter(r['source_mode'] for r in metadata)),
            'cross_official_split_components':sum(len(g['source_splits'])>1 for g in groups),
            'patient_independence_verified':False,'model_calls':0,'admission_sha256':sha(d/'admission.json')}
    with (d/'independent-verification.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
