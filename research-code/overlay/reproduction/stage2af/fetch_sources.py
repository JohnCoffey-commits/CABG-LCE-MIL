"""Bounded primary-source metadata capture; no medical image access."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

def capture(root: Path, name: str, url: str, maximum: int = 20_000_000):
    target = root / 'sources' / name
    if target.exists():
        raise FileExistsError(target)
    req = urllib.request.Request(url, headers={'User-Agent': 'CABG-scale-study/1.0'})
    start = time.time()
    data = b''
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            final_url = response.url
            while True:
                part = response.read(min(1024*1024, maximum+1-len(data)))
                if not part: break
                data += part
                if len(data) > maximum: raise ValueError('Metadata size limit')
        with target.open('xb') as stream: stream.write(data)
        record = dict(name=name,url=url,final_url=final_url,status=status,
                      bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                      start_epoch=start,end_epoch=time.time())
    except Exception as exc:
        if data:
            with target.with_suffix(target.suffix+'.partial').open('xb') as f:f.write(data)
        record = dict(name=name,url=url,bytes=len(data),start_epoch=start,
                      end_epoch=time.time(),error=repr(exc))
        with (root/'control/downloads.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        raise
    with (root/'control/downloads.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    repo='Deepwise-AILab/ChestX-Det-Dataset'
    commit=json.loads(capture(a.root,'chestx-revision.json',f'https://api.github.com/repos/{repo}/commits/main'))['sha']
    for name in ('README.md','LICENSE','ChestX_Det_train.json','ChestX_Det_test.json'):
        capture(a.root,'chestx-'+name,f'https://raw.githubusercontent.com/{repo}/{commit}/{name}')
    capture(a.root,'jarvis-pricing.html','https://jarvislabs.ai/pricing')
    capture(a.root,'jarvis-vm.html','https://jarvislabs.ai/products/vm')
    summary={'source_commit':commit,'splits':{}}
    for split in ('train','test'):
        rows=json.loads((a.root/'sources'/f'chestx-ChestX_Det_{split}.json').read_text())
        summary['splits'][split]={'n':len(rows),'no_listed_findings':sum(not r['syms'] for r in rows),
                                  'with_listed_findings':sum(bool(r['syms']) for r in rows),
                                  'keys':sorted(rows[0]),'filename_examples':[r['file_name'] for r in rows[:3]]}
    (a.root/'sources/chestx-metadata-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
