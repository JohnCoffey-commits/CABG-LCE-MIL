"""Bounded resumable primary image archive download; preserves partial files."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

def append(root, record):
    with (root/'control/downloads.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())

def received(root):
    return sum(json.loads(line).get('bytes',0) for line in (root/'control/downloads.jsonl').read_text().splitlines())

def download(root, split, expected):
    url=f'https://resource.deepwise.com/ChestX-Det/{split}_data.zip'
    path=root/'data'/f'{split}_data.zip'
    start_size=path.stat().st_size if path.exists() else 0
    if start_size>expected:raise ValueError('Archive larger than registered source')
    if start_size==expected:
        print(json.dumps({'reused_complete_download':str(path),'bytes':expected}),flush=True)
        return path
    prior=received(root)
    if prior+expected-start_size>50_000_000_000:raise RuntimeError('Download budget')
    if shutil.disk_usage(root).free-(expected-start_size)<50_000_000_000:raise RuntimeError('Disk floor')
    request=urllib.request.Request(url,headers={'User-Agent':'CABG-scale-study/1.0',**({'Range':f'bytes={start_size}-'} if start_size else {})})
    new=0;started=time.time();chunk_since=0
    try:
        with urllib.request.urlopen(request,timeout=60) as response:
            if start_size and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start_size}-')):
                raise ValueError('Server did not honor exact byte resume; retained partial file')
            with path.open('ab' if start_size else 'xb') as f:
                while True:
                    cost=(time.time()-json.loads((root/'resource-ledger.json').read_text())['start_epoch'])/3600*.51518
                    if cost>=49.95 or shutil.disk_usage(root).free<51_000_000_000:raise RuntimeError('Cost or disk floor')
                    chunk=response.read(4*1024*1024)
                    if not chunk:break
                    new+=len(chunk);chunk_since+=len(chunk)
                    if start_size+new>expected or prior+new>50_000_000_000:raise RuntimeError('Download exceeds budget/size')
                    f.write(chunk)
                    if chunk_since>=64*1024*1024:
                        f.flush();os.fsync(f.fileno())
                        append(root,dict(kind='image_download_chunk',url=url,bytes=chunk_since,epoch=time.time(),path=str(path),offset=start_size+new))
                        chunk_since=0
                        print(json.dumps({'split':split,'bytes':start_size+new,'expected':expected}),flush=True)
    finally:
        if chunk_since:append(root,dict(kind='image_download_chunk',url=url,bytes=chunk_since,epoch=time.time(),path=str(path),offset=start_size+new))
    if path.stat().st_size!=expected:raise RuntimeError('Incomplete archive; resumable')
    return path

def extract(root,split,path):
    output=root/'data'/split
    with zipfile.ZipFile(path) as z:
        entries=[i for i in z.infolist() if not i.is_dir()]
        if sum(i.file_size for i in entries)>50_000_000_000:raise RuntimeError('Expanded size cap')
        if shutil.disk_usage(root).free-sum(i.file_size for i in entries)<50_000_000_000:raise RuntimeError('Extraction disk floor')
        rows=[]
        for info in entries:
            member=Path(info.filename)
            if member.is_absolute() or '..' in member.parts:raise ValueError('Unsafe archive path')
            if member.suffix.lower() not in {'.png','.jpg','.jpeg'}:continue
            target=output/member
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.stat().st_size!=info.file_size:raise ValueError('Existing partial extracted file; preserve for repair')
            else:
                with z.open(info) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest)
            rows.append({'member':info.filename,'bytes':info.file_size,'crc32':info.CRC})
    record={'split':split,'archive_bytes':path.stat().st_size,'image_files':len(rows),'entries':rows}
    out=root/'data'/f'{split}-archive-inventory.json'
    if not out.exists():out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='entries'}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for split,size in [('train',1219964333),('test',222502111)]:extract(a.root,split,download(a.root,split,size))

if __name__=='__main__':main()
