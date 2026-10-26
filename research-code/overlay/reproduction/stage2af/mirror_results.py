"""Mirror verified stage metadata/readouts through the existing SSH socket only.

Does not export images, caches, checkpoints, or full gradient bodies. These stay
at their recorded remote locations. The destination must be a new directory.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from reproduction.stage2af.remote import SSH

ROOT = '/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1'

REMOTE_EXPORT = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(ROOT)
for rel in ['results-v1/independent-verification.json','training-independent-verification-v1.json','training-summary-v1/verification.json','coverage-independent-verification-v1.json','results-detail-v1/verification.json']:
 assert json.loads((root/rel).read_text())['status']=='PASS',rel
assert json.loads((root/'training-summary-v1/summary.json').read_text())['status']=='DESCRIPTIVE_REDUCTION_OF_INDEPENDENTLY_VERIFIED_RECORDS'
def selected(p):
 rel=p.relative_to(root);parts=rel.parts
 if p.is_symlink() or not p.is_file():return False
 if len(parts)==1:return p.suffix in ['.json','.md','.diff']
 top=parts[0]
 if top=='data':return parts[1] in ['admission-v1','prepared-v1'] and p.suffix in ['.json','.jsonl','.csv']
 if top=='environment':return len(parts)==2 and p.suffix in ['.json','.txt']
 if top=='code':return (parts[1]=='runtime' and parts[2:4]==('reproduction','stage2af') and p.suffix=='.py') or (parts[1]=='revisions' and p.suffix=='.py')
 if top in ['formal','debug']:return p.suffix in ['.json','.jsonl','.csv','.log']
 if top=='evaluation':return p.suffix in ['.json','.jsonl','.npy','.npz','.csv']
 if top in ['control','logs','sources','results-v1','results-detail-v1','training-summary-v1']:return p.suffix in ['.json','.jsonl','.csv','.log','.md','.py','.html','.txt'] or p.name.endswith('LICENSE')
 if top=='figures-v1':return p.suffix in ['.png','.svg','.json','.md','.py']
 return False
directories=['data/admission-v1','data/prepared-v1','environment','code/runtime/reproduction/stage2af','code/revisions','formal','debug','evaluation','control','logs','sources','results-v1','results-detail-v1','training-summary-v1','figures-v1']
candidates=[p for p in root.iterdir() if p.is_file()]
for directory in directories:
 base=root/directory
 if directory=='environment':candidates.extend(base.iterdir())
 elif base.exists():candidates.extend(base.rglob('*'))
files=sorted(set(p for p in candidates if selected(p)))
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in files:
  before=path.stat();body=path.read_bytes();after=path.stat()
  assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed during export'
  rel=path.relative_to(root).as_posix();manifest.append(dict(path=rel,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
  info=tarfile.TarInfo(rel);info.size=len(body);info.mtime=int(before.st_mtime);archive.addfile(info,io.BytesIO(body))
 body=(json.dumps(dict(status='VERIFIED_METADATA_AND_READOUT_EXPORT',remote_root=str(root),files=manifest,excluded_bodies='images/prepared cache/model checkpoints/full raw gradients'),indent=2)+'\n').encode()
 info=tarfile.TarInfo('mirror-manifest.json');info.size=len(body);archive.addfile(info,io.BytesIO(body))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists(), 'Use a fresh local output directory'
    assert shutil.disk_usage(out.parent).free >= 10_000_000_000
    out.mkdir()
    archive = out / 'transport.tar.gz'
    script = REMOTE_EXPORT.replace('ROOT', repr(ROOT), 1)
    with archive.open('xb') as stream:
        process = subprocess.run(SSH + ['python3 -B -'], input=script.encode(), stdout=stream,
                                 stderr=subprocess.PIPE, timeout=1200)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors='replace'))
    extracted = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            assert member.name not in extracted
            extracted.add(member.name)
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
    manifest = json.loads((out / 'mirror-manifest.json').read_text())
    expected = {row['path'] for row in manifest['files']}
    assert extracted == expected | {'mirror-manifest.json'}
    for row in manifest['files']:
        body = (out / row['path']).read_bytes()
        assert len(body) == row['bytes'] and hashlib.sha256(body).hexdigest() == row['sha256']
    result = dict(status='PASS', verified_files=len(expected), bytes=sum(r['bytes'] for r in manifest['files']),
                  archive_bytes=archive.stat().st_size, remote_root=ROOT,
                  scope='local body hashes match export manifest; complements remote numerical verification',
                  excluded_bodies=manifest['excluded_bodies'])
    (out / 'local-mirror-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
