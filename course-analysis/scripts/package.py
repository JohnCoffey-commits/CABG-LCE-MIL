"""Inventory and archive only this derived analysis package; no research execution."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/(ROOT.name+'.zip')
INVENTORY=ROOT/'provenance/output-inventory.json'
CHECKSUM=ARCHIVE.with_suffix('.zip.sha256')
CHECK=ROOT/'verification/archive-check.json'

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    assert json.loads((ROOT/'verification/independent-analysis-check.json').read_text())['status']=='PASS'
    excluded={ARCHIVE,INVENTORY,CHECKSUM,CHECK}
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p not in excluded and '__pycache__' not in p.parts)
    records=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in paths]
    INVENTORY.write_text(json.dumps({'scope':'analysis package; excludes inventory itself, archive, checksum and archive-check record','files':records},indent=2)+'\n')
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[INVENTORY]:z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(records)+1
        for row in records:
            data=z.read(ROOT.name+'/'+row['path'])
            assert len(data)==row['bytes'] and digest(data)==row['sha256']
        assert z.read(ROOT.name+'/'+str(INVENTORY.relative_to(ROOT)))==INVENTORY.read_bytes()
    sha=digest(ARCHIVE.read_bytes())
    CHECKSUM.write_text(sha+'  '+ARCHIVE.name+'\n')
    result={'status':'PASS','files_in_archive':len(records)+1,'archived_payload_bytes':sum(r['bytes'] for r in records),'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha,'all_member_hashes_verified':True}
    CHECK.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
