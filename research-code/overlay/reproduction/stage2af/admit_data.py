"""Model-independent ChestX-Det admission, duplicate grouping and fixed splits."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

SALT='cabg-stage2af-chestx-20260918-v1'
FINDINGS={'Atelectasis','Calcification','Cardiomegaly','Consolidation','Diffuse Nodule',
          'Effusion','Emphysema','Fibrosis','Fracture','Mass','Nodule','Pleural Thickening','Pneumothorax'}

def digest(value):return hashlib.sha256(value).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def representations(image):
    gray=image.convert('L');array=np.asarray(gray)
    ys,xs=np.where(array>5)
    trimmed=gray.crop((int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)) if len(xs) else gray
    hashes=[];thumbs=[]
    for im in [gray,ImageOps.mirror(gray),trimmed,ImageOps.mirror(trimmed)]:
        small=np.asarray(im.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
        frequencies=dctn(small,type=2,norm='ortho')[:8,:8].ravel()
        median=np.median(frequencies[1:]);bits=frequencies>median
        hashes.append(int.from_bytes(np.packbits(bits).tobytes(),'big'))
        thumb=np.asarray(im.resize((128,128),Image.Resampling.BILINEAR),dtype=np.float32).ravel()
        thumb=thumb-thumb.mean();norm=np.linalg.norm(thumb)
        thumbs.append(thumb/max(float(norm),1e-12))
    return hashes,thumbs

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'data/admission-v1';out.mkdir(exist_ok=False)
    excluded_hashes=set(json.loads((root/'data/historical-excluded-sha256.json').read_text())['sha256'])
    rows=[];bad=[];hashes=[];thumbs=[];archive_counts={}
    for split in ['train','test']:
        annotations=json.loads((root/'sources'/f'chestx-ChestX_Det_{split}.json').read_text())
        files=collections.defaultdict(list)
        for path in (root/'data'/split).rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.png','.jpg','.jpeg'}:files[path.name].append(path)
        archive_counts[split]={'annotation_rows':len(annotations),'image_filenames':len(files)}
        seen=set()
        for source in annotations:
            name=source['file_name'];assert name not in seen;seen.add(name)
            if len(files[name])!=1:raise RuntimeError(f'Image/annotation mapping not unique: {split}/{name}')
            path=files[name][0];data=path.read_bytes();sha=digest(data)
            if sha in excluded_hashes:
                bad.append({'source_split':split,'filename':name,'reason':'historical_exclusion','sha256':sha});continue
            if not set(source['syms'])<=FINDINGS:raise RuntimeError('Unknown finding label')
            if not (len(source['syms'])==len(source['boxes'])==len(source['polygons'])):raise RuntimeError('Annotation count mismatch')
            try:
                with Image.open(path) as im:
                    im.load();source_mode=im.mode;rgb=im.convert('RGB');width,height=rgb.size;arr=np.asarray(rgb)
                if source_mode not in ('L','RGB','RGBA','P'):raise ValueError('Unreviewed source pixel mode '+source_mode)
                if min(width,height)<32 or arr.max()==arr.min():raise ValueError('Empty/constant/small image')
                for box,polygon in zip(source['boxes'],source['polygons']):
                    if len(box)!=4 or not np.isfinite(box).all() or not np.isfinite(polygon).all():raise ValueError('Nonfinite/malformed annotation')
                    if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('Box outside image geometry')
                hs,ts=representations(rgb)
            except Exception as exc:
                bad.append({'source_split':split,'filename':name,'reason':'quality_or_geometry','error':repr(exc),'sha256':sha});continue
            target=int(bool(source['syms']))
            rows.append({'sample_id':f'chestx-{Path(name).stem}-{sha[:12]}','path':str(path),'filename':name,
                         'source_split':split,'source_commit':'d82eb07e66aae1423ba2502522bafaf10cdceb0d',
                         'source_dataset':'ChestX-Det','target':target,'label':'listed_finding' if target else 'no_listed_finding',
                         'findings':source['syms'],'sha256':sha,'pixel_sha256':digest(arr.tobytes()),'bytes':len(data),
                         'width':width,'height':height,'source_mode':source_mode,'patient_id':None,'group_basis':'image_duplicate_component',
                         'phash':[f'{v:016x}' for v in hs]})
            hashes.append(hs);thumbs.append(ts)
            if len(rows)%250==0:print(json.dumps({'images_checked':len(rows),'excluded':len(bad)}),flush=True)
    n=len(rows);parent=list(range(n));pairs=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    exact={}
    for i,row in enumerate(rows):
        for kind in ['sha256','pixel_sha256']:
            key=(kind,row[kind])
            if key in exact:
                j=exact[key];union(i,j);pairs.append({'left':j,'right':i,'kind':kind,'linked':True})
            else:exact[key]=i
    # Candidate generation considers reflected and border-trimmed representations.
    h=np.array(hashes,dtype=np.uint64)
    for i in range(n):
        best=np.full(n-i-1,65,dtype=np.int16)
        orientation=np.zeros(n-i-1,dtype=np.int8)
        for k in range(4):
            # Compare full/trimmed references with both orientation alternatives.
            for anchor in [0,2]:
                distances=np.bitwise_count(h[i,anchor]^h[i+1:,k])
                mask=distances<best;best[mask]=distances[mask];orientation[mask]=anchor*4+k
        for offset in np.where(best<=8)[0]:
            j=i+1+int(offset);orientation_code=int(orientation[offset]);left=orientation_code//4;right=orientation_code%4
            correlation=float(np.dot(thumbs[i][left],thumbs[j][right]));distance=int(best[offset])
            linked=distance<=4 or correlation>=.98
            pairs.append({'left':i,'right':j,'kind':'perceptual','distance':distance,'correlation':correlation,
                          'orientation':[left,right],'linked':bool(linked)})
            if linked:union(i,j)
        if i%500==0:print(json.dumps({'pair_rows_scanned':i,'candidate_pairs':len(pairs)}),flush=True)
    components=collections.defaultdict(list)
    for i in range(n):components[find(i)].append(i)
    candidates={'train':[],'test':[]};groups=[]
    for members in components.values():
        member_ids=sorted(rows[i]['sample_id'] for i in members)
        gid='image-group-'+digest('\n'.join(member_ids).encode())[:20]
        labels={rows[i]['target'] for i in members};splits={rows[i]['source_split'] for i in members}
        groups.append({'group_id':gid,'members':member_ids,'targets':sorted(labels),'source_splits':sorted(splits)})
        if len(labels)>1:
            for i in members:bad.append({'sample_id':rows[i]['sample_id'],'reason':'contradictory_duplicate_component','group_id':gid})
            continue
        desired='test' if 'test' in splits else 'train'
        eligible=sorted((i for i in members if rows[i]['source_split']==desired),key=lambda i:rows[i]['sample_id'])
        chosen=eligible[0];rows[chosen]['group_id']=gid;candidates[desired].append(rows[chosen])
        for i in members:
            if i!=chosen:bad.append({'sample_id':rows[i]['sample_id'],'reason':'cross_split_duplicate' if rows[i]['source_split']!=desired else 'duplicate_noncanonical','group_id':gid})
    write(out/'all-image-metadata.json',rows);write(out/'duplicate-pairs.json',pairs);write(out/'duplicate-groups.json',groups)
    selected={'train':[],'development':[],'final':sorted(candidates['test'],key=lambda r:r['sample_id'])}
    for label,train_n,dev_n in [(0,334,64),(1,666,64)]:
        pool=sorted((r for r in candidates['train'] if r['target']==label),key=lambda r:digest((SALT+'|'+r['sample_id']).encode()))
        if len(pool)<train_n+dev_n:
            write(out/'insufficient-counts.json',{'target':label,'available':len(pool),'needed':train_n+dev_n,'candidate_counts':{s:dict(collections.Counter(r['target'] for r in rs)) for s,rs in candidates.items()}})
            raise RuntimeError('Locked target counts unavailable; inspect before model use')
        selected['development']+=pool[:dev_n];selected['train']+=pool[dev_n:dev_n+train_n]
        for r in pool[dev_n+train_n:]:bad.append({'sample_id':r['sample_id'],'reason':'not_selected_by_fixed_size'})
    for split,rs in selected.items():
        rs.sort(key=lambda r:r['sample_id'])
        for row in rs:row['split']=split
        with (out/f'{split}-manifest.jsonl').open('x') as f:
            for row in rs:f.write(json.dumps(row,sort_keys=True)+'\n')
        with (out/f'{split}-manifest.csv').open('x',newline='') as f:
            fields=['sample_id','path','source_split','source_dataset','target','label','group_id','patient_id','sha256','pixel_sha256','bytes','width','height']
            w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rs)
    for l,r in [('train','development'),('train','final'),('development','final')]:
        for key in ['sample_id','sha256','pixel_sha256','group_id']:assert not {x[key] for x in selected[l]}&{x[key] for x in selected[r]}
    write(out/'exclusions.json',bad)
    counts={s:dict(collections.Counter(r['label'] for r in rs)) for s,rs in selected.items()}
    result={'status':'PASS_IMAGE_LEVEL_ADMISSION','counts':counts,'unique_train':len(selected['train']),
            'patient_independence_verified':False,'all_selected_source_images_readable':True,
            'historical_excluded_images_opened':0,'model_evaluations':0,'source_archive_counts':archive_counts,
            'duplicate_components':sum(len(x['members'])>1 for x in groups),'excluded_or_unselected':len(bad),
            'exclusion_reasons':dict(collections.Counter(r['reason'] for r in bad)),
            'limits':['13 target annotation task, not all-disease health','image-level duplicate grouping, no patient mapping','base pretraining overlap unknown']}
    write(out/'admission.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
