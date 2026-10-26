"""Small prespecified source-mode and duplicate QC montage, never model-driven."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;d=r/'data/admission-v1'
    rows=json.loads((d/'all-image-metadata.json').read_text());pairs=json.loads((d/'duplicate-pairs.json').read_text())
    cross=[x for x in pairs if x['linked'] and rows[x['left']]['source_split']!=rows[x['right']]['source_split']]
    conflict=[x for x in pairs if x['linked'] and rows[x['left']]['target']!=rows[x['right']]['target']]
    chosen=[]
    for name,subset in [('cross',cross),('conflict',conflict)]:
        for pair in sorted(subset,key=lambda x:(x.get('distance',-1),x['left'],x['right']))[:2]:
            for side in ['left','right']:chosen.append((name,rows[pair[side]]))
    rgba=[x for x in rows if x['source_mode']=='RGBA'];alpha=[]
    for row in rgba:
        with Image.open(row['path']) as im:extrema=im.getchannel('A').getextrema()
        alpha.append({'sample_id':row['sample_id'],'alpha_minmax':extrema})
    chosen += [('RGBA',x) for x in rgba[:4]]
    train=[json.loads(s) for s in (d/'train-manifest.jsonl').read_text().splitlines()]
    chosen += [('selected_train',x) for y in [0,1] for x in [z for z in train if z['target']==y][:2]]
    canvas=Image.new('RGB',(1280,4*350),'white');draw=ImageDraw.Draw(canvas)
    for i,(kind,row) in enumerate(chosen):
        with Image.open(row['path']) as im:thumb=ImageOps.contain(im.convert('RGB'),(320,320))
        x=(i%4)*320;y=(i//4)*350;canvas.paste(thumb,(x+(320-thumb.width)//2,y))
        draw.text((x+4,y+321),f"{kind} {row['source_split']} {row['filename']} y={row['target']}",fill='black')
    canvas.save(d/'qc-preview-v1.jpg',quality=90)
    with (d/'qc-preview-v1.json').open('x') as f:json.dump({'selection':'two lowest-distance cross-split pairs; two lowest-distance contradictory pairs; first4 RGBA; first2 train per class','rows':[{'kind':k,'sample_id':x['sample_id']} for k,x in chosen],'all_RGBA_alpha':alpha},f,indent=2)
if __name__=='__main__':main()
