"""Assemble complete accepted endpoint records; never select by performance."""
import argparse,csv,json
from pathlib import Path
from reproduction.stage2af.metrics import paired
from reproduction.stage2af.frozen import require_frozen
from reproduction.stage2af.runtime import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;plan,protocol=require_frozen(root)
    # Explicit accepted paths, fixed by execution status, are supplied by closure.
    accepted=json.loads((root/'accepted-endpoints.json').read_text());out=root/'results-v1';out.mkdir(exist_ok=False)
    table=[];observations={};per_image=[]
    for name,path in accepted.items():
        d=Path(path);s=json.loads((d/'summary.json').read_text());assert s['status']=='PASS' and s['protocol_sha256']==protocol
        rows=json.loads((d/'observations.json').read_text())
        if name.startswith('linear-head'):
            for split in sorted({r['split'] for r in rows}):
                key='linear-head-'+split;observations[key]=[r for r in rows if r['split']==split]
                table.append({'endpoint':key,'split':split,**{k:v for k,v in s['metrics'][split].items() if not isinstance(v,dict)}})
        else:
            observations[name]=rows;table.append({'endpoint':name,'split':rows[0]['split'],**{k:v for k,v in s['metrics'].items() if not isinstance(v,dict)}})
    for name,rows in observations.items():
        for row in rows:per_image.append({'endpoint':name,**{k:row.get(k) for k in ['sample_id','target','parsed','score','text','effective_support','top11_mass','yes_no_logit_gap','pixel_sha256_prepared']}})
    comparisons={}
    for seed in [43,44]:
        for split in ['development','final']:
            steps=[24,100,300,1000] if split=='development' else [1000]
            for step in steps:
                prefix=f's{seed}-';suffix=f'-step{step:04}-{split}'
                for left,right in [('cabg_lce','lm_only'),('cabg_lce','fixed_lce'),('fixed_lce','lm_only')]:
                    lk=prefix+left+suffix;rk=prefix+right+suffix
                    if lk in observations and rk in observations:comparisons[lk+'__minus__'+rk]=paired(observations[lk],observations[rk])
                if step==1000:
                    for arm in plan['arms']:
                        lk=prefix+arm+suffix;rk='linear-head-'+split
                        if lk in observations and rk in observations:comparisons[lk+'__minus__'+rk]=paired(observations[lk],observations[rk])
    for name,rows in [('endpoint-summary.csv',table),('per-image.csv',per_image)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    write(out/'paired-comparisons.json',comparisons);write(out/'endpoint-summary.json',table)
    write(out/'analysis-status.json',{'status':'GENERATED_PENDING_INDEPENDENT_VERIFICATION','accepted_readouts':len(accepted),'summary_rows':len(table),'per_image_rows':len(per_image),'protocol_sha256':protocol})
if __name__=='__main__':main()
