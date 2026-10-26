"""Complete predetermined comparisons; requires the sealed final release."""
import argparse
import csv
import json
from pathlib import Path

from reproduction.stage2af.metrics import summary, paired


def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;old=Path('/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1')
    assert (root/'final-evaluation-release.json').is_file()
    accepted={};oldaccepted=read(old/'accepted-endpoints.json')
    for seed in [43,44]:
        for split in ['development','final']:
            for step in ([24,100,300,1000] if split=='development' else [1000]):
                for arm in ['aligned','shuffled','fixed_lce','lm_only','cabg_lce']:
                    name=f's{seed}-{arm}-step{step:04}-{split}'
                    new=arm in ['aligned','shuffled']
                    path=root/'evaluation'/name/'v1' if new else Path(oldaccepted[name])
                    assert read(path/'summary.json')['status']=='PASS'
                    accepted[name]=dict(path=str(path),origin='stage2ah' if new else 'stage2af')
                    if new and arm=='aligned' and split=='development' and step==1000:
                        extra=name+'-inference_shuffle';extra_path=root/'evaluation'/extra/'v1'
                        assert read(extra_path/'summary.json')['status']=='PASS'
                        accepted[extra]=dict(path=str(extra_path),origin='stage2ah')
    # Reuse the old fitted head strictly as classification context.
    for split in ['development','final']:
        key=next((k for k in oldaccepted if k.startswith('linear-head') and (split in k or (split=='development' and k=='linear-head-fit'))),None)
        assert key is not None,(split,list(oldaccepted))
        accepted['linear-head-'+split]=dict(path=oldaccepted[key],origin='stage2af',subset=split)
    write(root/'accepted-endpoints.json',accepted)
    out=root/'results-v1';out.mkdir(exist_ok=False)
    table=[];observed={};perimage=[];classrows=[]
    for name,entry in accepted.items():
        rows=read(Path(entry['path'])/'observations.json')
        if entry.get('subset'):rows=[r for r in rows if r['split']==entry['subset']]
        assert rows
        observed[name]=rows;metrics=summary(rows)
        table.append(dict(endpoint=name,origin=entry['origin'],split=rows[0]['split'],**{k:v for k,v in metrics.items() if not isinstance(v,dict)}))
        for y,r in metrics['by_class'].items():classrows.append(dict(endpoint=name,target=y,**r))
        for row in rows:perimage.append(dict(endpoint=name,**{k:row.get(k) for k in ['sample_id','target','parsed','score','text','yes_no_logit_gap','effective_support','top11_mass']}))
    comparisons={}
    for seed in [43,44]:
        for split in ['development','final']:
            for step in ([24,100,300,1000] if split=='development' else [1000]):
                suffix=f'-step{step:04}-{split}'
                for left,right in [('aligned','fixed_lce'),('aligned','shuffled'),('aligned','lm_only'),('aligned','cabg_lce'),('shuffled','fixed_lce')]:
                    lk=f's{seed}-{left}'+suffix;rk=f's{seed}-{right}'+suffix
                    comparisons[lk+'__minus__'+rk]=paired(observed[lk],observed[rk],bootstrap_seed=20260919)
        lk=f's{seed}-aligned-step1000-development-inference_shuffle';rk=f's{seed}-aligned-step1000-development'
        comparisons[lk+'__minus__'+rk]=paired(observed[lk],observed[rk],bootstrap_seed=20260919)
    write(out/'endpoint-summary.json',table);write(out/'paired-comparisons.json',comparisons)
    for name,rows in [('endpoint-summary.csv',table),('class-counts.csv',classrows),('per-image.csv',perimage),
                      ('paired-comparisons.csv',[dict(comparison=k,**v) for k,v in comparisons.items()])]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    write(out/'analysis-status.json',dict(status='PENDING_INDEPENDENT_VERIFICATION',endpoints=len(table),comparisons=len(comparisons)))
    print(json.dumps(dict(endpoints=len(table),comparisons=len(comparisons))),flush=True)


if __name__=='__main__':main()
