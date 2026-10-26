"""Export verified numerical evidence into compact tables; local data-only operation."""
import argparse,csv,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    execution=root/'verified-execution-v1';a1path=execution/'results/a1-v1/summary.json';a2path=execution/'results/a2-independent-verification-v1.json'
    a1=json.loads(a1path.read_text());a2=json.loads(a2path.read_text())
    assert json.loads((execution/'results/a1-v1/independent-verification-v1.json').read_text())['status']=='PASS' and a2['status']=='PASS'
    out=root/'tables';out.mkdir(exist_ok=False);rows=[];comparisons=[]
    for stage,data,source in [('A1',a1,a1path),('A2',a2,a2path)]:
        for name,v in data['endpoints'].items():
            rows.append(dict(stage=stage,endpoint=name,n=v['n'],negative_n=v['by_class']['0']['n'],negative_correct=v['by_class']['0']['correct'],positive_n=v['by_class']['1']['n'],positive_correct=v['by_class']['1']['correct'],correct=v['correct'],invalid=v['invalid'],balanced_accuracy=v['balanced_accuracy'],accuracy=v['accuracy'],auroc=v['auroc'],average_precision=v['average_precision'],source=str(source.relative_to(root))))
        for name,v in data['comparisons'].items():
            comparisons.append(dict(stage=stage,comparison=name,n=v['n'],left_only_correct=v['left_only_correct'],right_only_correct=v['right_only_correct'],both_correct=v['both_correct'],both_wrong=v['both_wrong'],delta_BA=v['delta_balanced_accuracy'],ci_low=v['paired_balanced_accuracy_95ci'][0],ci_high=v['paired_balanced_accuracy_95ci'][1],source=str(source.relative_to(root))))
    for name,content in [('endpoints.csv',rows),('paired-comparisons.csv',comparisons)]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=content[0].keys());w.writeheader();w.writerows(content)
    manifest=dict(status='VERIFIED_SOURCE_EXPORT',tables=dict(endpoints=len(rows),comparisons=len(comparisons)),population='same128 development images,64 per class; repeated records are not new cases',units='metrics and intervals as proportions; no rounding in CSV',interval_method='10000 class-stratified paired image bootstrap; descriptive, unadjusted; two seeds not population inference',missing='none; invalid counted wrong, all denominators retained',figures=[],figure_reason='No plot needed for these compact discrete endpoints; no synthetic or planning results',sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a1path,a2path]})
    (out/'table-and-figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':main()
