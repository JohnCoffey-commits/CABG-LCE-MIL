"""Lightweight existing-number analysis only; no images, weights or model imports."""
import argparse,hashlib,json,csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr

def auc(y,x):
    y=np.array(y);x=np.array(x);p=sum(y);n=len(y)-p
    return float((sum(rankdata(x)[y==1])-p*(p+1)/2)/(p*n))

def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=dict(scope='existing128 development records only; descriptive, no threshold search',historical={},branch_swaps={},source_files=[])
    def read(path):
        b=path.read_bytes();report['source_files'].append(dict(path=str(path),sha256=hashlib.sha256(b).hexdigest()))
        return json.loads(b)
    for seed in [43,44]:
        for arm in ['lm_only','fixed_lce','cabg_lce']:
            rs=read(a.old/'evaluation'/f's{seed}-{arm}-step1000-development/v1/observations.json')
            y=np.array([x['target'] for x in rs]);g=np.array([x['yes_no_logit_gap'] for x in rs]);s=np.array([x['score'] for x in rs]);c=np.array([x['parsed']==x['target'] for x in rs]);pred=np.array([x['parsed'] for x in rs])
            report['historical'][f's{seed}-{arm}']=dict(n=len(rs),correct=int(sum(c)),class_correct={str(k):int(sum(c[y==k])) for k in [0,1]},score_auroc=auc(y,s),logit_auroc=auc(y,g),score_logit_spearman=float(spearmanr(s,g).statistic),logit_argmax_generation_disagreement=int(sum((g>0)!=pred)),zero_gap_ties=int(sum(g==0)),median_logit_by_class={str(k):float(np.median(g[y==k])) for k in [0,1]},yes_answers=int(sum(pred==1)),invalid=int(sum([x['parsed'] is None for x in rs])))
        rows=read(a.new/'diagnostics'/f'seed{seed}'/f'diagnostic-s{seed}-v1/observations.json');native={x['sample_id']:x for x in rows if x['condition']=='native'}
        for condition in ['ano_swap','visual_swap']:
            rs=[x for x in rows if x['condition']==condition];changed=[x for x in rs if x['parsed']!=native[x['sample_id']]['parsed']]
            counts=dict(n=len(rs),changed=len(changed),changed_follow_donor_answer=sum(x['parsed']==native[x['donor_sample_id']]['parsed'] for x in changed),all_follow_donor_answer=sum(x['parsed']==native[x['donor_sample_id']]['parsed'] for x in rs),subsets={})
            for label_match in [False,True]:
                subset=[x for x in rs if (x['target']==native[x['donor_sample_id']]['target'])==label_match]
                counts['subsets']['same_label' if label_match else 'opposite_label']=dict(n=len(subset),correct=sum(x['parsed']==x['target'] for x in subset),native_correct=sum(native[x['sample_id']]['parsed']==x['target'] for x in subset))
            report['branch_swaps'][f's{seed}-{condition}']=counts
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_files'},indent=2))

if __name__=='__main__':main()
