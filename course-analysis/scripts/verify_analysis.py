"""Separate-code verification of figures/tables from frozen snapshots.

Does not import the producer. Rank/pair-count and explicitly expanded-resample
implementations check sklearn/weighted calculations. Same operator, not new data.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import json,hashlib,re,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data/source';T=ROOT/'tables'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return pd.read_csv(SRC/name)
def table(name):return pd.read_csv(T/name)
def close(x,y,tol=3e-12):assert abs(float(x)-float(y))<tol,(x,y)
def auc(y,s):
    n1=int((y==1).sum());n0=int((y==0).sum())
    return (rankdata(s,method='average')[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def ap(y,s):
    # Direct threshold formula; independent of the producer's vectorized rank grouping.
    result=0.;n1=int((y==1).sum())
    for value in np.unique(s[y==1]):
        at=(s==value)&(y==1);above=s>=value
        result+=at.sum()/n1*((y==1)&above).sum()/above.sum()
    return result
def measures(g):
    y=g.target.to_numpy(int);p=g.parsed.to_numpy(int);s=g.score.to_numpy(float)
    return {'auroc':auc(y,s),'average_precision':ap(y,s),'balanced_accuracy':.5*((p[y==0]==0).mean()+(p[y==1]==1).mean()),
            'accuracy':(p==y).mean(),'n':len(y),'correct':int((p==y).sum()),'invalid':int((~np.isin(p,[0,1])).sum())}

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.add(d['id'])
        for k in ['href','src']:
            if k in d:self.links.append(d[k])

def main():
    checked={'source_snapshots':0,'original_sources_available':0,'endpoint_reductions':0,'accepted_pairs':0,
             'expanded_bootstrap_replicates':0,'interval_rows':0,'figures':0,'local_html_links':0}
    manifest=json.loads((ROOT/'provenance/input-manifest.json').read_text())
    for r in manifest['files']:
        p=ROOT/r['snapshot'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];checked['source_snapshots']+=1
        orig=Path(r['source_absolute'])
        if orig.exists():assert sha(orig)==r['sha256'];checked['original_sources_available']+=1
    raw={k:load(k+'_per_image.csv') for k in ['af','ah']}
    master=table('01_endpoint_master.csv').set_index('endpoint');maximum=0
    for stage,d in raw.items():
        ref=load(stage+'_endpoints.csv').set_index('endpoint')
        assert not d.duplicated(['endpoint','sample_id']).any()
        for endpoint,g in d.groupby('endpoint'):
            mm=measures(g)
            for key,v in mm.items():
                close(v,ref.loc[endpoint,key]);close(v,master.loc[endpoint,key]);maximum=max(maximum,abs(v-ref.loc[endpoint,key]))
            # Pairwise AUROC probability provides another direct arithmetic check.
            yp=g.target.to_numpy();s=g.score.to_numpy();sp=s[yp==1,None];sn=s[yp==0][None,:]
            close(((sp>sn)+.5*(sp==sn)).mean(),mm['auroc'])
            checked['endpoint_reductions']+=1
        for _,r in table('02_scale_paired_original.csv' if stage=='af' else '03_transport_paired_original.csv').iterrows():
            a=d[d.endpoint==r.left].set_index('sample_id').sort_index();b=d[d.endpoint==r.right].set_index('sample_id').sort_index()
            assert a.index.equals(b.index) and a.target.equals(b.target)
            ca=(a.parsed==a.target).to_numpy();cb=(b.parsed==b.target).to_numpy()
            for name,v in [('left_only_correct',(ca&~cb).sum()),('right_only_correct',(cb&~ca).sum()),('both_correct',(ca&cb).sum()),('both_wrong',(~ca&~cb).sum())]:close(v,r[name])
            close(r.delta_balanced_accuracy,measures(a)['balanced_accuracy']-measures(b)['balanced_accuracy'])
            checked['accepted_pairs']+=1
    # Explicitly reconstruct the first batch and expand six sampled image lists.
    final=raw['af'][raw['af'].endpoint.str.endswith('-final')]
    base=final[final.endpoint=='linear-head-final'].sort_values(['target','sample_id']);ids=base.sample_id.tolist();y=base.target.to_numpy(int)
    z=np.load(ROOT/'data/derived/bootstrap_final.npz');eps=z['endpoints'].tolist();draw=z['values'];assert draw.shape==(20000,7,3)
    rng=np.random.default_rng(20260920)
    w=np.concatenate([rng.multinomial(64,np.ones(64)/64,size=256),rng.multinomial(478,np.ones(478)/478,size=256)],axis=1)
    for j,endpoint in enumerate(eps):
        g=final[final.endpoint==endpoint].set_index('sample_id').loc[ids].reset_index()
        for r in [0,1,2,11,127,255]:
            expanded=g.iloc[np.repeat(np.arange(542),w[r])];m=measures(expanded)
            for k,col in enumerate(['auroc','average_precision','balanced_accuracy']):close(m[col],draw[r,j,k])
            checked['expanded_bootstrap_replicates']+=1
    for _,r in table('05_scale_final_with_posthoc_intervals.csv').iterrows():
        j=eps.index(r.endpoint)
        for k,col in enumerate(['auroc','average_precision','balanced_accuracy']):
            lo,hi=np.quantile(draw[:,j,k],[.025,.975]);close(lo,r[col+'_ci_low']);close(hi,r[col+'_ci_high'])
        checked['interval_rows']+=1
    for _,r in table('06_auroc_paired_posthoc.csv').iterrows():
        v=draw[:,eps.index(r.left),0]-draw[:,eps.index(r.right),0]
        for col,q in [('ci95_low',.025),('ci95_high',.975),('family8_ci_low',.05/16),('family8_ci_high',1-.05/16)]:close(np.quantile(v,q),r[col])
        checked['interval_rows']+=1
    for _,r in table('18_ba_final_multiplicity_sensitivity.csv').iterrows():
        v=draw[:,eps.index(r.left),2]-draw[:,eps.index(r.right),2]
        for col,q in [('new_ci95_low',.025),('new_ci95_high',.975),('family12_ci_low',.05/24),('family12_ci_high',1-.05/24)]:close(np.quantile(v,q),r[col])
        checked['interval_rows']+=1
    # All original AG endpoint rows are reconciled with their accepted, independently checked summaries.
    a1=json.loads((SRC/'ag_a1_summary.json').read_text());a2=json.loads((SRC/'ag_a2_verification.json').read_text())
    for _,r in load('ag_endpoints.csv').iterrows():
        ref=(a1 if r.stage=='A1' else a2)['endpoints'][r.endpoint]
        for k in ['n','correct','invalid','balanced_accuracy','accuracy','auroc','average_precision']:close(r[k],ref[k])
    for _,r in table('12_intervention_effects.csv').iterrows():
        if r.stage=='A1':
            ref=a1['endpoints'][f's{r.seed}-{r.condition}'];pair=a1['comparisons'][f's{r.seed}-{r.condition}__minus__native']
            close(r.answers_changed,ref['answers_changed'])
        else:pair=a2['comparisons'][f's{r.seed}'];close(r.answers_changed,pair['answers_changed'])
        close(r.corrected,pair['left_only_correct']);close(r.harmed,pair['right_only_correct']);close(r.delta_BA,pair['delta_balanced_accuracy'])
        assert r.answers_changed==r.corrected+r.harmed
    for _,r in table('09_readout_comparison_posthoc.csv').iterrows():
        g=raw['af'][raw['af'].endpoint==r.endpoint];close(r.first_token_margin_auroc,auc(g.target.to_numpy(),g.yes_no_logit_gap.to_numpy()))
    steps=load('af_steps.csv');assert len(steps)==6000
    for _,r in table('10_training_control_and_resources.csv').iterrows():
        g=steps[steps.trajectory==r.trajectory].sort_values('step');assert list(g.step)==list(range(1,1001))
        close(r.ratio_max,max(g.auxiliary_to_lm_shared_ratio));close(r.ratio_median,np.median(g.auxiliary_to_lm_shared_ratio))
        if 'cabg' in r.trajectory:assert r.ratio_max<.200001
    catalog=json.loads((ROOT/'figures/catalog.json').read_text());assert len(catalog)==16
    for item in catalog:
        for path in item['outputs']:
            p=ROOT/path;assert p.is_file() and p.stat().st_size>1000
            if p.suffix=='.png':
                with Image.open(p) as im:assert min(im.size)>2000 and min(im.info.get('dpi',(0,0)))>449
            elif p.suffix=='.svg':ET.parse(p)
            else:assert p.read_bytes().startswith(b'%PDF') and len(re.findall(rb'/Type\s*/Page\b',p.read_bytes()))==1
        for data,name in zip(item['data'],item['tables']):
            aa=pd.read_csv(ROOT/data);bb=table(name)
            pd.testing.assert_frame_equal(aa,bb,check_exact=False,rtol=1e-12,atol=1e-12)
        checked['figures']+=1
    assert len(re.findall(rb'/Type\s*/Page\b',(ROOT/'figures-all.pdf').read_bytes()))==16
    parser=Links();parser.feed((ROOT/'analysis-index.html').read_text())
    for link in parser.links:
        if link.startswith(('https:','http:','mailto:','data:')):continue
        parts=urlparse(link)
        if not parts.path:
            if parts.fragment:assert parts.fragment in parser.ids,link
        else:assert (ROOT/unquote(parts.path)).is_file(),link
        checked['local_html_links']+=1
    assert len(list(T.glob('*.csv')))==18
    result={'status':'PASS','scope':'separate implementation, same operator; numerical/asset verification, not new model validation',
            'checks':checked,'max_original_endpoint_error':maximum,
            'all_input_snapshots_hash_verified':True,'new_model_calls':0,
            'bootstrap_check':'42 explicitly expanded resamples across all7 endpoints; quantiles checked for all27 interval rows',
            'limitations':['No patient-level independence','No independent-laboratory replication','Post-hoc intervals; descriptive image resampling',
                          'Visual QA is recorded separately; automated format checks do not establish visual quality']}
    (ROOT/'verification/independent-analysis-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
