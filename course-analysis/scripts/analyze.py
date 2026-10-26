"""Numerical reductions of saved scores/logs only. No model/image/checkpoint I/O."""
from pathlib import Path
import json, re, time, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/source'; TAB=ROOT/'tables'; DER=ROOT/'data/derived'
NBOOT=20000; RNG_SEED=20260920
LABELS={'lm_only':'LM-only','fixed_lce':'Fixed LCE-MIL','cabg_lce':'CABG-LCE-MIL',
        'aligned':'Aligned transport','shuffled':'Shuffled transport','linear_head':'Linear head'}

def read(name): return pd.read_csv(SRC/name)
def js(name): return json.loads((SRC/name).read_text())
def save(df,name): df.to_csv(TAB/name,index=False,float_format='%.15g')
def writejson(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def ident(endpoint):
    if endpoint.startswith('linear-head-'):
        return {'seed':None,'method':'linear_head','step':None,'split':endpoint.split('-')[-1],
                'readout':'fixed classifier','inference_shuffle':False}
    m=re.match(r's(\d+)-(.+)-step(\d+)-(development|final)(.*)',endpoint)
    assert m,endpoint
    return {'seed':int(m[1]),'method':m[2],'step':int(m[3]),'split':m[4],
            'readout':'generated answer','inference_shuffle':bool(m[5])}

def metrics(g):
    y=g.target.to_numpy(int); p=g.parsed.to_numpy(int); s=g.score.to_numpy(float)
    assert set(y)=={0,1} and np.isfinite(s).all()
    n0=int((y==0).sum());n1=int((y==1).sum());tn=int(((y==0)&(p==0)).sum());tp=int(((y==1)&(p==1)).sum())
    return {'n':len(g),'negative_n':n0,'positive_n':n1,'tn':tn,'fp':int(((y==0)&(p==1)).sum()),
            'fn':int(((y==1)&(p==0)).sum()),'tp':tp,'invalid':int((~np.isin(p,[0,1])).sum()),
            'correct':int((p==y).sum()),'negative_recall':tn/n0,'positive_recall':tp/n1,
            'accuracy':float((p==y).mean()),'balanced_accuracy':.5*(tn/n0+tp/n1),
            'auroc':float(roc_auc_score(y,s)),'average_precision':float(average_precision_score(y,s))}

def endpoint_reduce(raw,expected,stage):
    assert not raw.duplicated(['endpoint','sample_id']).any()
    assert set(raw.endpoint)==set(expected.endpoint)
    rows=[];errs=[]
    for ep,g in raw.groupby('endpoint',sort=True):
        r={'stage':stage,'endpoint':ep,**ident(ep),**metrics(g)}
        ref=expected.set_index('endpoint').loc[ep]
        for col in ['n','correct','invalid','accuracy','balanced_accuracy','auroc','average_precision']:
            er=abs(float(ref[col])-r[col]);errs.append(er)
            assert er<2e-12,(stage,ep,col,er)
        rows.append(r)
    return pd.DataFrame(rows),max(errs)

def pairs_reduce(raw,pairs,stage):
    groups={ep:g.set_index('sample_id').sort_index() for ep,g in raw.groupby('endpoint')}
    checks=[]
    for _,r in pairs.iterrows():
        if stage=='af': a,b=r['left'],r['right']
        else: a,b=r.comparison.split('__minus__')
        aa,bb=groups[a],groups[b]
        assert aa.index.equals(bb.index) and aa.target.equals(bb.target)
        ca=(aa.parsed==aa.target).to_numpy();cb=(bb.parsed==bb.target).to_numpy()
        out={'stage':stage,'left':a,'right':b,'n':len(aa),
             'left_only_correct':int((ca&~cb).sum()),'right_only_correct':int((cb&~ca).sum()),
             'both_correct':int((ca&cb).sum()),'both_wrong':int((~ca&~cb).sum())}
        ma,mb=metrics(aa),metrics(bb)
        for k in ['accuracy','balanced_accuracy','auroc','average_precision']:
            out['delta_'+k]=ma[k]-mb[k]
        for k,v in out.items():
            if k in r.index and k not in ('stage','left','right'): assert abs(float(r[k])-v)<2e-12,(stage,a,b,k)
        if stage=='af':
            out['ba_ci_low']=float(r.balanced_accuracy_ci95_low);out['ba_ci_high']=float(r.balanced_accuracy_ci95_high)
        else:
            out['ba_ci_low'],out['ba_ci_high']=json.loads(r.paired_balanced_accuracy_95ci)
        out['uncertainty']='Original 10000 paired stratified image bootstrap; descriptive, unadjusted'
        checks.append(out)
    return pd.DataFrame(checks)

def weighted_metrics(weights,y,s,p):
    """Exact weighted ranks including ties; multinomial counts are bootstrap sample weights."""
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    starts=np.r_[0,np.flatnonzero(np.diff(ss)!=0)+1]
    ww=weights[:,order]
    pos=np.add.reduceat(ww*(yy==1),starts,axis=1)
    neg=np.add.reduceat(ww*(yy==0),starts,axis=1)
    cp=np.cumsum(pos,axis=1);cn=np.cumsum(neg,axis=1)
    nt=neg.sum(1);pt=pos.sum(1)
    # Descending order: each positive exceeds all lower-scored negatives.
    auc=(pos*(nt[:,None]-cn+.5*neg)).sum(1)/(nt*pt)
    precision=np.divide(cp,cp+cn,out=np.zeros_like(cp,dtype=float),where=cp+cn>0)
    ap=(pos*precision).sum(1)/pt
    ba=.5*((weights*((y==0)&(p==0))).sum(1)/nt+(weights*((y==1)&(p==1))).sum(1)/pt)
    return np.column_stack([auc,ap,ba])

def bootstrap(final_raw,final_points):
    base=final_raw[final_raw.endpoint=='linear-head-final'].sort_values(['target','sample_id'])
    ids=base.sample_id.tolist();y=base.target.to_numpy(int)
    eps=final_points.endpoint.tolist();arrays={}
    for ep in eps:
        g=final_raw[final_raw.endpoint==ep].set_index('sample_id').loc[ids]
        assert np.array_equal(g.target,y)
        arrays[ep]=(g.score.to_numpy(float),g.parsed.to_numpy(int))
    n0=int((y==0).sum());n1=int((y==1).sum());assert(n0,n1)==(64,478)
    out=np.empty((NBOOT,len(eps),3));rng=np.random.default_rng(RNG_SEED)
    check=[]
    for start in range(0,NBOOT,256):
        nb=min(256,NBOOT-start)
        weights=np.concatenate([rng.multinomial(n0,np.ones(n0)/n0,size=nb),rng.multinomial(n1,np.ones(n1)/n1,size=nb)],axis=1)
        for j,ep in enumerate(eps):
            s,p=arrays[ep];v=weighted_metrics(weights,y,s,p);out[start:start+nb,j]=v
            if start==0:
                for q in range(5):
                    aa=roc_auc_score(y,s,sample_weight=weights[q]);ap=average_precision_score(y,s,sample_weight=weights[q])
                    assert abs(aa-v[q,0])<1e-12 and abs(ap-v[q,1])<1e-12
                    check.append({'endpoint':ep,'replicate':q,'auc_error':float(abs(aa-v[q,0])),'ap_error':float(abs(ap-v[q,1]))})
    np.savez_compressed(DER/'bootstrap_final.npz',values=out,endpoints=np.array(eps),metrics=np.array(['auroc','average_precision','balanced_accuracy']))
    interval=[]
    for j,ep in enumerate(eps):
        row=final_points.set_index('endpoint').loc[ep].to_dict();row['endpoint']=ep
        for k,col in enumerate(['auroc','average_precision','balanced_accuracy']):
            row[col+'_ci_low'],row[col+'_ci_high']=np.quantile(out[:,j,k],[.025,.975])
        interval.append(row)
    contrasts=[]
    for seed in (43,44):
        get=lambda m:f's{seed}-{m}-step1000-final'
        for a,b,label in [(get('fixed_lce'),get('lm_only'),'Fixed LCE − LM-only'),
                          (get('cabg_lce'),get('lm_only'),'CABG − LM-only'),
                          (get('fixed_lce'),'linear-head-final','Fixed LCE − linear head'),
                          (get('cabg_lce'),get('fixed_lce'),'CABG − Fixed LCE')]:
            j,k=eps.index(a),eps.index(b);d=out[:,j,0]-out[:,k,0]
            r={'seed':seed,'comparison':label,'left':a,'right':b,
               'delta_auroc':float(final_points.set_index('endpoint').loc[a,'auroc']-final_points.set_index('endpoint').loc[b,'auroc']),
               'ci95_low':np.quantile(d,.025),'ci95_high':np.quantile(d,.975),
               'family8_ci_low':np.quantile(d,.05/(2*8)),'family8_ci_high':np.quantile(d,1-.05/(2*8)),
               'analysis_status':'post-hoc','replicates':NBOOT,'resampling_seed':RNG_SEED,
               'unit':'absolute AUROC difference','interval_scope':'conditional on fitted models; image-stratified; unknown patient clustering'}
            contrasts.append(r)
    return pd.DataFrame(interval),pd.DataFrame(contrasts),check

def main():
    t=time.time()
    TAB.mkdir(exist_ok=True);DER.mkdir(parents=True,exist_ok=True)
    (ROOT/'verification').mkdir(exist_ok=True)
    for f in ['af_verification.json','af_training_verification.json','ag_a1_verification.json','ag_a2_verification.json','ah_verification.json','ah_training_verification.json']:
        assert js(f)['status']=='PASS',(f,js(f).get('status'))
    af=read('af_per_image.csv');ah=read('ah_per_image.csv')
    afr,eaf=endpoint_reduce(af,read('af_endpoints.csv'),'af');ahr,eah=endpoint_reduce(ah,read('ah_endpoints.csv'),'ah')
    # Shared historical endpoints are references, never independent new runs.
    overlap=set(af.endpoint)&set(ah.endpoint)
    for ep in overlap:
        a=af[af.endpoint==ep].sort_values('sample_id').reset_index(drop=True)
        b=ah[ah.endpoint==ep].sort_values('sample_id').reset_index(drop=True)
        assert a.sample_id.equals(b.sample_id) and a.target.equals(b.target) and a.parsed.equals(b.parsed)
        assert np.max(abs(a.score-b.score))<1e-12
    master=pd.concat([afr,ahr[~ahr.endpoint.isin(overlap)]],ignore_index=True)
    save(master,'01_endpoint_master.csv')
    ap=pairs_reduce(af,read('af_pairs.csv'),'af');hp=pairs_reduce(ah,read('ah_pairs.csv'),'ah')
    save(ap,'02_scale_paired_original.csv');save(hp,'03_transport_paired_original.csv')
    # Counts and identity assertions use manifests only; never dereference image paths.
    cohorts=[];sets={}
    for split in ['train','development','final']:
        m=read('af_'+split+'_manifest.csv');assert not m.sample_id.duplicated().any();sets[split]=set(m.sample_id)
        for c in (0,1):
            cohorts.append({'split':split,'class':c,'label':'No listed finding' if c==0 else 'Listed finding',
                            'n':int((m.target==c).sum()),'split_total':len(m)})
        expected={'train':1000,'development':128,'final':542}[split];assert len(m)==expected
        for df in [af,ah]:
            for ep,g in df.groupby('endpoint'):
                if ident(ep)['split']==split:
                    assert set(g.sample_id)==sets[split],ep
                    joined=g.set_index('sample_id').target.sort_index();expected_y=m.set_index('sample_id').target.sort_index()
                    assert joined.equals(expected_y),ep
    assert all(not sets[a]&sets[b] for a,b in [('train','development'),('train','final'),('development','final')])
    save(pd.DataFrame(cohorts),'04_cohort_counts.csv')
    final=afr[afr.split=='final'].copy();dev=afr[afr.split=='development'].copy()
    intervals,contrasts,bc=bootstrap(af[af.endpoint.isin(final.endpoint)],final)
    save(intervals,'05_scale_final_with_posthoc_intervals.csv');save(contrasts,'06_auroc_paired_posthoc.csv')
    save(dev,'07_scale_development_all_nodes.csv')
    # Curve coordinates and per-image score distributions, not fitted thresholds.
    curves=[]
    for ep,g in af[af.endpoint.isin(final.endpoint)].groupby('endpoint'):
        fpr,tpr,_=roc_curve(g.target,g.score,drop_intermediate=False)
        pre,rec,_=precision_recall_curve(g.target,g.score)
        for x,z in zip(fpr,tpr):curves.append({'endpoint':ep,'curve':'ROC','x':x,'y':z})
        for x,z in zip(rec,pre):curves.append({'endpoint':ep,'curve':'PR','x':x,'y':z})
    save(pd.DataFrame(curves),'08_roc_pr_coordinates.csv')
    # No probability calibration interpretation: scores/logit margins are not probabilities.
    readouts=[]
    for ep,g in af.groupby('endpoint'):
        d=ident(ep)
        if d['method']=='linear_head' or d['step']!=1000:continue
        assert np.isfinite(g.yes_no_logit_gap).all()
        readouts.append({'endpoint':ep,**d,'n':len(g),'auxiliary_auroc':roc_auc_score(g.target,g.score),
                         'first_token_margin_auroc':roc_auc_score(g.target,g.yes_no_logit_gap),
                         'generated_BA':metrics(g)['balanced_accuracy'],'analysis_status':'post-hoc descriptive readout comparison'})
    save(pd.DataFrame(readouts),'09_readout_comparison_posthoc.csv')
    steps=read('af_steps.csv');traj=read('af_trajectories.csv');tr=[];smooth=[]
    for name,g in steps.groupby('trajectory',sort=True):
        g=g.sort_values('step');assert list(g.step)==list(range(1,1001))
        ref=traj.set_index('trajectory').loc[name];cap=g.actual_dynamic_cap_active.astype(str).str.lower().eq('true')
        ratio=g.auxiliary_to_lm_shared_ratio
        r={'trajectory':name,'seed':int(name[1:3]),'method':name[4:],'n_updates':len(g),
           'ratio_median':ratio.median(),'ratio_q95':ratio.quantile(.95),'ratio_max':ratio.max(),
           'coefficient_min':g.applied_coefficient.min(),'coefficient_median':g.applied_coefficient.median(),
           'coefficient_max':g.applied_coefficient.max(),'global_clipping_fraction':float((g.clip_scale<1).mean()),
           'cap_active_fraction':float(cap.mean()) if 'cabg' in name else np.nan,
           'first24_lm_loss':g.lm_loss.iloc[:24].mean(),'last24_lm_loss':g.lm_loss.iloc[-24:].mean(),
           'model_process_minutes':float(ref.model_process_seconds)/60,'peak_MiB':float(ref.sampled_peak_memory_MiB),
           'unique_train_images':int(ref.unique_train_images_seen),'image_exposures':int(ref.image_exposures)}
        for ours,theirs in [('ratio_median','auxiliary_to_lm_shared_ratio_median'),('ratio_max','auxiliary_to_lm_shared_ratio_max'),('global_clipping_fraction','global_clipping_fraction')]:
            assert abs(r[ours]-float(ref[theirs]))<1e-10
        if 'cabg' in name:
            assert r['ratio_max']<=.200001 and abs(r['cap_active_fraction']-float(ref.actual_dynamic_cap_fraction))<1e-12
        tr.append(r)
        gg=g.copy()
        for c in ['lm_loss','class_balanced_lce_loss','auxiliary_to_lm_shared_ratio']:
            gg[c+'_median50']=g[c].rolling(50,min_periods=1).median()
        smooth.append(gg)
    save(pd.DataFrame(tr),'10_training_control_and_resources.csv');save(pd.concat(smooth),'11_training_steps_with_median50.csv')
    # Interventions: accepted A1 summary is backed by the separate recorded PASS above.
    a1=js('ag_a1_summary.json');a2=js('ag_a2_verification.json');mech=[]
    for seed in (43,44):
        for cond in ['center_rms','softsign_rms','softsign_shuffle_rms','ano_swap','visual_swap']:
            end=a1['endpoints'][f's{seed}-{cond}'];pair=a1['comparisons'][f's{seed}-{cond}__minus__native']
            mech.append({'stage':'A1','seed':seed,'condition':cond,'n':128,'answers_changed':end['answers_changed'],
                         'corrected':pair['left_only_correct'],'harmed':pair['right_only_correct'],
                         'BA':end['balanced_accuracy'],'delta_BA':pair['delta_balanced_accuracy'],
                         'ci_low':pair['paired_balanced_accuracy_95ci'][0],'ci_high':pair['paired_balanced_accuracy_95ci'][1],
                         'median_abs_logit_change':end.get('median_absolute_logit_change'),
                         'median_fed_ano_relative_change':end.get('median_relative_change',{}).get('fed_ano')})
        pair=a2['comparisons'][f's{seed}'];end=a2['endpoints'][f's{seed}-prior_odds']
        mech.append({'stage':'A2','seed':seed,'condition':'prior_odds','n':128,'answers_changed':pair['answers_changed'],
                     'corrected':pair['left_only_correct'],'harmed':pair['right_only_correct'],'BA':end['balanced_accuracy'],
                     'delta_BA':pair['delta_balanced_accuracy'],'ci_low':pair['paired_balanced_accuracy_95ci'][0],
                     'ci_high':pair['paired_balanced_accuracy_95ci'][1]})
    save(pd.DataFrame(mech),'12_intervention_effects.csv')
    agp=read('ag_pairs.csv');assert len(agp)==14
    # Cross-check all published mechanism pair entries against original accepted summaries.
    for _,r in agp.iterrows():
        source=a1['comparisons'][r.comparison] if r.stage=='A1' else a2['comparisons'][r.comparison[:3]]
        for c in ['n','left_only_correct','right_only_correct','both_correct','both_wrong']:assert int(r[c])==source[c]
        assert abs(r.delta_BA-source['delta_balanced_accuracy'])<1e-12
    traces=js('ah_traces.json');trs=[];change=[];ver=js('ah_verification.json')
    for seed in (43,44):
        d=traces['per_seed'][str(seed)]
        for case in d['full_trace_cases']:trs.append({'seed':seed,**case})
        change.append({'seed':seed,'n':128,'ano_changed':ver['development_inference_shuffle'][str(seed)]['ano_changed'],
                       'logit_gap_changed':d['all_128_nonzero_logit_gap_changes'],
                       'answers_changed':d['all_128_generated_answer_changes'],'trace_subset_n':8})
    save(pd.DataFrame(trs),'13_transport_trace_subset_posthoc.csv');save(pd.DataFrame(change),'14_transport_inference_changes.csv')
    save(ahr[ahr.method.isin(['aligned','shuffled'])],'15_transport_endpoints.csv')
    # Confusion matrices and response disagreements are exact saved-answer reductions.
    conf=[]
    for _,r in final.iterrows():
        for y in (0,1):
            for p in (0,1):
                count=int(r[{(0,0):'tn',(0,1):'fp',(1,0):'fn',(1,1):'tp'}[(y,p)]])
                conf.append({'endpoint':r.endpoint,'seed':r.seed,'method':r.method,'target':y,'prediction':p,
                             'count':count,'class_fraction':count/int(r.negative_n if y==0 else r.positive_n)})
    save(pd.DataFrame(conf),'16_final_confusion_counts.csv')
    # Per-image records below are small numeric exports, never source images.
    save(af[af.endpoint.isin(final.endpoint)].copy(),'17_final_score_distributions.csv')
    # Main allowed-claim statistics for the interpretation guide.
    wins=0
    for seed in (43,44):
        for step in (24,100,300,1000):
            lookup=afr.set_index('endpoint')
            wins+=int(lookup.loc[f's{seed}-fixed_lce-step{step:04d}-development','auroc']>lookup.loc[f's{seed}-lm_only-step{step:04d}-development','auroc'])
    report={'status':'PASS','scope':'saved small exports; no model calls; same-operator analysis',
            'af_rows':len(af),'ah_rows':len(ah),'af_endpoints':len(afr),'ah_endpoints':len(ahr),
            'shared_historical_endpoints':len(overlap),'unique_endpoints_in_master':len(master),
            'af_pair_checks':len(ap),'ah_pair_checks':len(hp),'ag_pair_checks':len(agp),
            'maximum_endpoint_error':max(eaf,eah),'train_steps':len(steps),
            'unique_cohort_images':sum(len(s) for s in sets.values()),'development_AUROC_wins_of8':wins,
            'bootstrap_replicates':NBOOT,'bootstrap_seed':RNG_SEED,'weighted_metric_checks':bc,
            'analysis_seconds':time.time()-t,'original_BA_primary_endpoint_preserved':True,
            'posthoc_AUROC_family_size':8,'patient_independence_claim':False}
    writejson('verification/analysis-checks.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='weighted_metric_checks'},indent=2))
    print(contrasts[['seed','comparison','delta_auroc','ci95_low','ci95_high','family8_ci_low','family8_ci_high']].to_string(index=False))

if __name__=='__main__':main()
