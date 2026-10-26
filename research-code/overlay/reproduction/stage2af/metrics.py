"""Producer metrics: answers and anomaly ranking are intentionally separate."""
import math,re
import numpy as np

def parse_answer(text):
    match=re.fullmatch(r'(yes|no)[.!]?',text.strip().lower())
    return None if match is None else int(match.group(1)=='yes')

def score_maps(abnormal,normal):
    import torch
    assert abnormal.shape==normal.shape==(1,4,1024)
    gap=abnormal.float()-normal.float();assert torch.isfinite(gap).all()
    c=(gap/(1+gap.abs())).mean(1);tau=1/math.log(1024)
    score=tau*(torch.logsumexp(c/tau,-1)-math.log(1024));w=torch.softmax(c/tau,-1)
    support=(-(w*w.clamp_min(torch.finfo(w.dtype).tiny).log()).sum(-1)).exp()
    return {'score':score.item(),'effective_support':support.item(),'top11_mass':torch.topk(w,11,-1).values.sum(-1).item()}

def summary(rows):
    y=np.array([r['target'] for r in rows]);s=np.array([r['score'] for r in rows],dtype=np.float64)
    pred=np.array([-1 if r['parsed'] is None else r['parsed'] for r in rows]);assert set(y)=={0,1} and np.isfinite(s).all()
    correct=pred==y;p=s[y==1];n=s[y==0]
    auc=float(((p[:,None]>n)+.5*(p[:,None]==n)).mean())
    ap=float(np.mean([sum(p>=x)/sum(s>=x) for x in p]))
    by_class={str(k):{'n':int(sum(y==k)),'correct':int(sum(correct&(y==k))),'correct_rate':float(correct[y==k].mean()),'invalid':int(sum((pred<0)&(y==k)))} for k in [0,1]}
    result={'n':len(rows),'accuracy':float(correct.mean()),'balanced_accuracy':sum(v['correct_rate'] for v in by_class.values())/2,
            'correct':int(sum(correct)),'invalid':int(sum(pred<0)),'by_class':by_class,'auroc':auc,'average_precision':ap}
    for key in ['effective_support','top11_mass','yes_no_logit_gap']:
        if key in rows[0]:
            values=np.array([r[key] for r in rows]);result[key]={'median':float(np.median(values)),'mean':float(values.mean()),'min':float(values.min()),'max':float(values.max())}
    return result

def paired(left,right,bootstrap_seed=20260918,repeats=10000):
    assert [(r['sample_id'],r['target'],r['pixel_sha256_prepared']) for r in left]==[(r['sample_id'],r['target'],r['pixel_sha256_prepared']) for r in right]
    y=np.array([r['target'] for r in left]);a=np.array([r['parsed']==r['target'] for r in left]);b=np.array([r['parsed']==r['target'] for r in right]);delta=a.astype(float)-b
    rng=np.random.default_rng(bootstrap_seed);samples=np.zeros(repeats)
    for c in [0,1]:
        values=delta[y==c];indices=rng.integers(0,len(values),size=(repeats,len(values)));samples+=values[indices].mean(1)/2
    sa=summary(left);sb=summary(right)
    result={'n':len(left),'left_only_correct':int(sum(a&~b)),'right_only_correct':int(sum(b&~a)),
            'both_correct':int(sum(a&b)),'both_wrong':int(sum(~a&~b)),
            'delta_balanced_accuracy':sa['balanced_accuracy']-sb['balanced_accuracy'],
            'delta_accuracy':sa['accuracy']-sb['accuracy'],'delta_auroc':sa['auroc']-sb['auroc'],
            'delta_average_precision':sa['average_precision']-sb['average_precision'],
            'paired_balanced_accuracy_95ci':np.quantile(samples,[.025,.975]).tolist(),
            'uncertainty_scope':'10000 paired class-stratified image bootstrap; repeated patient identity and base-training overlap unknown; descriptive, unadjusted intervals'}
    if not np.any(delta):
        result['bootstrap_degenerate_zero_discordance']=True
        result['zero_discordance_absolute_BA_bound']=float(sum(1-.025**(1/sum(y==c)) for c in [0,1])/2)
        result['zero_discordance_note']='A degenerate empirical bootstrap is not proof of zero population difference. Bound uses per-class 97.5% exact upper probability of unseen discordance, combined conservatively.'
    return result
