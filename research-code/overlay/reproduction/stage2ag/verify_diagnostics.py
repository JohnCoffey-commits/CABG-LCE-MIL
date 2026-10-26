"""Independent NumPy/standard-library reconstruction; no producer maths imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import rankdata

def bf16(x):
    v=np.asarray(x,dtype=np.float32).copy();u=v.view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def parse(s):
    s=s.strip().lower()
    if s.endswith(('.','!')):s=s[:-1]
    return {'yes':1,'no':0}.get(s)

def metric(rows):
    y=np.array([r['target'] for r in rows]);x=np.array([r['score'] for r in rows]);pred=np.array([-1 if parse(r['text']) is None else parse(r['text']) for r in rows])
    c=pred==y;n1=int(sum(y));n0=len(y)-n1
    au=(rankdata(x)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    ap=sum(sum((y==1)&(x>=t))/sum(x>=t) for t in x[y==1])/n1
    return dict(n=len(y),correct=int(sum(c)),accuracy=float(c.mean()),balanced_accuracy=float((c[y==0].mean()+c[y==1].mean())/2),invalid=int(sum(pred<0)),auroc=float(au),average_precision=float(ap))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True)
    p.add_argument('--attempts',nargs=2,required=True);p.add_argument('--reduction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('/home/checkpoints/Lingshu-7B/tokenizer.json')
    prod=json.loads(a.reduction.read_text());maximum={};count=0
    def error(name,value,limit):
        maximum[name]=max(maximum.get(name,0),float(value));assert value<=limit,(name,value,limit)
    for seed,attempt in zip([43,44],a.attempts):
        out=a.root/'diagnostics'/f'seed{seed}'/attempt;rows=json.loads((out/'observations.json').read_text())
        assert len(rows)==768
        groups={c:[r for r in rows if r['condition']==c] for c in ['native','center_rms','softsign_rms','softsign_shuffle_rms','ano_swap','visual_swap']}
        original={r['sample_id']:r for r in groups['native']};native={}
        oldroot=a.old_root/'evaluation'/f's{seed}-fixed_lce-step1000-development'/'v1'
        old={r['sample_id']:r for r in json.loads((oldroot/'observations.json').read_text())}
        for row in groups['native']:
            sid=row['sample_id'];assert row['output_ids']==old[sid]['output_ids']
            for k in ['sample_id','target','sha256','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']:
                assert row[k]==old[sid][k]
            with np.load(out/row['trace_file'],allow_pickle=False) as z:native[sid]={k:z[k] for k in z.files}
            with np.load(oldroot/old[sid]['raw_file'],allow_pickle=False) as z:
                assert np.array_equal(native[sid]['a'].squeeze(2),z['abnormal']) and np.array_equal(native[sid]['n'].squeeze(2),z['normal'])
        order=sorted(original,key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        donors={sid:order[(i+1)%len(order)] for i,sid in enumerate(order)}
        assert donors==json.loads((out/'donors.json').read_text())
        noops=json.loads((out/'no-op-checks.json').read_text())
        assert len(noops)==4 and {r['sample_id'] for r in noops}==set(order[:4])
        for r in noops:
            b=original[r['sample_id']]
            assert r['output_ids']==b['output_ids'] and r['yes_no_logits']==b['yes_no_logits']
            assert all(v==0 for v in r['relative_change'].values())
        for cond,rs in groups.items():
            assert len(rs)==128 and len({r['sample_id'] for r in rs})==128 and {r['split'] for r in rs}=={'development'}
            for row in rs:
                b=original[row['sample_id']]
                for k in ['sample_id','target','sha256','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']:assert row[k]==b[k]
                assert tokenizer.decode(row['output_ids'],skip_special_tokens=True)==row['text']
                assert parse(row['text'])==row['parsed']
                path=out/row['trace_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['trace_sha256']
                with np.load(path,allow_pickle=False) as z:t={k:z[k] for k in z.files}
                assert all(np.isfinite(v).all() for v in t.values())
                assert np.array_equal(t['a'],native[row['sample_id']]['a']) and np.array_equal(t['n'],native[row['sample_id']]['n'])
                aa,nn=t['a'].astype('float64'),t['n'].astype('float64');d=aa-nn
                # Torch sigmoid/subtraction use BF16 materialization at both operations.
                e=bf16(bf16(expit(aa))-bf16(expit(nn)))
                error('original_bf16_e',np.max(np.abs(e-t['original_e'])),.0078125)
                if cond in ['center_rms','softsign_rms','softsign_shuffle_rms']:
                    x=np.tanh(d/4) if cond=='center_rms' else d/(1+np.abs(d))
                    x*=np.sqrt(np.mean(t['original_e'].astype('float64')**2,-1,keepdims=True))/np.maximum(np.sqrt(np.mean(x*x,-1,keepdims=True)),1e-30)
                    if cond=='softsign_shuffle_rms':x=x[...,(np.arange(1024)*405+137)%1024]
                    # Single BF16 ULP permits library/reduction last-bit differences; no empirical tuning.
                    delta=np.abs(bf16(x)-t['replacement_e']);allowed=np.maximum(np.abs(x)/128,1e-6)
                    error('map_ulp_ratio',np.max(delta/allowed),1.01)
                else:assert np.array_equal(t['original_e'],t['replacement_e'])
                m=(d/(1+np.abs(d))).mean(axis=1).ravel();tau=1/np.log(1024)
                score=tau*(logsumexp(m/tau)-np.log(1024));w=np.exp(m/tau-logsumexp(m/tau))
                error('score',abs(score-row['score']),2e-6)
                error('effective_support',abs(np.exp(-np.sum(w*np.log(np.maximum(w,1e-300))))-row['effective_support']),.002)
                error('top11_mass',abs(np.sort(w)[-11:].sum()-row['top11_mass']),2e-6)
                if cond!='native':
                    for k,v in row['relative_change'].items():
                        base=native[row['sample_id']][k];value=np.linalg.norm(t[k].astype('float64')-base)/max(np.linalg.norm(base),1e-30)
                        error('relative_change',abs(value-v),1e-8)
                donor=donors[row['sample_id']]
                want_ano=native[donor]['fed_ano'] if cond=='ano_swap' else t['post_ffn'].reshape(16,-1)
                assert np.array_equal(t['fed_ano'],want_ano)
                want_direct=native[donor]['direct'] if cond=='visual_swap' else native[row['sample_id']]['direct']
                # Producer hashes BF16 bytes, independently extract high16 bits from stored float32.
                raw=(want_direct.astype(np.float32).view(np.uint32)>>16).astype(np.uint16).tobytes()
                assert hashlib.sha256(raw).hexdigest()==row['fed_direct_sha256']
                count+=1
            calculated=metric(rs);reported=prod['endpoints'][f's{seed}-{cond}']
            for k,v in calculated.items():error('metric_'+k,abs(v-reported[k]),1e-12)
            for label in [0,1]:
                subset=[r for r in rs if r['target']==label];by=reported['by_class'][str(label)]
                assert by['n']==len(subset)==64
                assert by['correct']==sum(parse(r['text'])==label for r in subset)
                assert by['invalid']==sum(parse(r['text']) is None for r in subset)
        for name,v in prod['comparisons'].items():
            if not name.startswith(f's{seed}-'):continue
            left,right=name[len(f's{seed}-'):].split('__minus__');ls,rr=groups[left],groups[right]
            y=np.array([r['target'] for r in ls]);ac=np.array([parse(r['text'])==r['target'] for r in ls]);bc=np.array([parse(r['text'])==r['target'] for r in rr]);delta=ac.astype(int)-bc.astype(int)
            assert int(sum(ac&~bc))==v['left_only_correct'] and int(sum(bc&~ac))==v['right_only_correct']
            assert int(sum(ac&bc))==v['both_correct'] and int(sum(~ac&~bc))==v['both_wrong']
            error('paired_BA',abs(.5*(delta[y==0].mean()+delta[y==1].mean())-v['delta_balanced_accuracy']),1e-12)
            rng=np.random.default_rng(seed);boot=0
            for c in [0,1]:
                vals,counts=np.unique(delta[y==c],return_counts=True)
                draws=rng.multinomial(64,counts/64,size=100000)
                boot=boot+(draws@vals)/(2*64)
            ci=np.quantile(boot,[.025,.975]);error('paired_CI',max(abs(ci-np.array(v['paired_balanced_accuracy_95ci']))),2/128)
    result=dict(status='PASS',scope='Independent NumPy raw/map/branch/metric/count/interval implementation, same operator; no new model calls',records=count,maximum_errors=maximum)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
