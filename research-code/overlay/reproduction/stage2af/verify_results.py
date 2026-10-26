"""Independent raw-score, decoded-answer, ranking and paired-count verification."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(rows):
    # Rank-sum AUROC and grouped threshold AP, not producer pairwise broadcast.
    ordered=sorted(rows,key=lambda x:x['score']);n1=sum(r['target'] for r in rows);n0=len(rows)-n1
    assert n1>0 and n0>0;rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j]['score']==ordered[i]['score']:j+=1
        rank_sum+=sum(r['target'] for r in ordered[i:j])*(i+1+j)/2;i=j
    auc=(rank_sum-n1*(n1+1)/2)/(n1*n0)
    reverse=list(reversed(ordered));tp=0;ap=0.;i=0
    while i<len(reverse):
        j=i+1
        while j<len(reverse) and reverse[j]['score']==reverse[i]['score']:j+=1
        positives=sum(r['target'] for r in reverse[i:j]);tp+=positives;ap+=positives/n1*tp/j;i=j
    correct={y:sum(r['parsed'] is not None and r['parsed']==y for r in rows if r['target']==y) for y in [0,1]}
    return {'n':len(rows),'correct':sum(correct.values()),'accuracy':sum(correct.values())/len(rows),
            'balanced_accuracy':.5*(correct[0]/n0+correct[1]/n1),'invalid':sum(r['parsed'] is None for r in rows),
            'auroc':auc,'average_precision':ap}

def reconstruct(a,n):
    assert a.shape==n.shape==(1,4,1024)
    gap=a.astype(np.float32)-n.astype(np.float32);assert np.isfinite(gap).all()
    c=np.mean(gap/(1+abs(gap)),axis=1,dtype=np.float32).astype(float).ravel()
    z=c*math.log(1024);m=float(z.max());e=np.exp(z-m);weights=e/e.sum()
    return {'score':(m+math.log(e.sum())-math.log(1024))/math.log(1024),
            'effective_support':math.exp(float(-sum(weights*np.log(weights)))),'top11_mass':float(sum(sorted(weights)[-11:]))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    plan=read(root/'protocol.json');protocol=sha(root/'protocol.json');assert read(root/'protocol-seal.json')['sha256']==protocol
    assert plan['raw_tolerances']=={'score':2e-6,'effective_support':.002,'top11_mass':2e-6}
    inputs={r['sample_id']:r for r in [json.loads(s) for s in (root/'data/prepared-v1/manifest.jsonl').read_text().splitlines()]}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/home/checkpoints/Lingshu-7B',local_files_only=True)
    accepted=read(root/'accepted-endpoints.json');observed={};results={};errors={k:0. for k in plan['raw_tolerances']};count=0
    for endpoint,path in accepted.items():
        out=Path(path);report=read(out/'summary.json');assert report['status']=='PASS' and report['protocol_sha256']==protocol
        rows=read(out/'observations.json');assert len({r['sample_id'] for r in rows})==len(rows)
        if not endpoint.startswith('linear-head'):
            assert read(out/'parameters-before.json')==read(out/'parameters-after.json')
            cp=read(out/'checkpoint.json');assert sha(cp['path'])==cp['sha256']
            import torch
            payload=torch.load(cp['path'],map_location='cpu',weights_only=False)
            parameters={r['name']:r for r in read(out/'parameters-before.json')['parameters']}
            for name,tensor in payload['adapter'].items():
                raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                assert hashlib.sha256(raw).hexdigest()==parameters[name]['sha256']
            del payload
            split=rows[0]['split'];expected=[x['sample_id'] for x in inputs.values() if x['split']==split]
            assert [r['sample_id'] for r in rows]==expected
        for i,r in enumerate(rows):
            source=inputs[r['sample_id']]
            for key in ['target','split','group_id','sha256','pixel_sha256_prepared','image_grid_thw','input_ids','prepared_sha256']:assert r[key]==source[key],(endpoint,key)
            if not endpoint.startswith('linear-head'):
                assert r==read(out/f'record-{i:04}.json')
                assert 0<len(r['output_ids'])<=16
                text=tok.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False);assert text==r['text']
                clean=text.strip().lower();valid={'yes':1,'yes.':1,'yes!':1,'no':0,'no.':0,'no!':0};assert r['parsed']==valid.get(clean)
                assert sha(out/r['raw_file'])==r['raw_sha256']
                with np.load(out/r['raw_file'],allow_pickle=False) as raw:
                    computed=reconstruct(raw['abnormal'],raw['normal']);assert abs(float(raw['yes_no_logits'][0]-raw['yes_no_logits'][1])-r['yes_no_logit_gap'])<2e-6
                for key,value in computed.items():
                    error=abs(value-r[key]);errors[key]=max(errors[key],error);assert error<plan['raw_tolerances'][key],(endpoint,i,key,error)
                count+=1
        if endpoint.startswith('linear-head'):
            f=np.load(out/'features.npy',allow_pickle=False);assert f.shape==(len(rows),3584) and sha(out/'features.npy')==report['features_sha256']
            hp=out/'head.json' if (out/'head.json').is_file() else Path(accepted['linear-head-fit'])/'head.json';head=read(hp)
            mean=np.array(head['mean']);scale=np.array(head['scale']);w=np.array(head['weight']);b=head['bias']
            scores=(f.astype(float)-mean)/scale@w+b
            np.testing.assert_allclose(scores,[r['score'] for r in rows],atol=1e-10,rtol=1e-10)
            assert [int(s>=0) for s in scores]==[r['parsed'] for r in rows]
            if (out/'head.json').is_file():
                train=np.array([r['split']=='train' for r in rows]);x=f[train].astype(float);y=np.array([r['target'] for r in rows])[train];assert len(x)==1000
                mu=x.mean(0);sigma=x.std(0);sigma[sigma<1e-6]=1
                np.testing.assert_array_equal(mu,mean);np.testing.assert_array_equal(sigma,scale)
                z=(x-mean)/scale;s=z@w+b;prob=np.exp(-np.logaddexp(0,-s));weights=np.where(y==1,.5/sum(y==1),.5/sum(y==0))
                grad=z.T@(weights*(prob-y))+w/len(y);bias_grad=sum(weights*(prob-y));assert max(np.max(abs(grad)),abs(bias_grad))<=1e-6
            for split in sorted({r['split'] for r in rows}):
                key='linear-head-'+split;part=[r for r in rows if r['split']==split];observed[key]=part;results[key]=metrics(part)
                for k,v in results[key].items():assert abs(report['metrics'][split][k]-v)<1e-12
        else:
            observed[endpoint]=rows;results[endpoint]=metrics(rows)
            for k,v in results[endpoint].items():assert abs(report['metrics'][k]-v)<1e-12
    table=read(root/'results-v1/endpoint-summary.json');assert {r['endpoint'] for r in table}==set(results)
    for row in table:
        for k,v in results[row['endpoint']].items():assert abs(row[k]-v)<1e-12
    pairs=read(root/'results-v1/paired-comparisons.json')
    for name,report in pairs.items():
        left,right=name.split('__minus__');a,b=observed[left],observed[right]
        assert [x['sample_id'] for x in a]==[x['sample_id'] for x in b]
        categories={'left_only_correct':0,'right_only_correct':0,'both_correct':0,'both_wrong':0}
        for x,y in zip(a,b):
            ac=x['parsed']==x['target'];bc=y['parsed']==y['target'];key='both_correct' if ac and bc else 'left_only_correct' if ac else 'right_only_correct' if bc else 'both_wrong';categories[key]+=1
        for k,v in categories.items():assert report[k]==v
        for k in ['balanced_accuracy','accuracy','auroc','average_precision']:assert abs(report['delta_'+k]-(results[left][k]-results[right][k]))<1e-12
        # Independent multinomial bootstrap of each empirical {-1,0,1} paired
        # distribution checks the producer's resampling CI within Monte Carlo
        # resolution; both recipes and the tolerance are frozen beforehand.
        rng=np.random.default_rng(20260919);samples=np.zeros(100000)
        for target in [0,1]:
            values=[int(x['parsed']==x['target'])-int(y['parsed']==y['target']) for x,y in zip(a,b) if x['target']==target];n=len(values)
            probabilities=[values.count(v)/n for v in [-1,0,1]];draws=rng.multinomial(n,probabilities,size=len(samples));samples+=(draws[:,2]-draws[:,0])/(2*n)
        independent=np.quantile(samples,[.025,.975]);reported=np.array(report['paired_balanced_accuracy_95ci'])
        resolution=sum(1/sum(x['target']==c for x in a) for c in [0,1])/2
        assert np.max(abs(independent-reported))<=max(.005,2*resolution),'Bootstrap interval beyond declared Monte Carlo tolerance'
        if categories['left_only_correct']+categories['right_only_correct']==0:
            assert report['bootstrap_degenerate_zero_discordance']
            bound=sum(1-.025**(1/sum(x['target']==c for x in a)) for c in [0,1])/2
            assert abs(bound-report['zero_discordance_absolute_BA_bound'])<1e-12
    output={'status':'PASS','scope':'independent NumPy/standard-library metric and paired-count reconstruction; same operator, not independent lab/patients',
            'verified_raw_records':count,'verified_endpoints':len(results),'paired_comparisons':len(pairs),'maximum_raw_errors':errors,'protocol_sha256':protocol}
    with (root/'results-v1/independent-verification.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output),flush=True)
if __name__=='__main__':main()
