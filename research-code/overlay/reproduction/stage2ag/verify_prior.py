"""Independent record/token/decision check; no model inference or producer imports."""
import argparse,json,math
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from reproduction.stage2ag.verify_diagnostics import parse,metric

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tokenizer=Tokenizer.from_file('/home/checkpoints/Lingshu-7B/tokenizer.json');endpoints={};pairs={};n=0
    for seed in [43,44]:
        folder=a.root/'diagnostics'/f'seed{seed}'/f'prior-s{seed}-v1'
        rows=json.loads((folder/'observations.json').read_text());prod=json.loads((folder/'summary.json').read_text())
        old=json.loads((a.root/'diagnostics'/f'seed{seed}'/f'diagnostic-s{seed}-v1/observations.json').read_text())
        old={r['sample_id']:r for r in old if r['condition']=='native'}
        groups={c:[r for r in rows if r['condition']==c] for c in ['native','prior_odds']}
        native={r['sample_id']:r for r in groups['native']}
        assert len(rows)==256
        for c,rs in groups.items():
            assert len(rs)==128 and len({r['sample_id'] for r in rs})==128
            for r in rs:
                b=old[r['sample_id']]
                for k in ['sample_id','target','split','sha256','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids','score']:
                    assert r[k]==b[k]
                assert tokenizer.decode(r['output_ids'],skip_special_tokens=True)==r['text'] and parse(r['text'])==r['parsed']
                calls=r['processor_calls'];assert len(calls)==len(r['output_ids']) and [v['step'] for v in calls]==list(range(len(calls)))
                offset=0 if c=='native' else math.log(2);assert r['offset']==offset
                first=calls[0];before=first['before_yes_no'];after=first['after_yes_no']
                assert before==native[r['sample_id']]['processor_calls'][0]['before_yes_no']
                assert abs((before[0]-after[0])-offset)<2e-6 and before[1]==after[1]
                assert first['changed_indices']==([] if c=='native' else [[0,r['yes_token_id']]])
                assert all(v['changed_indices']==[] for v in calls[1:])
                if c=='native':assert r['output_ids']==b['output_ids']
                # This check does not replace generation; it checks observed binary choices against their logits.
                if r['output_ids'][0] in [r['yes_token_id'],r['no_token_id']] and after[0]!=after[1]:
                    assert (r['output_ids'][0]==r['yes_token_id'])==(after[0]>after[1])
                n+=1
            m=metric(rs);reported=prod['endpoints'][c]
            assert all(abs(v-reported[k])<1e-12 for k,v in m.items())
            for label in [0,1]:
                subset=[r for r in rs if r['target']==label];b=reported['by_class'][str(label)]
                assert len(subset)==b['n']==64 and sum(parse(r['text'])==label for r in subset)==b['correct']
            endpoints[f's{seed}-{c}']=reported
        left,right=groups['prior_odds'],groups['native'];assert [r['sample_id'] for r in left]==[r['sample_id'] for r in right]
        y=np.array([r['target'] for r in left]);la=np.array([parse(r['text'])==r['target'] for r in left]);ra=np.array([parse(r['text'])==r['target'] for r in right]);d=la.astype(int)-ra.astype(int)
        v=prod['comparison'];assert v['left_only_correct']==int(sum(la&~ra)) and v['right_only_correct']==int(sum(ra&~la))
        delta=.5*(d[y==0].mean()+d[y==1].mean());assert abs(delta-v['delta_balanced_accuracy'])<1e-12
        rng=np.random.default_rng(seed);boot=0
        for c in [0,1]:
            vals,counts=np.unique(d[y==c],return_counts=True);boot=boot+(rng.multinomial(64,counts/64,size=100000)@vals)/128
        ci=np.quantile(boot,[.025,.975]);assert np.max(np.abs(ci-v['paired_balanced_accuracy_95ci']))<=2/128
        pairs[f's{seed}']=dict(v,answers_changed=sum(x['output_ids']!=z['output_ids'] for x,z in zip(left,right)),independent_100000_multinomial_ci=ci.tolist())
    report=dict(status='PASS',records=n,scope='separate raw token decoding, sample identities, intervention offsets, counts, ranking and paired intervals; same operator',endpoints=endpoints,comparisons=pairs)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',records=n,comparisons=pairs)))

if __name__=='__main__':main()
