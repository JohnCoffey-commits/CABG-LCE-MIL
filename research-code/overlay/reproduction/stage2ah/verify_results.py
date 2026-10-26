"""Independent NumPy reconstruction; never imports the transport implementation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bf16(array):
    x = np.array(array, dtype=np.float32, copy=True)
    bits = x.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.view(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    assert Path('/home/Medic-AD').exists(), 'Server only tokenizer and raw arrays'
    from transformers import AutoTokenizer
    # Independent prior verifier routines use rank sums/grouped AP and a separate
    # softsign/LSE implementation, not production summary or transport functions.
    from reproduction.stage2af.verify_results import metrics, reconstruct
    tok = AutoTokenizer.from_pretrained('/home/checkpoints/Lingshu-7B', local_files_only=True)
    root = a.root
    old = Path('/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1')
    inputs = {r['sample_id']:r for r in [json.loads(s) for s in (old/'data/prepared-v1/manifest.jsonl').read_text().splitlines()]}
    accepted = read(root/'accepted-endpoints.json')
    table = {r['endpoint']:r for r in read(root/'results-v1/endpoint-summary.json')}
    allrows = {}
    errors = dict(score=0.,effective_support=0.,top11_mass=0.,weight=0.,pooled_relative=0.,pooled_max=0.)
    count = full = 0
    for endpoint, entry in accepted.items():
        out = Path(entry['path'])
        rows = read(out/'observations.json')
        if entry.get('subset'):
            rows = [r for r in rows if r['split']==entry['subset']]
        allrows[endpoint] = rows
        if entry['origin'] == 'stage2ah':
            import torch
            meta = read(out/'checkpoint.json')
            # Streaming SHA avoids loading serialized checkpoints into the local client.
            digest = hashlib.sha256()
            with Path(meta['path']).open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''): digest.update(block)
            assert digest.hexdigest() == meta['sha256']
            payload = torch.load(meta['path'],map_location='cpu',weights_only=False)
            assert payload['protocol_sha256'] == sha(root/'protocol.json')
            parameters = {p['name']:p for p in read(out/'parameters-before.json')['parameters']}
            for name,tensor in payload['adapter'].items():
                raw=tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                assert hashlib.sha256(raw).hexdigest()==parameters[name]['sha256']
            del payload
        assert len({r['sample_id'] for r in rows}) == len(rows)
        split = rows[0]['split']
        expected = [r['sample_id'] for r in inputs.values() if r['split']==split]
        assert [r['sample_id'] for r in rows] == expected
        for i, r in enumerate(rows):
            source = inputs[r['sample_id']]
            for key in ['target','sha256','split','group_id','prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']:
                assert r[key] == source[key], (endpoint,key)
            if endpoint.startswith('linear-head'): continue
            text = tok.decode(r['output_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            assert text == r['text'] and 0 < len(r['output_ids']) <= 16
            assert r['parsed'] == {'yes':1,'yes.':1,'yes!':1,'no':0,'no.':0,'no!':0}.get(text.strip().lower())
            if entry['origin'] != 'stage2ah': continue
            assert r == read(out/f'record-{i:04}.json')
            assert sha(out/r['raw_file']) == r['raw_sha256']
            with np.load(out/r['raw_file'],allow_pickle=False) as z:
                computed = reconstruct(z['abnormal'],z['normal'])
                for key, tolerance in [('score',2e-6),('effective_support',.002),('top11_mass',2e-6)]:
                    error = abs(r[key]-computed[key]);errors[key]=max(errors[key],error)
                    assert error < tolerance, (endpoint,i,key,error)
                gap = z['abnormal'].astype(float)-z['normal'].astype(float)
                c = (gap/(1+abs(gap))).mean(axis=1).ravel()
                weight = np.exp((c-c.max())*math.log(1024));weight/=weight.sum()
                assert np.max(abs(weight-z['weight32'].ravel())) < 2e-6
                small = np.zeros((16,16))
                for y in range(32):
                    for x in range(32): small[y//2,x//2] += weight[y*32+x]
                small = small.ravel()
                if r['transport_arm']=='shuffled': small = small[(np.arange(256)*101+37)%256]
                error = float(np.max(abs(small-z['weight16'].ravel())))
                errors['weight']=max(errors['weight'],error)
                assert error < 2e-6 and abs(small.sum()-1)<1e-12
                assert abs(r['transported_mass']-1)<2e-6
                if r['full_transport_trace']:
                    # Reconstruct BF16 input matmul with float64 accumulation,
                    # then RNE quantization; tolerance covers device accumulation.
                    pooled = bf16(bf16(z['weight16']).astype(float) @ z['values'][0].astype(float))[:,None,:]
                    actual = z['pooled'];maximum=float(np.max(abs(actual-pooled)))
                    relative=float(np.linalg.norm(actual-pooled)/max(np.linalg.norm(pooled),1e-30))
                    errors['pooled_relative']=max(errors['pooled_relative'],relative)
                    errors['pooled_max']=max(errors['pooled_max'],maximum)
                    assert relative<=.01 and maximum<=.25,(endpoint,i,relative,maximum)
                    mixed=bf16((z['native'].astype(np.float32)+actual.astype(np.float32))*.5)
                    assert np.array_equal(mixed,z['output'])
                    assert np.array_equal(z['post_ffn'].reshape(16,3584),z['fed_ano'])
                    assert hashlib.sha256(z['fed_ano'].tobytes()).hexdigest()==r['fed_ano_sha256']
                    assert hashlib.sha256(z['direct'].tobytes()).hexdigest()==r['fed_direct_sha256']
                    full+=1
            count+=1
        measured=metrics(rows)
        for key,value in measured.items(): assert abs(table[endpoint][key]-value)<1e-12,(endpoint,key)
        if entry['origin']=='stage2ah':
            assert read(out/'parameters-before.json')==read(out/'parameters-after.json')
            assert read(out/'file-open-audit.json')['violations']==[]
            summary=read(out/'summary.json')
            assert summary['protocol_sha256']==sha(root/'protocol.json')
            for key,value in measured.items(): assert abs(summary['metrics'][key]-value)<1e-12
    pairs=read(root/'results-v1/paired-comparisons.json')
    intervention={}
    for seed in [43,44]:
        native=allrows[f's{seed}-aligned-step1000-development']
        sham=allrows[f's{seed}-aligned-step1000-development-inference_shuffle']
        assert [r['sample_id'] for r in native]==[r['sample_id'] for r in sham]
        assert all(x['fed_direct_sha256']==y['fed_direct_sha256'] and x['score']==y['score'] for x,y in zip(native,sham))
        intervention[str(seed)]=dict(direct_tokens_and_auxiliary_score_exact=True,
             ano_changed=sum(x['fed_ano_sha256']!=y['fed_ano_sha256'] for x,y in zip(native,sham)),
             generated_answer_changed=sum(x['parsed']!=y['parsed'] for x,y in zip(native,sham)))
    for name, report in pairs.items():
        left,right=name.split('__minus__');lr,rr=allrows[left],allrows[right]
        assert [r['sample_id'] for r in lr]==[r['sample_id'] for r in rr]
        categories={'left_only_correct':0,'right_only_correct':0,'both_correct':0,'both_wrong':0}
        for x,y in zip(lr,rr):
            xc=x['parsed']==x['target'];yc=y['parsed']==y['target']
            categories['both_correct' if xc and yc else 'left_only_correct' if xc else 'right_only_correct' if yc else 'both_wrong']+=1
        assert all(report[k]==v for k,v in categories.items())
        for key in ['balanced_accuracy','accuracy','auroc','average_precision']:
            assert abs(report['delta_'+key]-(metrics(lr)[key]-metrics(rr)[key]))<1e-12
        rng=np.random.default_rng(20260919);samples=np.zeros(100000)
        for target in [0,1]:
            delta=[int(x['parsed']==x['target'])-int(y['parsed']==y['target']) for x,y in zip(lr,rr) if x['target']==target]
            n=len(delta);counts=rng.multinomial(n,[delta.count(v)/n for v in [-1,0,1]],size=len(samples))
            samples+=(counts[:,2]-counts[:,0])/(2*n)
        resolution=sum(1/sum(r['target']==y for r in lr) for y in [0,1])/2
        assert np.max(abs(np.quantile(samples,[.025,.975])-report['paired_balanced_accuracy_95ci']))<=max(.005,2*resolution)
    result=dict(status='PASS',verified_new_generated_records=count,verified_transport_full_traces=full,
                endpoints=len(accepted),comparisons=len(pairs),maximum_errors=errors,development_inference_shuffle=intervention,
                scope='Independent implementation, same operator and reused cases; not independent patients/laboratory')
    with (root/'results-v1/independent-verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
