"""Independent scalar-controller, sparse-gradient and checkpoint audit at closure."""
import argparse,gzip,io,json,math,hashlib
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def norm(g):return math.sqrt(sum(float(np.sum(v.astype(float)**2)) for v in g.values()))
def rms(rows):return math.sqrt(.5*norm(rows[0])**2+.25*norm(rows[1])**2+.25*norm(rows[2])**2+1e-16)
def close(actual,expected,relative=2e-6):assert abs(actual-expected)<=max(1e-10,relative*max(abs(actual),abs(expected))),(actual,expected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trajectory');a=p.parse_args();root=a.root;plan=read(root/'protocol.json');protocol=sha(root/'protocol.json');old=Path('/home/Medic-AD/outputs/cabg-mil-v1.2/scale-study-20260918-v1');baseline=read(old/'protocol.json');amend=read(old/'engineering-amendment-02.json');plan.update(initial_frozen_parameters=baseline['initial_frozen_parameters'],initial_buffers=baseline['initial_buffers'])
    torch.set_num_threads(4);results={};sparse_count=0;checkpoint_count=0;steps_count=0
    # This closure checker requires all trajectory statuses. Partial/failed
    # trajectories remain explicit rather than being silently removed.
    expected={f's{s}-{arm}' for s in [43,44] for arm in plan['arms']}
    attempts=read(root/'accepted-training-attempts.json') if (root/'accepted-training-attempts.json').exists() else {name:'attempt-v1' for name in expected}
    amendment=amend
    if a.trajectory:
        assert a.trajectory in expected
        d=root/'formal'/a.trajectory/attempts[a.trajectory]
        terminal={a.trajectory:read(d/('summary.json' if (d/'summary.json').is_file() else 'failure.json'))}
    else:
        terminal={name:read(root/'formal'/name/'attempt-v1'/'summary.json') for name in expected}
        assert set(terminal)==expected
    for trajectory,status in terminal.items():
        d=root/'formal'/trajectory/attempts[trajectory];seed=int(trajectory[1:3]);arm=trajectory[4:]
        initial=read(d/'initial-state.json');parameters={r['name']:r for r in initial['parameters']}
        frozen_reference=dict(plan['initial_frozen_parameters'])
        if amendment and seed==44:frozen_reference.update(amendment['seed44_frozen_overrides'])
        for name,row in frozen_reference.items():assert parameters[name]==row
        assert initial['buffers']==plan['initial_buffers'];sequence=read(d/'sequence.json');assert len(sequence)==3000
        oldattempt='attempt-v1' if seed==43 else 'attempt-v2'
        assert sequence==read(old/'formal'/f's{seed}-fixed_lce'/oldattempt/'sequence.json')
        assert read(d/'file-open-audit-final.json')['violations']==[]
        assert status['frozen']['frozen_parameters_unchanged']
        assert status['transport']['nonzero_gradient_events']>0
        if seed==43:assert read(d/'resume-test.json')['status']=='PASS'
        bystep={int(p.stem[4:]):read(p) for p in d.glob('step[0-9][0-9][0-9][0-9].json')};last=max(bystep,default=0)
        assert set(bystep)==set(range(1,last+1))
        if status['status']=='COMPLETE':assert last==1000
        else:assert status['status']=='NUMERICAL_FAILURE' and last==status['last_completed_step']
        lm_ema=lce_ema=0.;coefficient=plan['fixed_coefficients'][str(seed)]
        for step,record in sorted(bystep.items()):
            assert [x['sample_id'] for x in record['images']]==[x['sample_id'] for x in sequence[(step-1)*3:step*3]]
            assert [x['label'] for x in record['images']]==['normal','abnormal','abnormal']
            lr=1e-4*.5*(1+math.cos(math.pi*(step-1)/1000));close(record['lr'],lr,1e-12)
            lm_ema=.9*lm_ema+.1*record['lm_rms'];lce_ema=.9*lce_ema+.1*record['lce_rms'];correction=1-.9**step
            budget=record['controller'];close(budget['lm_ema'],lm_ema);close(budget['lce_ema'],lce_ema)
            expected_raw=.1*(lm_ema/correction)/(lce_ema/correction+1e-8)
            expected_cap=.2*budget['lm_s9_norm']/(budget['lce_s9_norm']+1e-8)
            close(budget['lambda_raw'],expected_raw);close(budget['lambda_cap'],expected_cap)
            applied=coefficient
            close(record['applied_coefficient'],applied);assert record['applied_dynamic_cap']==(arm=='cabg_lce')
            g=record['gradient']
            close(g['lm_shared_norm'],budget['lm_s9_norm'])
            close(g['scaled_auxiliary_shared_norm'],applied*budget['lce_s9_norm'])
            if arm=='cabg_lce':
                assert g['scaled_auxiliary_shared_norm']<=.2*g['lm_shared_norm']+g['trust_cap_tolerance']
            if step in plan['sparse_gradient_steps']:
                raw=record['raw'];assert raw and sha(raw['path'])==raw['sha256']
                with gzip.open(raw['path'],'rb') as f:data=torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=False)
                lm=[{n:t.float().numpy() for n,t in row.items()} for row in data['lm']];aux=[{n:t.float().numpy() for n,t in row.items()} for row in data['lce']];names=set(aux[0])
                assert len(lm)==len(aux)==3 and len(lm[0])==21 and len(names)==9
                close(record['lm_rms'],rms([{n:g[n] for n in names} for g in lm]));close(record['lce_rms'],rms(aux))
                gl={n:(lm[0][n].astype(float)+lm[1][n]+lm[2][n])/3 for n in lm[0]};ga={n:.5*aux[0][n].astype(float)+.25*aux[1][n]+.25*aux[2][n] for n in names}
                combined={n:gl[n]+applied*ga[n] if n in names else gl[n] for n in gl};pre=norm(combined);scale=min(1,1/(pre+1e-12))
                close(record['gradient']['preclip_full_norm'],pre);close(record['gradient']['clip_scale'],scale)
                close(record['gradient']['scaled_auxiliary_shared_norm'],applied*norm(ga));close(record['gradient']['postclip_full_norm'],pre*scale)
                sparse_count+=1;del data,lm,aux,gl,ga,combined
            steps_count+=1
        for step in plan['checkpoint_steps']:
            cp=d/f'checkpoint-step{step:04}.pt';meta=cp.with_suffix('.json')
            if step>last:continue
            assert cp.is_file() and meta.is_file();record=read(meta);assert cp.stat().st_size==record['bytes'] and sha(cp)==record['sha256']
            payload=torch.load(cp,map_location='cpu',weights_only=False);assert payload['step']==step and payload['protocol_sha256']==protocol
            assert payload['arm']==arm and payload['seed']==seed and len(payload['masters'])==len(payload['adapter'])==21
            assert payload['scheduler']['last_epoch']==step and payload['controller']['valid_blocks']==step
            for name,tensor in payload['masters'].items():assert torch.isfinite(tensor).all() and torch.equal(tensor.to(torch.bfloat16),payload['adapter'][name])
            assert len(payload['optimizer']['state'])==21
            for state in payload['optimizer']['state'].values():
                assert int(state['step'])==step and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and (state['exp_avg_sq']>=0).all()
            checkpoint_count+=1;del payload
        results[trajectory]={'status':status['status'],'attempt':attempts[trajectory],'completed_steps':last,'sparse_expected':[s for s in plan['sparse_gradient_steps'] if s<=last]}
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()];reserved=[x for x in events if x['kind']=='update_reserved']
    assert len(reserved)==4000 and all(x['category']=='formal' for x in reserved)
    result={'status':'PASS','scope':'all scalar/controller/schedule/order records and checkpoint state structure; sparse raw-gradient NumPy norm/composition checks; not a dense replay of every optimizer update',
            'trajectories':results,'steps_verified':steps_count,'sparse_blocks_verified':sparse_count,'checkpoints_verified':checkpoint_count,'protocol_sha256':protocol,'relative_norm_tolerance':2e-6}
    filename='training-independent-verification'+('-'+a.trajectory if a.trajectory else '')+'-v1.json'
    with (root/filename).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
