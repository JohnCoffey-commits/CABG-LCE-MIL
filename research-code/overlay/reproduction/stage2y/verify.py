"""Independent raw NumPy reductions, complete diagonal bridges and matrix verdict."""
import argparse,gc,math,signal,time
import numpy as np
from reproduction.stage2y.common import *
from reproduction.stage2x.verify import measure,effects,close
from reproduction.stage2t.verify import check_spatial

def verdict(reports):
    assert set(reports)==set(ANCHORS)
    matched={};policy_same=[];clip_same=[]
    for name,cfg in ANCHORS.items():
        cells=reports[name]['queries'];assert set(cells)=={'1','9','13','21'}
        def direction(q):
            vals=[q['means'][m]['total'] for m in ('log_support','top11')]
            return [1 if x>0 else -1 if x<0 else 0 for x in vals]
        target=direction(cells[str(cfg['step'])]);matched[name]=[]
        for b in (1,9,13,21):
            if b==cfg['step']:continue
            q=cells[str(b)]
            if direction(q)==target:matched[name].append(str(b))
            policy_same.append(q['policy_raw_equal']);clip_same.append(q['clip_raw_equal'])
    num=sum(len(x)==3 for x in matched.values())
    label='NO_RESOLVED_OFF_BATCH_POLICY_RESPONSE' if all(policy_same) else 'ALL_ANCHOR_DIRECTIONS_TRANSFER' if num==10 else 'INPUT_DEPENDENT_OR_MIXED_TRANSFER'
    return dict(transfer_response=label,fully_transferring_anchors=num,matching_off_batch_cells=sum(len(x) for x in matched.values()),matching_blocks=matched,
        clipping_response='NO_RESOLVED_OFF_BATCH_CLIP_RESPONSE' if all(clip_same) else 'OFF_BATCH_CLIP_MEDIATED_ATTENTION_RESPONSE')

def averages(rows):
    return {m:{p:float(np.mean([x['effects'][m][p] for x in rows])) for p in rows[0]['effects'][m]} for m in rows[0]['effects']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    assert not torch.cuda.is_available();torch.set_num_threads(4)
    def expired(*_):raise TimeoutError('CPU verification two-hour ceiling')
    signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
    started=time.time();guard=WriteGuard(a.output);ins=Inputs();source=source_identity(a.code_commit,ins)
    plan=read(a.output/'plan.json');assert plan['code_commit']==a.code_commit and source==read(a.output/'source.json')
    assert plan['protocol_sha256']==sha256_file(Path(__file__).with_name('PROTOCOL.md'))
    manifest=read(a.output/'train-manifest.json');assert manifest==ins.js('train-manifest.json')
    producer=read(a.output/'response-result.json');assert producer['status']=='PASS';attempts=producer['attempts'];assert set(attempts)==set(ANCHORS)
    reports={};raw_files={};forwards=bridges=0;maximum_lm_error=0.
    for name,cfg in ANCHORS.items():
        anchor=a.output/name/attempts[name];r=read(anchor/'result.json');assert r['status']=='PASS' and r['code_commit']==a.code_commit
        assert set(r['queries'])=={'1','9','13','21'} and r['real_updates']==r['backward']==0 and r['complete_anchor_state_restored'] and not r['historical_write_violations']
        entry=read(anchor/'entry.json');assert entry['initial_full_exact'] and entry['anchor_full_exact'] and entry['source_sha256']==canonical_json_sha256(source)
        cfg,old,oldmaps,before,runtime,rng,_=state(ins,name)
        assert entry['master_sha256']==old['master_before'] and entry['runtime_sha256']==old['runtime_before'] and entry['rng_sha256']==old['rng']['before_block']
        reps,aliases=variants(ins.wjs('corners/'+name+'.json'));assert aliases==r['aliases']
        assert read(anchor/'monitor-summary.json')['peak_memory_used_mib']<=22500
        audit=read(anchor/'file-open-audit.json');assert audit['status']=='SUCCESS' and not audit['non_allowlisted_paths']
        assert set(audit['allowed_paths'])==set(plan['allowed_images']) and audit['open_event_count']==r['forwards']==12*(1+len(reps))
        bridge_attempt=ins.xjs('attempts.json')[name];cells={};off_rows=[]
        expected_hashes={'baseline':state_digest({n:v.bfloat16() for n,v in before.items()})}
        for key in reps:expected_hashes[key]=state_digest(candidate(ins,name,key,before))
        for block in queries(cfg['step']):
            out=anchor/f'query{block}';q=r['queries'][str(block)];assert read(out/'result.json')==dict(status='PASS',**q)
            diagonal=block==cfg['step'];assert q['diagonal']==diagonal
            assert set(q['rows'])==set(q['variants'])=={'baseline',*reps}
            if diagonal:
                gate=read(out/'baseline-gate.json');assert gate['status']=='PASS' and all(gate['gates'].values())
            measured={};map_hashes={}
            for key in ('baseline',*reps):
                vi=read(out/f'{key}-state.json');assert vi==q['variants'][key] and vi['trainable_sha256']==expected_hashes[key]
                assert vi['runtime_entry_sha256']==old['runtime_before'] and vi['rng_before']==old['rng']['before_block']
                assert vi['rng_after']==q['variants']['baseline']['rng_after'] and vi['input_list_sha256']==q['variants']['baseline']['input_list_sha256']
                if diagonal:
                    assert vi['input_list_sha256']==old['inputs_sha256'] and vi['rng_after']==old['rng']['after_block']
                    assert vi==ins.xjs(f'{name}/{bridge_attempt}/{key}-state.json')
                values=[];maps=[];assert len(q['rows'][key])==3
                for j,row in enumerate(q['rows'][key]):
                    assert row==read(out/f'{key}-image{j}.json') and row['index']==(block-1)*3+j
                    m=manifest[row['index']];o=row['original']
                    assert (o['sample_id'],o['sha256'],o['label'],o['exposure_id'])==(m['sample_id'],m['sha256'],m['scout_label'],m['scout_exposure_id'])
                    assert row['input_sha256']==vi['inputs'][j]==q['rows']['baseline'][j]['input_sha256']
                    path=out/f'{key}-image{j}.pt.gz';assert str(path)==row['raw']['file'] and sha256_file(path)==row['raw']['sha256']
                    raw=load_raw(path);raw_files[str(path)]=row['raw']['sha256'];assert state_digest(raw['maps'])==row['maps_sha256']
                    if diagonal:
                        prior=f'{name}/{bridge_attempt}/{key}-image{j}';px=ins.xjs(prior+'.json');rx=ins.xraw(prior+'.pt.gz')
                        assert row['original']==px['original'] and row['analysis']==px['analysis'] and row['input_sha256']==px['input_sha256']
                        assert state_digest(raw)==state_digest(rx);bridges+=1;del rx
                        if key=='baseline':assert o==old['images'][j] and state_digest(raw['maps'])==state_digest(oldmaps[j])
                    check_spatial(raw['maps'],o);ob=measure(raw,o['label']);close(ob,row['analysis'])
                    error=abs(ob['lm_loss']-o['lm_loss']);maximum_lm_error=max(maximum_lm_error,error)
                    assert math.isclose(ob['lm_loss'],o['lm_loss'],rel_tol=2e-6,abs_tol=2e-6)
                    values.append(ob);maps.append(row['maps_sha256']);forwards+=1;del raw
                measured[key]=values;map_hashes[key]=maps
            paired=[];pe=ce=True
            for j in range(3):
                vals={k:measured[aliases[k]][j] for k in aliases};h={k:map_hashes[aliases[k]][j] for k in aliases};o=q['rows']['baseline'][j]['original']
                pe &= h['11']==h['00'];ce &= h['01']==h['00'] and h['11']==h['10']
                paired.append(dict(sample_id=o['sample_id'],label=o['label'],exposure_id=o['exposure_id'],effects=effects(vals)))
            cells[str(block)]=dict(rows=paired,means=averages(paired),policy_raw_equal=pe,clip_raw_equal=ce)
            if not diagonal:off_rows.extend(paired)
        assert len(off_rows)==9;reports[name]=dict(queries=cells,off_batch_means=averages(off_rows));close(reports[name],producer['reports'][name])
        del before,runtime,oldmaps;gc.collect();print(json.dumps(dict(verified_anchor=name,forwards=forwards,bridges=bridges)),flush=True)
    assert forwards==408 and bridges==102 and len(raw_files)==408
    decision=verdict(reports);assert decision==producer['decision']
    budget=read(a.output/'budget.json');assert budget['real_updates']==budget['backward']==0
    assert 408<=budget['counts']['forward']<=528 and 10<=budget['counts']['process']<=14
    assert producer['finished_epoch']-budget['started_epoch']<=5400
    write(a.output/'verification.json',dict(status='PASS',decision=decision,reports=reports,accepted_forwards=forwards,exact_diagonal_bridges=bridges,raw_files=raw_files,
        budget=budget['counts'],real_updates=0,backward=0,development_evaluations=0,protected_access=0,code_commit=a.code_commit,
        source_sha256=canonical_json_sha256(source),producer_sha256=sha256_file(a.output/'response-result.json'),
        maximum_lm_error=maximum_lm_error,prior_inputs_checked=ins.checked,candidate_inputs_checked=ins.wchecked,bridge_inputs_checked=ins.xchecked,
        seconds=time.time()-started,scope='Complete fixed state/input matrix on twelve reused training images; not held-out transfer or trajectory evidence'))
    print(json.dumps(dict(status='PASS',decision=decision),indent=2))
if __name__=='__main__':main()
