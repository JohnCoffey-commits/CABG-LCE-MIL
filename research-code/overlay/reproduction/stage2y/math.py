"""Complete fixed input/state matrix; no independent-sample inference."""
from statistics import mean
from reproduction.stage2x.math import contrast
from reproduction.stage2y.common import ANCHORS,BLOCKS,queries,read,write

def signs(means):
    return tuple((v>0)-(v<0) for v in (means['log_support']['total'],means['top11']['total']))

def summarize(rows):
    assert len(rows) in (3,9)
    return {m:{p:mean(r[m][p] for r in rows) for p in rows[0][m]} for m in rows[0]}

def decision(reports):
    assert set(reports)==set(ANCHORS)
    transfers={};off_null=True;clip_null=True
    for name,cfg in ANCHORS.items():
        r=reports[name];assert set(r['queries'])==set(map(str,BLOCKS))
        diagonal=r['queries'][str(cfg['step'])];target=signs(diagonal['means'])
        other=[r['queries'][str(b)] for b in BLOCKS if b!=cfg['step']]
        transfers[name]=[str(b) for b in BLOCKS if b!=cfg['step'] and signs(r['queries'][str(b)]['means'])==target]
        off_null &= all(q['policy_raw_equal'] for q in other)
        clip_null &= all(q['clip_raw_equal'] for q in other)
    label='NO_RESOLVED_OFF_BATCH_POLICY_RESPONSE' if off_null else 'ALL_ANCHOR_DIRECTIONS_TRANSFER' if all(len(x)==3 for x in transfers.values()) else 'INPUT_DEPENDENT_OR_MIXED_TRANSFER'
    return dict(transfer_response=label,fully_transferring_anchors=sum(len(x)==3 for x in transfers.values()),
        matching_off_batch_cells=sum(map(len,transfers.values())),matching_blocks=transfers,
        clipping_response='NO_RESOLVED_OFF_BATCH_CLIP_RESPONSE' if clip_null else 'OFF_BATCH_CLIP_MEDIATED_ATTENTION_RESPONSE')

def main():
    import argparse,time
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--attempts',type=Path);a=p.parse_args()
    attempts=read(a.attempts or a.output/'attempts.json');assert set(attempts)==set(ANCHORS)
    reports={}
    for name,cfg in ANCHORS.items():
        r=read(a.output/name/attempts[name]/'result.json');assert r['status']=='PASS'
        cells={};off_rows=[]
        for b in queries(cfg['step']):
            q=r['queries'][str(b)];rows=[];pe=ce=True
            for j in range(3):
                c={k:q['rows'][r['aliases'][k]][j] for k in r['aliases']}
                pe &= c['11']['maps_sha256']==c['00']['maps_sha256']
                ce &= c['01']['maps_sha256']==c['00']['maps_sha256'] and c['11']['maps_sha256']==c['10']['maps_sha256']
                rows.append(dict(sample_id=c['00']['original']['sample_id'],label=c['00']['original']['label'],
                    exposure_id=c['00']['original']['exposure_id'],effects=contrast({k:v['analysis'] for k,v in c.items()})))
            cells[str(b)]=dict(rows=rows,means=summarize([x['effects'] for x in rows]),policy_raw_equal=pe,clip_raw_equal=ce)
            if b!=cfg['step']:off_rows.extend(x['effects'] for x in rows)
        reports[name]=dict(queries=cells,off_batch_means=summarize(off_rows))
    write(a.output/'response-result.json',dict(status='PASS',reports=reports,decision=decision(reports),attempts=attempts,finished_epoch=time.time()))
    print(decision(reports))
if __name__=='__main__':main()
