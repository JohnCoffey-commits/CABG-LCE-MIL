"""No-model source/data/base/restored-artifact checks before GPU work."""
import argparse,json,shutil,time
from pathlib import Path
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.gate_d3_preflight import checkpoint_shard_audit
from reproduction.stage2p import BASE_SHA256,TRAIN_SHA256,EVAL_SHA256
from reproduction.stage2q.run import data_contract
from reproduction.stage2t.preflight import eval_contract
from reproduction.stage2v.common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    a.campaign.mkdir(exist_ok=False)
    source=source_identity(a.code_commit)
    old_source=read(HISTORICAL_CONTROL/'source.json')
    assert all(source[n]==sha for n,sha in old_source.items()),'Historical scientific/model source changed'
    inventory_path=HISTORICAL_CONTROL/'artifact-inventory-v1.json'
    assert sha256_file(inventory_path)=='a9a62501f30ecae50593b7ac8a130093b4a376154a2dacf166e05967bafdc625'
    inventory=read(inventory_path)
    for entry in inventory['entries']:
        path=FULL24.parent/entry['root']/entry['path']
        assert path.stat().st_size==entry['bytes'] and sha256_file(path)==entry['sha256'],str(path)
    write(a.campaign/'restored-stage2u-artifact-check.json',dict(status='PASS',files=len(inventory['entries']),bytes=inventory['bytes'],inventory_sha256=sha256_file(inventory_path)))
    verify=HISTORICAL/'verification.json';assert read(verify)['decision']=='VALID_COMPARISON'
    full_verify=FULL24/'accepted-v2/verification-v1/verification.json';assert read(full_verify)['decision']=='PASS'
    split,train=data_contract(Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json'),24)
    ev=eval_contract(split)
    assert split['status']=='SUCCESS' and not any(v for f in split['overlap'].values() for v in f.values())
    base=checkpoint_shard_audit(Path('/home/checkpoints/Lingshu-7B'));assert canonical_json_sha256(base)==BASE_SHA256
    # Explicit bridge locks are rechecked at producer restore and independent verification.
    bridge={}
    for cursor in (12,23,24):
        ck=FULL24/'R0'/f'block{cursor}-checkpoint/checkpoint.pt'
        bridge[str(cursor)]={name:sha256_file(ck.parent/name) for name in ('checkpoint.pt','checkpoint.pt.manifest.json','runtime-state.pt.gz','runtime-state.json','full-state.json')}
    free=shutil.disk_usage(a.campaign).free;assert free>=46_000_000_000
    result=dict(status='PASS',code_commit=a.code_commit,source_sha256=canonical_json_sha256(source),
        protocol_sha256=sha256_file(Path(__file__).with_name('PROTOCOL.md')),environment=environment(),base_checkpoint=base,
        train_manifest_sha256=TRAIN_SHA256,eval_manifest_sha256=EVAL_SHA256,free_bytes=free,bridge=bridge,
        stage2u_verification_sha256=sha256_file(verify),full24_verification_sha256=sha256_file(full_verify),
        seeds=list(SEEDS),planned_counts=dict(update=72,block=74,eval=224),limits=LIMITS,
        model_loaded=False,gpu_updates=0,created_epoch=time.time())
    write(a.campaign/'source.json',source);write(a.campaign/'train-manifest.json',train);write(a.campaign/'eval-manifest.json',ev)
    write(a.campaign/'preflight.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
