"""Pre-model identity and audited fixed train/development hash opens."""
import argparse
from reproduction.stage2aa.common import *
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit
from reproduction.stage2t.preflight import eval_contract
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2m.constants import TRAINABLE_NAMES

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    a.campaign.mkdir(exist_ok=False);WriteGuard(a.campaign);ins=Inputs();source=source_identity(a.code_commit)
    assert environment()==ins.zjs('preflight.json')['environment']
    split,train=data_contract(SPLIT,0);ev=ins.js('eval-manifest.json');assert train==ins.js('train-manifest.json')
    allowed=sorted({str((Path(split['image_root'])/r['relative_path']).resolve()) for r in train+ev});assert len(allowed)==96
    audit=MedicalImageOpenAudit(Path(split['image_root']),allowed);audit.install()
    check,train2=data_contract(SPLIT,24);ev2=eval_contract(split);assert check==split and train2==train and ev2==ev
    assert split['status']=='SUCCESS' and not any(v for f in split['overlap'].values() for v in f.values())
    ar=audit.result();assert ar['status']=='SUCCESS' and ar['open_event_count']==104
    base=ins.js('preflight.json')['base_checkpoint'];assert sha256_file(BASE/'model.safetensors.index.json')==base['index_sha256']
    for row in base['shards']:assert (BASE/row['file']).stat().st_size==row['bytes']
    # Historical inputs are checked against their closed inventory when consumed.
    locks={}
    for seed in SEEDS:
        name=f's{seed}-lm_only';prov=ins.zjs(name+'/provenance.json')
        payload,meta=load_strict(ins.zpath(name+'/block12-checkpoint/checkpoint.pt'),prov)
        record=ins.zjs(name+'/block12.json');runtime=ins.zraw(name+'/block12-checkpoint/runtime-state.pt.gz')
        assert payload['sampler_state']['cursor']==payload['trace_state']['completed_blocks']==12
        assert payload['trace_state']['last_record_sha256']==record['record_sha256']
        assert state_digest(payload['master_state'])==record['master_after']
        assert state_digest(payload['optimizer_state'])==record['optimizer_after']
        assert rng_state_fingerprint(payload['rng_state'])==record['rng']['after_block']
        assert state_digest(runtime)==record['runtime_after']
        assert all(torch.equal(payload['adapter_state'][n],payload['master_state'][n].bfloat16()) for n in TRAINABLE_NAMES)
        for step in range(1,13):assert ins.zjs(name+f'/block{step}.json')['policy']['selected_lambda']==0
        locks[str(seed)]=dict(initial_identity=ins.js(f's{seed}-prefix/initial-identity.json'),
            parent=meta,provenance=prov,endpoint=ins.zjs(name+'/block24-checkpoint/checkpoint.pt.manifest.json'))
        del payload,runtime
    free=shutil.disk_usage(a.campaign).free;assert free>=STORAGE+FLOOR
    result=dict(status='PASS',code_commit=a.code_commit,source_sha256=canonical_json_sha256(source),
        protocol_sha256=sha256_file(Path(__file__).with_name('PROTOCOL.md')),environment=environment(),base_prior_inventory=base,
        base_check='Index hash/shard sizes; actual full initial parameters required at every load',
        train_manifest_sha256=TRAIN_SHA256,eval_manifest_sha256=EVAL_SHA256,free_bytes=free,locks=locks,
        stage2z_verification_sha256=ZVERIFY,stage2z_inventory_sha256=ZINVENTORY,
        prior_inputs_checked=dict(stage2v=ins.checked,stage2z=ins.zchecked),
        seeds=list(SEEDS),planned_counts=PLANNED,limits=LIMITS,preflight_image_audit=ar,
        model_loaded=False,gpu_updates=0,created_epoch=time.time())
    write(a.campaign/'source.json',source);write(a.campaign/'train-manifest.json',train);write(a.campaign/'eval-manifest.json',ev)
    write(a.campaign/'split.json',split);write(a.campaign/'preflight.json',result)
    print(json.dumps(dict(status='PASS',sources=len(source),planned=PLANNED,free_bytes=free),indent=2))
if __name__=='__main__':main()
