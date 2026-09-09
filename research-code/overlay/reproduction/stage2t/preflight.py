"""No-model-load source/base/data/resource closure before the first GPU job."""
import argparse
import json
import shutil
from pathlib import Path
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2i.gate_d3_preflight import checkpoint_shard_audit,_gpu_snapshot
from reproduction.stage2p import BASE_SHA256,EVAL_SHA256,CONFIG_SHA256
from reproduction.stage2m.train_scout import matched_configuration
from reproduction.stage2q.run import data_contract
from reproduction.stage2t.common import source_identity,write

def eval_contract(split):
    assert sha256_file(Path(split['eval_manifest']))==EVAL_SHA256
    assert sha256_file(Path(split['eval_annotation']))==split['eval_annotation_sha256']
    rows=[json.loads(l) for l in Path(split['eval_manifest']).read_text().splitlines()]
    assert len(rows)==32 and sum(r['scout_label']=='normal' for r in rows)==12
    for key in ('sample_id','sha256','relative_path'): assert len({r[key] for r in rows})==32
    source=Path('/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2/training-development-manifest.jsonl')
    members={(r['sample_id'],r['sha256'],r['relative_path']) for r in map(json.loads,source.read_text().splitlines())}
    for r in rows:
        assert (r['sample_id'],r['sha256'],r['relative_path']) in members
        path=(Path(split['image_root'])/r['relative_path']).resolve()
        assert path.is_relative_to(Path(split['image_root']).resolve()) and sha256_file(path)==r['sha256']
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--preregistration',type=Path,required=True);a=p.parse_args()
    a.campaign.mkdir(exist_ok=False)
    source=source_identity(a.code_commit)
    split_path=Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json')
    split,train=data_contract(split_path,24);ev=eval_contract(split)
    assert split['status']=='SUCCESS' and not any(v for f in split['overlap'].values() for v in f.values())
    assert split['protected_internal_test_image_files_opened']==split['protected_internal_test_outputs_read']==0
    assert canonical_json_sha256(matched_configuration())==CONFIG_SHA256
    checkpoint=checkpoint_shard_audit(Path('/home/checkpoints/Lingshu-7B'))
    assert canonical_json_sha256(checkpoint)==BASE_SHA256
    free=shutil.disk_usage(a.campaign).free
    assert free>=38_000_000_000
    result={'status':'PASS','code_commit':a.code_commit,'source_sha256':canonical_json_sha256(source),
        'preregistration_sha256':sha256_file(a.preregistration),'base_checkpoint':checkpoint,
        'split_audit_sha256':sha256_file(split_path),'free_bytes':free,'gpu':_gpu_snapshot(),
        'train_exposures':len(train),'train_unique':len({r['sha256'] for r in train}),'eval_count':len(ev),
        'protected_internal_test_opened':0,'configuration_sha256':CONFIG_SHA256}
    write(a.campaign/'source.json',source);write(a.campaign/'train-manifest.json',train);write(a.campaign/'eval-manifest.json',ev)
    write(a.campaign/'preflight.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
