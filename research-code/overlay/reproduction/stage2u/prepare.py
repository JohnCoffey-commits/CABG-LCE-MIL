"""CPU-only preparation. Never loads a model or executes any optimizer update."""
import argparse,json,shutil
import torch
from pathlib import Path
from reproduction.stage2i.fingerprint import canonical_json_sha256,sha256_file
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2q.run import data_contract
from reproduction.stage2s.common import write,state_digest
from reproduction.stage2t.preflight import eval_contract
from reproduction.stage2u.common import FULL24,PARENT_SHA,source_identity,load_raw

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(4);source=source_identity(a.code_commit)
    verify=FULL24/'accepted-v2/verification-v1/verification.json';v=json.loads(verify.read_text())
    assert v['decision']=='PASS' and v['exact'] and v['concentration']=='REPRODUCED'
    parent=FULL24/'R0/block12-checkpoint/checkpoint.pt';assert sha256_file(parent)==PARENT_SHA
    parent_prov=json.loads((FULL24/'R0/provenance.json').read_text());payload,manifest=load_strict(parent,parent_prov)
    assert payload['sampler_state']['cursor']==12
    side=parent.with_name('runtime-state.pt.gz');info=json.loads(parent.with_name('runtime-state.json').read_text())
    assert sha256_file(side)==info['raw']['sha256'] and info['checkpoint_sha256']==PARENT_SHA
    assert state_digest(load_raw(side))==json.loads((FULL24/'R0/block12.json').read_text())['runtime_after']
    split,train=data_contract(Path('/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json'),24);ev=eval_contract(split)
    fixed=json.loads((FULL24/'R0/block12.json').read_text())['controller']['lambda_raw']
    # Bind the policy to actual archived T21/S9 gradients without a model or update.
    from reproduction.stage2l.controller import DynamicController
    from reproduction.stage2u.policy import gradients
    from reproduction.stage2q.verify import compare_map
    ref=json.loads((FULL24/'R0/block13.json').read_text());rawpath=FULL24/'R0/block13.pt.gz'
    assert sha256_file(rawpath)==ref['raw']['sha256'];raw=load_raw(rawpath)
    lm=[{n:v.float() for n,v in g.items()} for g in raw['lm']];lce=[{n:v.float() for n,v in g.items()} for g in raw['lce']]
    optimizer_before=state_digest(payload['optimizer_state']);policies={}
    for arm in ('dynamic','frozen_ratio','lce_off'):
        controller=DynamicController();controller.load_state_dict(payload['controller_state'])
        applied,details=gradients(lm,lce,[r['label'] for r in ref['images']],controller=controller,arm=arm,frozen=fixed)
        assert details['budget']==ref['controller']
        difference=compare_map(applied,raw['applied'])
        if arm=='dynamic':assert difference['exact']
        else:assert not difference['exact']
        policies[arm]={'selected_lambda':details['policy']['selected_lambda'],'shadow_exact':True,
                       'applied_exact_vs_archived_dynamic':difference['exact'],'applied_relative_l2_vs_dynamic':difference['relative_l2']}
    assert state_digest(payload['optimizer_state'])==optimizer_before
    write(a.output/'real-gradient-policy-cpu.json',{'status':'PASS','parent_checkpoint_sha256':PARENT_SHA,
        'raw_block13_sha256':ref['raw']['sha256'],'policies':policies,'optimizer_unchanged':True,
        'model_loaded':False,'gpu_operations':0,'optimizer_updates':0})
    protocol=Path(__file__).with_name('PROTOCOL.md')
    free=shutil.disk_usage(a.output).free
    assert free>=16_000_000_000,free
    plan={'status':'CPU_READY_GPU_NOT_AUTHORIZED','code_commit':a.code_commit,'source_sha256':canonical_json_sha256(source),
          'parent_checkpoint':str(parent),'parent_checkpoint_sha256':PARENT_SHA,'parent_provenance':parent_prov,
          'parent_manifest_sha256':sha256_file(parent.with_suffix('.pt.manifest.json')),'runtime_sha256':sha256_file(side),
          'full24_verification':str(verify),'full24_verification_sha256':sha256_file(verify),
          'protocol_sha256':sha256_file(protocol),'frozen_ratio':fixed,'parent_development_context':v['evaluation']['R0-b12'],
          'train_manifest_sha256':split['train_manifest_sha256'],'eval_manifest_sha256':split['eval_manifest_sha256'],
          'free_bytes':free,'planned_updates':36,'planned_eval_forwards':96,'hard_updates':40,'hard_eval_forwards':128,
          'hard_seconds':7200,'hard_artifact_bytes':12_000_000_000,'free_space_floor_bytes':4_000_000_000,
          'shared_parent_claim':'persistence/reversal, not prevention of first onset','gpu_operations':0}
    write(a.output/'source.json',source);write(a.output/'plan.json',plan)
    write(a.output/'cpu-parent-audit.json',{'status':'PASS','checkpoint_tensor_inventory':manifest['tensor_inventory_sha256'],
        'rng':manifest['rng_fingerprint'],'sampler':payload['sampler_state'],'runtime_digest':state_digest(load_raw(side)),
        'parent_optimizer_digest':state_digest(payload['optimizer_state']),'train_exposures':len(train),'eval_count':len(ev)})
    print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
