"""Bounded no-model preflight; inspect only the registered next-block image files."""
import argparse
from reproduction.stage2x.common import *
from reproduction.stage2p import TRAIN_SHA256
from reproduction.stage2q.run import data_contract
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit
from reproduction.stage2v.common import environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);WriteGuard(a.output);ins=Inputs();source=source_identity(a.code_commit,ins)
    split,manifest=data_contract(SPLIT,0);assert manifest==ins.js('train-manifest.json')
    assert split['status']=='SUCCESS' and not any(x for group in split['overlap'].values() for x in group.values())
    indices=sorted({i for cfg in ANCHORS.values() for i in range((cfg['step']-1)*3,cfg['step']*3)})
    allowed=[str((Path(split['image_root'])/manifest[i]['relative_path']).resolve()) for i in indices]
    assert len(indices)==len(set(allowed))==12
    audit=MedicalImageOpenAudit(Path(split['image_root']),allowed);audit.install()
    for i,path in zip(indices,allowed):assert sha256_file(Path(path))==manifest[i]['sha256']
    oldbase=ins.js('preflight.json')['base_checkpoint'];assert sha256_file(BASE/'model.safetensors.index.json')==oldbase['index_sha256']
    for row in oldbase['shards']:assert (BASE/row['file']).stat().st_size==row['bytes']
    anchors={}
    for name,cfg in ANCHORS.items():
        report=ins.wjs('corners/'+name+'.json');reps,aliases=variants(report)
        anchors[name]=dict(**cfg,representatives=reps,aliases=aliases,indices=list(range((cfg['step']-1)*3,cfg['step']*3)))
    forwards=sum(3*(1+len(x['representatives'])) for x in anchors.values());assert forwards==102
    result=dict(status='PASS',code_commit=a.code_commit,source_sha256=canonical_json_sha256(source),protocol_sha256=sha256_file(Path(__file__).with_name('PROTOCOL.md')),
        anchors=anchors,planned_forwards=forwards,planned_processes=10,allowed_images=allowed,indices=indices,environment=environment(),
        stage2w_verification_sha256=WVERIFY,stage2w_inventory_sha256=WINVENTORY,train_manifest_sha256=TRAIN_SHA256,
        base_check='Index hash and shard sizes; actual full model identity required after each model load',base_prior_inventory=oldbase,
        preflight_image_audit=audit.result(),free_bytes=shutil.disk_usage(a.output).free,model_loads=0,real_updates=0)
    disk(a.output);assert result['preflight_image_audit']['status']=='SUCCESS'
    write(a.output/'source.json',source);write(a.output/'train-manifest.json',manifest);write(a.output/'split.json',split);write(a.output/'plan.json',result)
    write(a.output/'attempts.json',{name:'attempt1' for name in ANCHORS})
    print(json.dumps(dict(status='PASS',sources=len(source),anchors=10,forwards=forwards,unique_images=12)))
if __name__=='__main__':main()
