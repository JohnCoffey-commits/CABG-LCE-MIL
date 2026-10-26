"""Cache coverage and direct official-wrapper equivalence before model work."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image
from reproduction.stage2af.inputs import read_rows,load_prepared,training_batch,cpu_wrapper,QUESTION,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;out=root/'data/prepared-v1'
    contract=json.loads((out/'input-contract.json').read_text());assert sha(out/'manifest.jsonl')==contract['manifest_sha256']
    rows=read_rows(out/'manifest.jsonl');original=[]
    for split in ['train','development','final']:original+=read_rows(root/'data/admission-v1'/f'{split}-manifest.jsonl')
    assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in original]
    wrapper=cpu_wrapper('/home/checkpoints/Lingshu-7B');direct=0
    # Full coverage verifies all stored tensor/manifest identities; independent
    # direct preprocessing at fixed first/last indices for each split checks path.
    direct_ids=set()
    for split in ['train','development','final']:
        subset=[r for r in rows if r['split']==split];direct_ids|={subset[0]['sample_id'],subset[-1]['sample_id']}
    for row in rows:
        batch=load_prepared(row,'cpu');train=training_batch(row,'cpu',contract);n=batch['input_ids'].shape[1]
        assert set(batch)=={'input_ids','attention_mask','pixel_values','image_grid_thw'}
        for k in ['pixel_values','image_grid_thw']:assert torch.equal(batch[k],train[k])
        assert torch.equal(batch['input_ids'],train['input_ids'][:,:n]);assert train['labels'][:,:n].eq(-100).all()
        assert int(train['anomaly_labels'].item())==row['target']
        if row['sample_id'] in direct_ids:
            with Image.open(row['path']) as im:fresh=wrapper.process_messages({'image':im.convert('RGB'),'prompt':QUESTION})
            assert set(fresh)==set(batch)
            for k in batch:assert torch.equal(fresh[k],batch[k]),k
            direct+=1
    report={'status':'PASS','cache_records_checked':len(rows),'direct_official_wrapper_checks':direct,
            'train_evaluation_pixels_and_prefix_identical':True,'labels_absent_from_prompt_inputs':True,
            'input_contract_sha256':sha(out/'input-contract.json'),'model_calls':0}
    with (out/'verification.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
