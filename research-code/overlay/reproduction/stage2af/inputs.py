"""One official-wrapper preprocessing cache shared by every arm and readout."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image

QUESTION=('Does this chest X-ray show any of the following findings: atelectasis, calcification, '
          'cardiomegaly, consolidation, diffuse nodule, effusion, emphysema, fibrosis, fracture, '
          'mass, nodule, pleural thickening, or pneumothorax?\nAnswer with a single word: Yes or No.')
DECODE=dict(do_sample=False,temperature=0.,top_p=1.,repetition_penalty=1.,max_new_tokens=16,
            use_cache=True,tune_mode='default',diff_mode=False)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(t):
    x=t.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()

def read_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]

def cpu_wrapper(base):
    from transformers import AutoProcessor
    from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
    w=object.__new__(Qwen2_5_VL);w.processor=AutoProcessor.from_pretrained(str(base),local_files_only=True)
    w.device='cpu';w.llm=SimpleNamespace(visual=SimpleNamespace(anomaly_qformer=SimpleNamespace(num_pooling_size=4)))
    ip=w.processor.image_processor;ip.max_pixels=50176;ip.min_pixels=784
    ip.size['longest_edge']=50176;ip.size['shortest_edge']=784
    return w

def prepare(root,base):
    root=Path(root);admission=root/'data/admission-v1'
    assert json.loads((admission/'admission.json').read_text())['status']=='PASS_IMAGE_LEVEL_ADMISSION'
    out=root/'data/prepared-v1';out.mkdir(exist_ok=False);wrapper=cpu_wrapper(base);records=[]
    for split in ['train','development','final']:
        for row in read_rows(admission/f'{split}-manifest.jsonl'):
            assert sha(row['path'])==row['sha256']
            with Image.open(row['path']) as image:batch=wrapper.process_messages({'image':image.convert('RGB'),'prompt':QUESTION})
            assert set(batch)=={'input_ids','attention_mask','pixel_values','image_grid_thw'},set(batch)
            payload={k:v.detach().cpu().contiguous() for k,v in batch.items()}
            path=out/(row['sample_id']+'.pt')
            with path.open('xb') as f:torch.save(payload,f)
            record={**row,'prepared_path':str(path),'prepared_sha256':sha(path),'pixel_sha256_prepared':tensor_sha(payload['pixel_values']),
                    'pixel_shape':list(payload['pixel_values'].shape),'image_grid_thw':payload['image_grid_thw'].tolist(),
                    'input_ids':payload['input_ids'][0].tolist(),'input_ids_sha256':tensor_sha(payload['input_ids'])}
            records.append(record)
            if len(records)%100==0:print(json.dumps({'inputs_prepared':len(records)}),flush=True)
    with (out/'manifest.jsonl').open('x') as f:
        for record in records:f.write(json.dumps(record,sort_keys=True)+'\n')
    info={'question':QUESTION,'decode':DECODE,'answer_suffix_tokens':{name:wrapper.processor.tokenizer.encode(name+'<|im_end|>',add_special_tokens=False) for name in ['Yes','No']},
          'manifest_sha256':sha(out/'manifest.jsonl'),'count':len(records),'input_has_ground_truth':False,
          'processor_limits':{'max_pixels':50176,'min_pixels':784},'source':'official Qwen2_5_VL.process_messages; CPU'}
    with (out/'input-contract.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    print(json.dumps(info),flush=True)

def load_prepared(row,device):
    path=Path(row['prepared_path'])
    if sha(path)!=row['prepared_sha256']:raise ValueError('Prepared input changed')
    batch=torch.load(path,map_location='cpu',weights_only=True)
    if tensor_sha(batch['pixel_values'])!=row['pixel_sha256_prepared']:raise ValueError('Pixel identity changed')
    if batch['input_ids'][0].tolist()!=row['input_ids']:raise ValueError('Prompt identity changed')
    if batch['image_grid_thw'].tolist()!=row['image_grid_thw']:raise ValueError('Image grid changed')
    if not torch.equal(batch['attention_mask'],torch.ones_like(batch['input_ids'])):raise ValueError('Attention mask changed')
    return {k:v.to(device) for k,v in batch.items()}

def training_batch(row,device,contract):
    batch=load_prepared(row,device);prefix=batch['input_ids'].shape[-1]
    answer=contract['answer_suffix_tokens']['Yes' if row['target'] else 'No']
    suffix=torch.tensor([answer],dtype=batch['input_ids'].dtype,device=device)
    batch['input_ids']=torch.cat([batch['input_ids'],suffix],dim=-1)
    batch['attention_mask']=torch.ones_like(batch['input_ids'])
    labels=torch.full_like(batch['input_ids'],-100);labels[:,prefix:]=suffix
    batch['labels']=labels;batch['anomaly_labels']=torch.tensor([row['target']],device=device)
    return batch

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',default='/home/checkpoints/Lingshu-7B');a=p.parse_args()
    prepare(a.root,a.base)
