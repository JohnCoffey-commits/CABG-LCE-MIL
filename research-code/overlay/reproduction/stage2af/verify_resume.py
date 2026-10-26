"""CPU-only correction of ndarray handling; reuse unchanged completed GPU replays."""
import argparse,json
from pathlib import Path
import torch
from reproduction.stage2af.exact_state import equal_state
from reproduction.stage2af.runtime import sha_file,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'debug/preflight-resume-verification-v2';out.mkdir(exist_ok=False)
    left=root/'debug/preflight-continuous-v1';right=root/'debug/preflight-resume-v1'
    for d in [left,right]:
        meta=json.loads((d/'step0002.json').read_text());assert sha_file(d/'step0002.pt')==meta['sha256']
    x=json.loads((left/'step0002-record.json').read_text());y=json.loads((right/'step0002-record.json').read_text())
    for key in ['images','rng','applied_digest','controller','lr']:equal_state(x[key],y[key],key)
    a=torch.load(left/'step0002.pt',map_location='cpu',weights_only=False);b=torch.load(right/'step0002.pt',map_location='cpu',weights_only=False)
    equal_state(a,b)
    write(out/'summary.json',{'status':'PASS','resume_exact':True,'scope':'all checkpoint tensors, dtype/shape, adapter, masters, Adam, scheduler, controller, runtime, Python/NumPy/torch/CUDA RNG; raw-gradient application and sample records exact',
        'source_continuous_checkpoint_sha256':sha_file(left/'step0002.pt'),'source_resume_checkpoint_sha256':sha_file(right/'step0002.pt'),
        'new_updates':0,'new_gpu_processes':0,'original_resume_exit':1,'original_failure':'CPU state_digest did not serialize ndarray in NumPy RNG state; all preceding step comparisons passed','original_GPU_artifacts_unchanged':True})
    print((out/'summary.json').read_text(),flush=True)
if __name__=='__main__':main()
