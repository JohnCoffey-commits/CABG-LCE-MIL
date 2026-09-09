"""Promotion test: actual captured tensors, scoped VJP, no model update."""
import argparse
from pathlib import Path
import torch
from reproduction.stage2q.run import load_raw
from reproduction.stage2q.verify import compare_tensor
from reproduction.stage2s.common import Budget,setup_flags,write
from reproduction.stage2s.resize import DeterministicBilinear

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();budget=Budget(a.campaign);setup_flags()
    budget.reserve('block','scoped_bilinear_correction_validation')
    caps=load_raw(a.campaign/'native/resize-captures.pt.gz')
    refs=load_raw(a.campaign/'operator-replay/replays.pt.gz');report={}
    for key,c in caps.items():
        results=[]
        for i in range(2):
            x=c['input'].cuda().requires_grad_(True)
            y=DeterministicBilinear.apply(x,c['size'])
            dx,=torch.autograd.grad(y,x,c['grad_output'].cuda())
            assert torch.equal(y.cpu(),c['output'])
            assert not torch.are_deterministic_algorithms_enabled()
            assert torch.equal(dx.cpu(),refs[key]['gradients']['torch_deterministic'][0])
            results.append(dx.cpu())
        assert torch.equal(*results)
        ref=refs[key]['cpu_double_gradient']
        error=compare_tensor(results[0].double(),ref)
        # BF16 rounding unit is 1/256; no meaningful sign error is permitted.
        assert error['relative_l2'] <= 1/256 and error['meaningful_sign_disagreements']==0
        report[key]={'native_forward_exact':True,'repeated_vjp_exact':True,
          'torch_deterministic_vjp_exact':True,'global_flag_restored':True,
          'cpu_double':error,'rounded_cpu_reference':compare_tensor(results[0],ref.to(torch.bfloat16))}
    write(a.campaign/'correction-validation.json',{'status':'PASS','cases':report,'real_updates':0})
    print('SCOPED_CORRECTION_PASS',flush=True)

if __name__=='__main__':main()
