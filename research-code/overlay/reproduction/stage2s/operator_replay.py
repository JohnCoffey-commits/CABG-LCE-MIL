"""Same-input VJP replay and independent CPU double reference, without updates."""
import argparse
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from reproduction.stage2q.run import load_raw, save_raw
from reproduction.stage2q.verify import compare_tensor
from reproduction.stage2s.common import Budget, setup_flags, write


def main():
    p=argparse.ArgumentParser(); p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args(); budget=Budget(a.campaign); setup_flags()
    budget.reserve('block','captured_bilinear_kernel_investigation')
    out=a.campaign/'operator-replay'; out.mkdir(exist_ok=False)
    captures=load_raw(a.campaign/'native/resize-captures.pt.gz')
    report={}; saved={}
    for key,c in captures.items():
        started=time.monotonic(); x=c['input']; go=c['grad_output']; size=c['size']
        outputs={}; gradients={}; timings={}
        for setting,det in [('native',False),('torch_deterministic',True)]:
            torch.use_deterministic_algorithms(det)
            ts=time.monotonic(); gs=[]
            for repeat in range(3):
                gx=x.cuda().requires_grad_(True)
                y=F.interpolate(gx,size=size,mode='bilinear',align_corners=False)
                g=torch.autograd.grad(y,gx,go.cuda())[0]
                if repeat==0: outputs[setting]=y.detach().cpu()
                gs.append(g.detach().cpu()); del gx,y,g
            torch.cuda.synchronize(); timings[setting]=time.monotonic()-ts
            gradients[setting]=gs
        torch.use_deterministic_algorithms(False)
        # CPU autograd uses a separate CPU interpolation implementation in float64.
        cx=x.double().requires_grad_(True)
        cy=F.interpolate(cx,size=size,mode='bilinear',align_corners=False)
        cg=torch.autograd.grad(cy,cx,go.double())[0].detach()
        report[key]={'shape':list(x.shape),'size':list(size),'scope':c['scope'],
          'native_forward_matches_capture':torch.equal(outputs['native'],c['output']),
          'deterministic_forward_matches_native':torch.equal(outputs['native'],outputs['torch_deterministic']),
          'native_repeat':[compare_tensor(gradients['native'][0],g) for g in gradients['native'][1:]],
          'deterministic_repeat':[compare_tensor(gradients['torch_deterministic'][0],g) for g in gradients['torch_deterministic'][1:]],
          'native_vs_cpu_double':compare_tensor(gradients['native'][0].double(),cg),
          'deterministic_vs_cpu_double':compare_tensor(gradients['torch_deterministic'][0].double(),cg),
          'native_vs_deterministic':compare_tensor(gradients['native'][0],gradients['torch_deterministic'][0]),
          'seconds':time.monotonic()-started,'timings':timings}
        saved[key]={'outputs':outputs,'gradients':gradients,'cpu_double_gradient':cg}
        print(key,report[key],flush=True)
    raw=save_raw(out/'replays.pt.gz',saved)
    write(out/'report.json',{'cases':report,'raw':raw,'real_updates':0})

if __name__=='__main__': main()
