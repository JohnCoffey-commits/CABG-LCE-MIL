"""Producer reductions and cloned native Adam kernels; no training calls."""
import math
import torch
from reproduction.stage2s.common import state_digest
from reproduction.stage2u.verify import reconstruct
from reproduction.stage2q.verify import norm32
from reproduction.stage2w.common import GROUPS,TRAINABLE_NAMES,S9_NAMES

def aggregates(d,r):
    actual=reconstruct(d,r,0.)
    lm={n:torch.stack([g[n].float() for g in d['lm']]).mean(0) for n in TRAINABLE_NAMES}
    labels=[x['label'] for x in r['images']]
    aux={n:.5*(torch.stack([d['lce'][i][n].float() for i,x in enumerate(labels) if x=='normal']).mean(0)+torch.stack([d['lce'][i][n].float() for i,x in enumerate(labels) if x=='abnormal']).mean(0)) for n in S9_NAMES}
    assert state_digest(lm)==r['aggregate_lm'] and state_digest(aux)==r['aggregate_lce']
    return lm,aux,actual

def corners(lm,aux,coefficient):
    g1={n:g.clone() for n,g in lm.items()}
    for n in aux:g1[n].add_(aux[n]*coefficient)
    c0=min(1.,1./(norm32(lm)+1e-12));c1=min(1.,1./(norm32(g1)+1e-12))
    return {a+b:{n:g*c for n,g in gradients.items()} for a,gradients in [('0',lm),('1',g1)] for b,c in [('0',c0),('1',c1)]},(c0,c1)

def dot(a,b,names):return sum(float((a[n].double()*b[n].double()).sum()) for n in names)
def geometry(lm,aux,coefficient):
    auxiliary={n:aux[n]*coefficient if n in aux else torch.zeros_like(lm[n]) for n in lm}
    out={}
    for group,names in GROUPS.items():
        ll,aa,la=dot(lm,lm,names),dot(auxiliary,auxiliary,names),dot(lm,auxiliary,names)
        den=math.sqrt(ll*aa)
        out[group]=dict(lm_norm=math.sqrt(ll),scaled_lce_norm=math.sqrt(aa),dot=la,
            cosine=la/den if den else None,cancellation_ratio=math.sqrt(max(0,ll+aa+2*la))/(math.sqrt(ll)+math.sqrt(aa)) if ll+aa else None)
    return out

@torch.no_grad()
def native(before,opt,applied,device):
    """Exactly the installed noncapturable foreach order, on detached clones."""
    group=opt['param_groups'][0];names=list(TRAINABLE_NAMES);states=opt['state']
    ps=[before[n].to(device).clone() for n in names];gs=[applied[n].to(device) for n in names]
    ms=[];vs=[];steps=[]
    for i,p in enumerate(ps):
        s=states.get(i)
        ms.append(s['exp_avg'].to(device).clone() if s else torch.zeros_like(p))
        vs.append(s['exp_avg_sq'].to(device).clone() if s else torch.zeros_like(p))
        steps.append(float(s['step'])+1 if s else 1.)
    torch._foreach_lerp_(ms,gs,1-.9)
    torch._foreach_mul_(vs,.999);torch._foreach_addcmul_(vs,gs,gs,1-.999)
    denom=torch._foreach_sqrt(vs);torch._foreach_div_(denom,[(1-.999**s)**.5 for s in steps]);torch._foreach_add_(denom,1e-8)
    torch._foreach_addcdiv_(ps,ms,denom,[-group['lr']/(1-.9**s) for s in steps])
    return dict(zip(names,[p.cpu() for p in ps])),{i:dict(step=torch.tensor(steps[i],dtype=torch.float32),exp_avg=ms[i].cpu(),exp_avg_sq=vs[i].cpu()) for i in range(len(names))}

def attribution(values):
    out={}
    for group,names in GROUPS.items():
        total2=direct2=clip2=dot_direct=dot_clip=closure=0.
        counts={k:0 for k in ('fp32_policy','bf16_policy','fp32_direct_c0','fp32_direct_c1','bf16_direct_c0','bf16_direct_c1','fp32_clip_g0','fp32_clip_g1','bf16_clip_g0','bf16_clip_g1')}
        for n in names:
            f={k:v[n].double() for k,v in values.items()}
            d=f['11']-f['00'];a=((f['10']-f['00'])+(f['11']-f['01']))*.5;b=((f['01']-f['00'])+(f['11']-f['10']))*.5
            total2+=float(d.square().sum());direct2+=float(a.square().sum());clip2+=float(b.square().sum())
            dot_direct+=float((a*d).sum());dot_clip+=float((b*d).sum());closure=max(closure,float((a+b-d).abs().max()))
            for pair,key in [(('11','00'),'policy'),(('10','00'),'direct_c0'),(('11','01'),'direct_c1'),(('01','00'),'clip_g0'),(('11','10'),'clip_g1')]:
                x,y=pair;counts['fp32_'+key]+=int((values[x][n]!=values[y][n]).sum());counts['bf16_'+key]+=int((values[x][n].bfloat16()!=values[y][n].bfloat16()).sum())
        assert closure<=1e-15
        out[group]=dict(total_squared_norm=total2,direct_squared_norm=direct2,clip_squared_norm=clip2,
            direct_projection=dot_direct/total2 if total2 else None,clip_projection=dot_clip/total2 if total2 else None,
            direct_dot_total=dot_direct,clip_dot_total=dot_clip,closure_max_abs=closure,**counts)
    t=out['T12'];assert t['fp32_direct_c0']==t['fp32_direct_c1']==t['bf16_direct_c0']==t['bf16_direct_c1']==0
    return out

def memory_attribution(before,opt,g,step):
    rows={};group=opt['param_groups'][0]
    for label,names in GROUPS.items():
        total2=memory2=current2=mp=cp=0.
        for n in names:
            i=TRAINABLE_NAMES.index(n);state=opt['state'].get(i);grad=g[n].double()
            m=state['exp_avg'].double() if state else torch.zeros_like(grad);v=state['exp_avg_sq'].double() if state else torch.zeros_like(grad)
            factor=-group['lr']/(1-.9**step)/((.999*v+(1-.999)*grad.square()).sqrt()/math.sqrt(1-.999**step)+1e-8)
            inherited=factor*(.9*m);current=factor*((1-.9)*grad);total=inherited+current
            total2+=float(total.square().sum());memory2+=float(inherited.square().sum());current2+=float(current.square().sum());mp+=float((inherited*total).sum());cp+=float((current*total).sum())
        rows[label]=dict(total_squared_norm=total2,memory_squared_norm=memory2,current_squared_norm=current2,memory_projection=mp/total2 if total2 else None,current_projection=cp/total2 if total2 else None)
    return rows

def decision(reports):
    rows=[r['attribution']['T12'] for r in reports.values()]
    outcome='CLIP_MEDIATED_BF16_SPILLOVER' if any(r['bf16_policy'] for r in rows) else 'FP32_ONLY_CLIP_SPILLOVER' if any(r['fp32_policy'] for r in rows) else 'NO_LOCAL_T12_SPILLOVER_AT_REGISTERED_ANCHORS'
    defined=[r['attribution']['T21']['clip_projection'] for r in reports.values() if r['attribution']['T21']['clip_projection'] is not None]
    dominance='DIRECT_DOMINANT_AT_ALL_ANCHORS' if len(defined)==len(reports) and all(abs(x)<=.10 for x in defined) else 'STATE_DEPENDENT_MIXED'
    return dict(spillover=outcome,dominance=dominance,undefined_anchors=len(reports)-len(defined))
