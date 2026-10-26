"""Long-horizon optimizer and three matched compositions using verified LCE math."""
from __future__ import annotations
import math
import statistics
import torch
from reproduction.stage2i.cabg_math import (
    aggregate_lm_gradients, aggregate_class_balanced_gradients,
    per_image_class_balanced_rms, gradient_norm, combine_and_clip_gradients,
)
from reproduction.stage2l.controller import DynamicController, compute_block_gradients
from reproduction.stage2l.constants import TRAINABLE_NAMES,S9_NAMES,EPSILON,RHO,RHO_MAX
from reproduction.stage2k.producer_math import compute_lce

ARMS=('lm_only','fixed_lce','cabg_lce')
HORIZON=1000

def multiplier(step):
    if step<0:raise ValueError('Negative scheduler step')
    return .5*(1+math.cos(math.pi*min(step,HORIZON)/HORIZON))

def make_optimizer(initial,device):
    masters={n:torch.nn.Parameter(initial[n].to(device).float().clone(),requires_grad=False) for n in TRAINABLE_NAMES}
    optimizer=torch.optim.AdamW(list(masters.values()),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=multiplier)
    return masters,optimizer,scheduler

def calibrate_initial_coefficient(lm,lce,labels,shared_names=S9_NAMES):
    lm_rms,_=per_image_class_balanced_rms([{n:g[n] for n in shared_names} for g in lm],labels,epsilon=EPSILON)
    lce_rms,_=per_image_class_balanced_rms(lce,labels,epsilon=EPSILON)
    gl=aggregate_lm_gradients(lm);ga=aggregate_class_balanced_gradients(lce,labels)
    ln=float(gradient_norm({n:gl[n] for n in shared_names}));an=float(gradient_norm(ga))
    raw=RHO*float(lm_rms)/(float(lce_rms)+EPSILON);cap=RHO_MAX*ln/(an+EPSILON)
    return {'lm_rms':float(lm_rms),'lce_rms':float(lce_rms),'lm_shared_norm':ln,
            'lce_shared_norm':an,'initial_raw':raw,'initial_cap':cap,'coefficient':min(raw,cap)}

def fixed_coefficient(records):
    if len(records)!=8:raise ValueError('Exactly eight initial training blocks required')
    values=[r['coefficient'] for r in records]
    if not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Nonpositive/nonfinite training calibration')
    return statistics.median(values)

def compose(lm,lce,labels,arm,controller,coefficient=None,trainable_names=TRAINABLE_NAMES,shared_names=S9_NAMES):
    if arm not in ARMS:raise ValueError(arm)
    # Identical diagnostic gradient collection and controller observation in all
    # arms; only the coefficient actually applied differs. Fixed and LM-only have
    # no hidden dynamic cap on their applied auxiliary gradient.
    dynamic,details=compute_block_gradients(lm,lce,labels,trainable_names=trainable_names,
                                          s9_names=shared_names,controller=controller)
    if arm=='cabg_lce':
        applied=dynamic;actual=details['budget']['lambda_final'];diagnostics=details['gradient']
    else:
        actual=0. if arm=='lm_only' else float(coefficient)
        applied,diagnostics=combine_and_clip_gradients(details['aggregate_lm'],details['aggregate_lce'],
            shared_names=shared_names,lambda_value=actual,rho_max=math.inf,max_norm=1.)
    details.update(applied_coefficient=actual,applied_gradient=diagnostics,arm=arm,
                   applied_dynamic_cap=arm=='cabg_lce')
    return applied,details

def apply_update(masters,optimizer,scheduler,applied,model):
    optimizer.zero_grad(set_to_none=True)
    for n,p in masters.items():p.grad=applied[n].to(p)
    lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
    named=dict(model.named_parameters())
    with torch.no_grad():
        for n,p in masters.items():
            if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite FP32 master')
            named[n].copy_(p.to(named[n]))
    optimizer.zero_grad(set_to_none=True)
    return lr
