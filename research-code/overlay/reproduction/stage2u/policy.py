"""The only intervention: choose the post-parent coefficient; retain shadow CABG."""
import math
from reproduction.stage2i.cabg_math import combine_and_clip_gradients
from reproduction.stage2l.controller import compute_block_gradients
from reproduction.stage2m.constants import TRAINABLE_NAMES,S9_NAMES
from reproduction.stage2u.common import ARMS

def selected_lambda(arm,budget,frozen):
    if arm not in ARMS:raise ValueError(arm)
    if not math.isfinite(frozen) or frozen<0:raise ValueError('Invalid pre-intervention ratio')
    return budget['lambda_final'] if arm=='dynamic' else min(frozen,budget['lambda_cap']) if arm=='frozen_ratio' else 0.

def gradients(lm,lce,labels,*,controller,arm,frozen):
    dynamic,details=compute_block_gradients(lm,lce,labels,trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=controller)
    coefficient=selected_lambda(arm,details['budget'],frozen)
    applied,diagnostics=combine_and_clip_gradients(details['aggregate_lm'],details['aggregate_lce'],shared_names=S9_NAMES,lambda_value=coefficient,rho_max=.2,max_norm=1.)
    if arm=='dynamic':
        import torch
        assert all(torch.equal(applied[n],dynamic[n]) for n in TRAINABLE_NAMES)
    details['gradient']=diagnostics
    details['policy']={'arm':arm,'frozen_ratio':frozen,'selected_lambda':coefficient,'shadow_lambda':details['budget']['lambda_final']}
    return applied,details
