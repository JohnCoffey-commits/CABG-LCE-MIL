"""Instrument installed FA entrypoints; optional explicit deterministic backward.

No backend, model math, optimizer, precision or gradient-collection change.
"""
import functools
import inspect
from collections import Counter

import torch

from reproduction.stage2i.rng_state import snapshot_rng_state, restore_rng_state, rng_state_fingerprint


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_tree(v) for v in x)
    return x


class AttentionExecution:
    def __init__(self, setting="original", microreplay=False):
        if setting not in ("original", "canonical_dormant", "canonical_flash_deterministic", "flash_deterministic"):
            raise RuntimeError("Unknown execution setting")
        self.setting = setting
        self.microreplay = microreplay
        self.counts = Counter()
        self.kernels = {}
        self.originals = []
        self.scope = "unassigned"
        self.deterministic = setting in ("canonical_flash_deterministic", "flash_deterministic")

    def install(self):
        from flash_attn import flash_attn_interface as interface
        from transformers import modeling_flash_attention_utils as utils
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as model
        self.interface = interface
        for module, name, route in ((utils, "flash_attn_func", "text_dense"), (utils, "flash_attn_varlen_func", "text_varlen"), (model, "flash_attn_varlen_func", "vision_varlen")):
            fn = getattr(module, name)
            signature = inspect.signature(fn)
            if "deterministic" not in signature.parameters:
                raise RuntimeError("Installed FA lacks explicit deterministic argument")
            def make_forward(fn, signature, route):
                @functools.wraps(fn)
                def forward(*args, **kwargs):
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    if self.deterministic:
                        bound.arguments["deterministic"] = True
                    flag = bool(bound.arguments["deterministic"])
                    self.counts[f"forward:{route}:deterministic={flag}"] += 1
                    return fn(*bound.args, **bound.kwargs)
                return forward
            self.originals.append((module, name, fn))
            setattr(module, name, make_forward(fn, signature, route))
        for name, index, kind in (("_wrapped_flash_attn_backward", 16, "dense"), ("_wrapped_flash_attn_varlen_backward", 20, "varlen")):
            fn = getattr(interface, name)
            def make_backward(fn, index, kind):
                def backward(*args, **kwargs):
                    if len(args) <= index or type(args[index]) is not bool:
                        raise RuntimeError("Installed backward signature changed")
                    flag = args[index]
                    if self.deterministic and not flag:
                        raise RuntimeError("Deterministic option did not reach backward")
                    self.counts[f"backward:{kind}:{self.scope}:deterministic={flag}"] += 1
                    result = fn(*args, **kwargs)
                    if self.microreplay and kind not in self.kernels:
                        before = snapshot_rng_state()
                        inputs = cpu_tree({"args": [a if j not in (6,7,8) else None for j,a in enumerate(args)], "kwargs": kwargs})
                        outputs = [cpu_tree({n: args[j] for n,j in (("dq",6),("dk",7),("dv",8))})]
                        for _ in range(2):
                            replay = list(args)
                            for j in (6,7,8):
                                replay[j] = torch.empty_like(args[j])
                            restore_rng_state(before)
                            fn(*replay, **kwargs)
                            outputs.append(cpu_tree({n: replay[j] for n,j in (("dq",6),("dk",7),("dv",8))}))
                        restore_rng_state(before)
                        if rng_state_fingerprint(snapshot_rng_state()) != rng_state_fingerprint(before):
                            raise RuntimeError("Kernel replay altered global RNG")
                        self.kernels[kind] = {"inputs": inputs, "outputs": outputs, "deterministic": flag, "scope": self.scope}
                    return result
                return backward
            self.originals.append((interface, name, fn))
            setattr(interface, name, make_backward(fn, index, kind))
        return self

    def report(self):
        backward = {k:v for k,v in self.counts.items() if k.startswith("backward:")}
        if not backward:
            raise RuntimeError("No actual FA backward observed")
        return {"setting": self.setting, "actual_calls": dict(self.counts), "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}

    def close(self):
        for module, name, fn in reversed(self.originals):
            setattr(module, name, fn)
        self.originals.clear()
