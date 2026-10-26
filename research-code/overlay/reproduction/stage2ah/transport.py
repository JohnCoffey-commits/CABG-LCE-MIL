"""A parameter-free differentiable MIL value path; original query path retained."""
import inspect
import math
import textwrap
import types

import torch

ARMS = ('aligned', 'shuffled')


class Transport:
    def __init__(self, model, arm):
        assert arm in ARMS
        self.module = model.visual.anomaly_qformer
        self.arm = arm
        self.enabled = True
        self.capture = False
        self.trace = {}
        self.calls = 0
        self.gradient_events = 0
        self.nonzero_gradient_events = 0
        self.old = self.module.forward

    def record(self, name, value):
        if self.capture:
            self.trace[name] = value.detach().float().cpu().numpy().copy()

    def aggregate(self, module, query, values, abnormal, normal):
        native, scores = module.q_former(query, values, values)
        if not self.enabled:
            return native, scores
        assert abnormal.shape == normal.shape == (1, 4, 1, 1024)
        assert values.shape == (1, 256, 3584) and query.shape == (1, 16, 1280)
        gap = abnormal.float() - normal.float()
        contrast = (gap / (1 + gap.abs())).mean(dim=1).squeeze(1)
        weight32 = torch.softmax(contrast * math.log(1024), dim=-1)
        # Sum each aligned 2x2 cell: preserve probability mass on the 16x16 K/V grid.
        weight = weight32.reshape(1, 16, 2, 16, 2).sum(dim=(2, 4)).reshape(1, 256)
        if self.arm == 'shuffled':
            permutation = (torch.arange(256, device=values.device) * 101 + 37) % 256
            weight = weight[:, permutation]
        if torch.is_grad_enabled() and weight.requires_grad:
            def backward(gradient):
                assert torch.isfinite(gradient).all(), 'Transport gradient nonfinite'
                self.gradient_events += 1
                self.nonzero_gradient_events += int(bool(torch.count_nonzero(gradient)))
            weight.register_hook(backward)
        pooled = torch.matmul(weight[:, None].to(values.dtype), values)
        output = ((native.float() + pooled.float()) * .5).to(native.dtype)
        self.calls += 1
        for key, value in [('abnormal', abnormal.squeeze(2)), ('normal', normal.squeeze(2)),
                           ('weight32', weight32), ('weight16', weight), ('native', native),
                           ('pooled', pooled), ('output', output), ('values', values), ('attention_logits', scores)]:
            self.record(key, value)
        return output, scores

    def install(self):
        source = textwrap.dedent(inspect.getsource(type(self.module).forward))
        needle = 'self.q_former(anomaly_query, merged_seq, merged_seq)'
        assert source.count(needle) == 1
        source = source.replace(needle, '_transport(self, anomaly_query, merged_seq, abnormal_map, normal_map)')
        namespace = dict(vars(inspect.getmodule(type(self.module))))
        namespace['_transport'] = self.aggregate
        exec(compile(source, '<stage2ah-frozen-transport>', 'exec'), namespace)
        self.module.forward = types.MethodType(namespace['forward'], self.module)
        return self

    def report(self):
        return dict(arm=self.arm, calls=self.calls, gradient_events=self.gradient_events,
                    nonzero_gradient_events=self.nonzero_gradient_events, additional_parameters=0)

    def close(self):
        self.module.forward = self.old
