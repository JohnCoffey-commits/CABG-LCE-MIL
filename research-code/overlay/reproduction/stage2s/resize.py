"""Observe the actually called bilinear operation without changing its output."""
import torch
import torch.nn.functional as F

NATIVE_INTERPOLATE = F.interpolate


class DeterministicBilinear(torch.autograd.Function):
    """Native forward, PyTorch deterministic VJP of the same linear resize.

    Keep BF16 input/output/gradient, size and align_corners unchanged. The VJP
    does not depend on input values, so a zero input avoids retaining activations.
    Determinism is scoped to this VJP; unrelated model operations are unchanged.
    """
    @staticmethod
    def forward(ctx, x, size):
        ctx.shape, ctx.dtype, ctx.device, ctx.size = x.shape, x.dtype, x.device, size
        return NATIVE_INTERPOLATE(x, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def backward(ctx, grad_output):
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.enable_grad():
                x = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device, requires_grad=True)
                y = NATIVE_INTERPOLATE(x, size=ctx.size, mode='bilinear', align_corners=False)
                dx, = torch.autograd.grad(y, x, grad_output, create_graph=False)
            return dx, None
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)


class ResizeProbe:
    def __init__(self, deterministic=False):
        self.original = F.interpolate
        self.captures = {}
        self.scope = "forward"
        self.capture = False
        self.calls = []
        self.deterministic = deterministic

    def install(self):
        def observed(x, *args, **kwargs):
            target = x.ndim == 4 and kwargs.get('mode') == 'bilinear' and x.requires_grad
            if target and self.deterministic:
                if args or set(kwargs) != {'size','mode','align_corners'} or kwargs['align_corners'] is not False:
                    raise RuntimeError('Unregistered bilinear call signature')
                y = DeterministicBilinear.apply(x, kwargs['size'])
            else:
                y = self.original(x, *args, **kwargs)
            if x.ndim == 4 and kwargs.get("mode") == "bilinear" and x.requires_grad:
                shape = tuple(x.shape)
                self.calls.append({"input": list(shape), "output": list(y.shape),
                                   "align_corners": kwargs.get("align_corners")})
                if self.capture:
                    saved = x.detach().cpu().clone()
                    output = y.detach().cpu().clone()
                    size = tuple(y.shape[-2:])
                    def hook(g):
                        key = f"{shape}:{size}:{self.scope}"
                        if key not in self.captures:
                            self.captures[key] = {"input": saved, "grad_output": g.detach().cpu().clone(),
                                                  "size": size, "scope": self.scope,
                                                  "output": output}
                        return g
                    y.register_hook(hook)
            return y
        F.interpolate = observed
        return self

    def close(self):
        F.interpolate = self.original
