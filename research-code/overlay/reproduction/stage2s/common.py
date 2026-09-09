"""Budget, runtime state and first-four-training-block contracts."""
import fcntl
import json
import os
import signal
import time
from pathlib import Path

import torch

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2i.rng_state import snapshot_rng_state, restore_rng_state, rng_state_fingerprint
from reproduction.stage2l.checkpoint import tensor_inventory
from reproduction.stage2q.execution import cpu_tree

BASELINE = "8305fdbf52ce5e17780ef6fc7f08d02742f94d6a"
CACHES = ("rope_deltas", "last_visual_tokens", "last_visual_grid_thw",
          "_last_segmentation_preview", "middle_hidden_states", "grid_thw")


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


class Budget:
    def __init__(self, root):
        self.path = Path(root)/"budget.json"
        if not self.path.exists():
            write(self.path, {"started_epoch": time.time(), "limit_seconds": 5400,
                              "real_updates_reserved": 0, "block_equivalents": 0, "events": []})
        state = json.loads(self.path.read_text())
        remaining = 5400 - (time.time()-state["started_epoch"])
        if remaining <= 0:
            raise RuntimeError("Route A wall budget exhausted")
        def expired(*args):
            raise TimeoutError("Route A cumulative 5400 second budget exhausted")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(max(1, int(remaining)))

    def reserve(self, kind, context, amount=1):
        with self.path.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            state = json.load(f)
            key, limit = ("real_updates_reserved", 8) if kind == "update" else ("block_equivalents", 16)
            if state[key]+amount > limit or time.time()-state["started_epoch"] >= 5400:
                raise RuntimeError("Route A hard budget would be exceeded")
            state[key] += amount
            state["events"].append({"kind": kind, "context": context, "amount": amount, "epoch": time.time()})
            f.seek(0); json.dump(state, f, indent=2); f.truncate(); f.flush(); os.fsync(f.fileno())


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value


def runtime_state(model):
    return {"buffers": cpu_tree(dict(model.named_buffers())),
            "modes": {n: m.training for n, m in model.named_modules()},
            "caches": {n: {a: cpu_tree(getattr(m, a)) for a in CACHES if hasattr(m, a)}
                       for n, m in model.named_modules() if any(hasattr(m, a) for a in CACHES)}}


def restore_runtime(model, state):
    buffers = dict(model.named_buffers())
    if set(buffers) != set(state["buffers"]):
        raise RuntimeError("Buffer schema changed")
    with torch.no_grad():
        for n, b in buffers.items(): b.copy_(state["buffers"][n].to(b))
    device = next(model.parameters()).device
    for n, m in model.named_modules():
        m.training = state["modes"][n]
        for attr in CACHES:
            if attr in state["caches"].get(n, {}):
                setattr(m, attr, to_device(state["caches"][n][attr], device))
            elif hasattr(m, attr): delattr(m, attr)


def state_digest(value):
    # Structural scalars plus tensor identities; includes caches and module modes.
    def tree(x):
        if isinstance(x, torch.Tensor): return {"tensor": tensor_inventory({"v": x})}
        if isinstance(x, dict): return {str(k): tree(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [tree(v) for v in x]
        return x
    return canonical_json_sha256(tree(value))


def device_inventory(value):
    result={}
    def walk(x,path):
        if isinstance(x,torch.Tensor): result[path]=str(x.device)
        elif isinstance(x,dict):
            for k,v in x.items():walk(v,path+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            for i,v in enumerate(x):walk(v,path+'/'+str(i))
    walk(value,'');return result


def runtime_devices(model):
    return device_inventory({'buffers':dict(model.named_buffers()),
       'caches':{n:{a:getattr(m,a) for a in CACHES if hasattr(m,a)} for n,m in model.named_modules()}})


def setup_flags():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS setting must precede process launch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def first_step_prediction(initial, applied):
    # Independent analytic first AdamW step; no optimizer.step or model update.
    return {n: (v.double()-1e-4*applied[n].double()/(applied[n].double().abs()+1e-8)).float()
            for n, v in initial.items()}
