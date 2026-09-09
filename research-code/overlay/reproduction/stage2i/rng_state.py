"""RNG snapshot, restoration, and fingerprint utilities."""

from __future__ import annotations

import hashlib
import pickle
import random
from typing import Any, Mapping

import numpy as np
import torch

from reproduction.stage2i.cabg_math import CABGContractError


RNG_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})


def snapshot_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if set(state) != RNG_KEYS:
        raise CABGContractError("RNG state schema mismatch.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise CABGContractError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        if len(cuda_states) != torch.cuda.device_count():
            raise CABGContractError("CUDA RNG device count changed.")
        torch.cuda.set_rng_state_all(cuda_states)


def rng_state_fingerprint(state: Mapping[str, Any]) -> str:
    if set(state) != RNG_KEYS:
        raise CABGContractError("RNG state schema mismatch.")
    digest = hashlib.sha256()
    digest.update(pickle.dumps(state["python"], protocol=5))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("utf-8"))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(pickle.dumps(numpy_state[2:], protocol=5))
    digest.update(state["torch_cpu"].detach().cpu().numpy().tobytes())
    for value in state["torch_cuda"]:
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
