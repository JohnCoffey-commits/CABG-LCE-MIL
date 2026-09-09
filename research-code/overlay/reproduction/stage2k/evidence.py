"""D3R producer observation, gradient, and descriptive evidence helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from reproduction.stage2k.constants import GRADIENT_FLOOR
from reproduction.stage2k.manifest import ManifestContractError


class RawAttentionObserver:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.calls: list[torch.Tensor] = []
        self.handle: Any = None

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], torch.Tensor):
            raise ManifestContractError("D3R anomaly-attention output changed.")
        self.calls.append(output[1])
        if len(self.calls) > 2:
            raise ManifestContractError("D3R observed more than two A/N attention calls.")

    def __enter__(self):
        if self.handle is not None:
            raise ManifestContractError("D3R observer cannot be entered twice.")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def finalize(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(self.calls) != 2 or evidence.ndim != 3 or evidence.shape[0] != 1:
            raise ManifestContractError("D3R observer call count or evidence shape changed.")
        abnormal, normal = (value.reshape(evidence.shape) for value in self.calls)
        a_probability, n_probability = torch.sigmoid(abnormal), torch.sigmoid(normal)
        reconstruction = a_probability - n_probability
        if not torch.equal(reconstruction, evidence):
            maximum = float((reconstruction.float() - evidence.float()).abs().max().detach().cpu())
            raise ManifestContractError(f"D3R anomaly evidence reconstruction changed: {maximum}")
        return {
            "abnormal_raw": abnormal,
            "normal_raw": normal,
            "abnormal_probability": a_probability,
            "normal_probability": n_probability,
            "evidence": reconstruction,
        }


def gradient_rows(names: Sequence[str], gradients: Sequence[torch.Tensor | None]) -> list[dict[str, object]]:
    if len(names) != len(gradients) or len(set(names)) != len(names):
        raise ManifestContractError("D3R gradient schema is misaligned.")
    result = []
    for name, gradient in zip(names, gradients):
        if gradient is None:
            result.append({"name": name, "present": False, "finite": True, "effective": False, "max_abs": 0.0, "l2_norm": 0.0})
            continue
        value = gradient.detach().float()
        finite = bool(torch.isfinite(value).all().item())
        maximum = float(value.abs().max().cpu()) if finite else None
        norm = float(torch.linalg.vector_norm(value.reshape(-1)).cpu()) if finite else None
        result.append(
            {
                "name": name,
                "present": True,
                "finite": finite,
                "effective": bool(finite and maximum is not None and maximum > GRADIENT_FLOOR),
                "max_abs": maximum,
                "l2_norm": norm,
            }
        )
    return result


def global_norm(rows: Sequence[dict[str, object]], names: Sequence[str]) -> float:
    selected = set(names)
    square = 0.0
    for row in rows:
        if row["name"] in selected:
            value = float(row["l2_norm"] or 0.0)
            if not math.isfinite(value):
                raise ManifestContractError("D3R gradient norm is non-finite.")
            square += value * value
    return math.sqrt(square)


def summary(value: torch.Tensor) -> dict[str, object]:
    working = value.detach().float().reshape(-1)
    if working.numel() == 0 or not torch.isfinite(working).all():
        raise ManifestContractError("D3R summary input is empty or non-finite.")
    return {
        "count": int(working.numel()),
        "min": float(working.min().cpu()),
        "max": float(working.max().cpu()),
        "mean": float(working.mean().cpu()),
        "std_population": float(working.std(unbiased=False).cpu()),
    }


def tensor_payload(observed: dict[str, torch.Tensor], lce: dict[str, torch.Tensor | str]) -> dict[str, object]:
    tensor_shape = list(observed["abnormal_raw"].shape)
    return {
        "tensor_shape": tensor_shape,
        "raw_logits": {
            "serialization": "float32_values_from_bfloat16_forward_tensors",
            "abnormal_flat": observed["abnormal_raw"].detach().float().cpu().reshape(-1).tolist(),
            "normal_flat": observed["normal_raw"].detach().float().cpu().reshape(-1).tolist(),
        },
        "producer_formula": {
            "gap_flat": lce["raw_gap"].detach().cpu().reshape(-1).tolist(),
            "contrast_flat": lce["contrast"].detach().cpu().reshape(-1).tolist(),
            "derivative_flat": lce["softsign_derivative"].detach().cpu().reshape(-1).tolist(),
            "layer_mean_flat": lce["layer_mean"].detach().cpu().reshape(-1).tolist(),
            "weights_flat": lce["spatial_weights"].detach().cpu().reshape(-1).tolist(),
        },
        "descriptive": {
            "abnormal_raw": summary(observed["abnormal_raw"]),
            "normal_raw": summary(observed["normal_raw"]),
            "abnormal_probability": summary(observed["abnormal_probability"]),
            "normal_probability": summary(observed["normal_probability"]),
            "raw_gap": summary(lce["raw_gap"]),
            "contrast": summary(lce["contrast"]),
            "softsign_derivative": summary(lce["softsign_derivative"]),
        },
    }
