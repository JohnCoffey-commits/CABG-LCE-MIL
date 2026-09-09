"""Temporary read-only observer for A/N attention probabilities in Gate D3."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from reproduction.stage2i.cabg_math import CABGContractError


class AnomalyAttentionObserver:
    """Capture exactly the existing abnormal then normal attention-score calls."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.calls: list[torch.Tensor] = []
        self._handle: Any = None

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], torch.Tensor):
            raise CABGContractError("D3 anomaly-attention hook observed an invalid output.")
        self.calls.append(output[1])
        if len(self.calls) > 2:
            raise CABGContractError("D3 observed more than two anomaly-attention calls.")

    def __enter__(self):
        if self._handle is not None:
            raise CABGContractError("D3 observer cannot be entered twice.")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def finalize(self, anomaly_evidence: torch.Tensor) -> dict[str, torch.Tensor | float | bool]:
        if len(self.calls) != 2:
            raise CABGContractError(f"D3 requires exactly two anomaly-attention calls, got {len(self.calls)}.")
        if anomaly_evidence.ndim != 3 or anomaly_evidence.shape[0] != 1:
            raise CABGContractError("D3 observer requires one-image evidence [1,layers,positions].")
        shape = anomaly_evidence.shape
        abnormal_scores, normal_scores = self.calls
        expected_elements = int(anomaly_evidence.numel())
        if abnormal_scores.numel() != expected_elements or normal_scores.numel() != expected_elements:
            raise CABGContractError("D3 observed A/N score shape is incompatible with returned evidence.")
        abnormal = F.sigmoid(abnormal_scores).reshape(shape)
        normal = F.sigmoid(normal_scores).reshape(shape)
        reconstructed = abnormal - normal
        exact = torch.equal(reconstructed, anomaly_evidence)
        max_abs = float((reconstructed.float() - anomaly_evidence.float()).abs().max().detach().cpu().item())
        if not exact:
            raise CABGContractError(
                f"D3 A/N reconstruction is not exact for returned evidence; max_abs={max_abs}."
            )
        return {
            "abnormal_probability": abnormal,
            "normal_probability": normal,
            "reconstructed_evidence": reconstructed,
            "reconstruction_exact": exact,
            "reconstruction_max_abs": max_abs,
        }
