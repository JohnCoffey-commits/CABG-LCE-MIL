"""Read-only capture of the two anomaly-attention score tensors."""

from __future__ import annotations

from typing import Any

import torch

from reproduction.stage2i.cabg_math import CABGContractError


class RawAnomalyAttentionObserver:
    """Capture abnormal and normal pre-sigmoid scores without detaching them."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.calls: list[torch.Tensor] = []
        self._handle: Any = None

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], torch.Tensor):
            raise CABGContractError("R0 anomaly-attention hook observed an invalid output.")
        self.calls.append(output[1])
        if len(self.calls) > 2:
            raise CABGContractError("R0 observed more than two anomaly-attention calls.")

    def __enter__(self):
        if self._handle is not None:
            raise CABGContractError("R0 observer cannot be entered twice.")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def finalize(self, anomaly_evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(self.calls) != 2:
            raise CABGContractError(f"R0 requires exactly two anomaly-attention calls, got {len(self.calls)}.")
        if anomaly_evidence.ndim != 3 or anomaly_evidence.shape[0] != 1:
            raise CABGContractError("R0 observer requires one-image evidence [1,layers,positions].")
        abnormal_score, normal_score = self.calls
        expected = int(anomaly_evidence.numel())
        if abnormal_score.numel() != expected or normal_score.numel() != expected:
            raise CABGContractError("R0 A/N raw-score shapes do not match returned evidence.")
        abnormal_raw = abnormal_score.reshape(anomaly_evidence.shape)
        normal_raw = normal_score.reshape(anomaly_evidence.shape)
        abnormal_probability = torch.sigmoid(abnormal_raw)
        normal_probability = torch.sigmoid(normal_raw)
        reconstructed = abnormal_probability - normal_probability
        if not torch.equal(reconstructed, anomaly_evidence):
            maximum = float((reconstructed.float() - anomaly_evidence.float()).abs().max().detach().cpu())
            raise CABGContractError(f"R0 A/N reconstruction is not exact; max_abs={maximum}.")
        return {
            "abnormal_raw": abnormal_raw,
            "normal_raw": normal_raw,
            "raw_gap": abnormal_raw - normal_raw,
            "abnormal_probability": abnormal_probability,
            "normal_probability": normal_probability,
            "evidence": reconstructed,
        }
