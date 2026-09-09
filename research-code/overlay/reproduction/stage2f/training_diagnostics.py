#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import TrainerCallback

from reproduction.stage2f.adapter_schema import normalize_query_mode


def tensor_summary(tensor: torch.Tensor) -> Dict:
    detached = tensor.detach().float()
    finite = bool(torch.isfinite(detached).all().item())
    if detached.numel() == 0:
        return {
            "elements": 0,
            "finite": finite,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
        }
    return {
        "elements": int(detached.numel()),
        "finite": finite,
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "abs_max": float(detached.abs().max().item()),
    }


def locate_anomaly_qformer(model):
    bare_model = getattr(model, "module", model)
    core_model = getattr(bare_model, "model", bare_model)
    visual = getattr(core_model, "visual", None)
    anomaly_qformer = getattr(visual, "anomaly_qformer", None)
    if anomaly_qformer is None:
        raise RuntimeError("Stage 2F activation diagnostics could not locate AnomalyQformer.")
    return anomaly_qformer


class Stage2FActivationDiagnosticCallback(TrainerCallback):
    """Observation-only hooks for per-step FB-MAQ and attention diagnostics."""

    def __init__(self, model, output_path: str, mode: str, expected_steps: int):
        super().__init__()
        self.output_path = Path(output_path).expanduser()
        if self.output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite activation diagnostics: {self.output_path}"
            )
        self.mode = normalize_query_mode(mode)
        self.expected_steps = int(expected_steps)
        if self.expected_steps <= 0:
            raise ValueError("Activation diagnostics require a positive expected step count.")
        self.anomaly_qformer = locate_anomaly_qformer(model)
        self.current_step: Optional[int] = None
        self.lambda_observations: List[Dict] = []
        self.delta_observations: List[Dict] = []
        self._pending_abnormal_attention: Optional[torch.Tensor] = None
        self._written_steps: List[int] = []
        self._handles = []
        self._handles.append(
            self.anomaly_qformer.anomaly_attention.register_forward_hook(
                self._attention_hook
            )
        )
        if self.mode == "multiscale":
            if self.anomaly_qformer.fusion_gate is None:
                raise RuntimeError("A3 activation diagnostics require fusion_gate.")
            self._handles.append(
                self.anomaly_qformer.fusion_gate.register_forward_hook(self._gate_hook)
            )
        elif self.anomaly_qformer.fusion_gate is not None:
            raise RuntimeError("B0 activation diagnostics found an unexpected fusion_gate.")

    def _write(self, record: Dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _is_recording(self, module) -> bool:
        return self.current_step is not None and bool(module.training)

    def _gate_hook(self, module, inputs, output) -> None:
        if not self._is_recording(module):
            return
        fusion_lambda = torch.sigmoid(output.detach().float())
        self.lambda_observations.append(tensor_summary(fusion_lambda))

    def _attention_hook(self, module, inputs, output) -> None:
        if not self._is_recording(module):
            return
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("Anomaly attention hook received an unexpected output.")
        attention_probability = torch.sigmoid(output[1].detach().float())
        if self._pending_abnormal_attention is None:
            self._pending_abnormal_attention = attention_probability
            return
        normal_attention = attention_probability
        delta = (
            self._pending_abnormal_attention - normal_attention
        ) * self.anomaly_qformer.gate_scale.detach().float()
        self.delta_observations.append(tensor_summary(delta))
        self._pending_abnormal_attention = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(
            {
                "event": "activation_diagnostic_start",
                "anomaly_query_mode": self.mode,
                "expected_steps": self.expected_steps,
                "observation_only": True,
            }
        )

    def on_step_begin(self, args, state, control, **kwargs):
        self.current_step = int(getattr(state, "global_step", 0)) + 1
        self.lambda_observations = []
        self.delta_observations = []
        self._pending_abnormal_attention = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.current_step is None:
            raise RuntimeError("Activation diagnostics have no active optimizer step.")
        if self._pending_abnormal_attention is not None:
            raise RuntimeError("Anomaly attention diagnostics observed an unpaired attention call.")
        if not self.delta_observations:
            raise RuntimeError("No training attention-delta observation was captured.")
        if self.mode == "multiscale" and not self.lambda_observations:
            raise RuntimeError("No A3 fusion-lambda observation was captured.")
        if self.mode == "single" and self.lambda_observations:
            raise RuntimeError("B0 unexpectedly produced fusion-lambda observations.")
        gate_parameters = None
        if self.mode == "multiscale":
            gate_parameters = {
                "weight": tensor_summary(self.anomaly_qformer.fusion_gate.weight),
                "bias": tensor_summary(self.anomaly_qformer.fusion_gate.bias),
            }
        self._write(
            {
                "event": "activation_diagnostic_step",
                "optimizer_step": self.current_step,
                "anomaly_query_mode": self.mode,
                "lambda_observations": self.lambda_observations,
                "attention_delta_observations": self.delta_observations,
                "gate_parameters_before_update": gate_parameters,
            }
        )
        self._written_steps.append(self.current_step)

    def on_train_end(self, args, state, control, **kwargs):
        expected = list(range(1, self.expected_steps + 1))
        status = "SUCCESS" if self._written_steps == expected else "FAILED"
        self._write(
            {
                "event": "activation_diagnostic_end",
                "anomaly_query_mode": self.mode,
                "observed_steps": self._written_steps,
                "status": status,
            }
        )
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if status != "SUCCESS":
            raise RuntimeError(
                "Activation diagnostic steps are incomplete: "
                f"expected {expected}, got {self._written_steps}"
            )
