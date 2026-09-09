#!/usr/bin/env python3

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from reproduction.stage2f.training_diagnostics import (
    Stage2FActivationDiagnosticCallback,
)


class FakeAttention(nn.Module):
    def forward(self, scores):
        return scores.mean(dim=-1, keepdim=True), scores


class FakeAnomalyQformer(nn.Module):
    def __init__(self, multiscale: bool):
        super().__init__()
        self.anomaly_attention = FakeAttention()
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.fusion_gate = nn.Conv2d(4, 1, 1) if multiscale else None


class FakeModel(nn.Module):
    def __init__(self, multiscale: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.anomaly_qformer = FakeAnomalyQformer(multiscale)


def exercise(mode: str, output_path: Path):
    model = FakeModel(mode == "multiscale")
    anomaly = model.model.visual.anomaly_qformer
    abnormal = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    normal = torch.tensor([[[0.5, 1.5], [2.5, 3.5]]])
    expected_abnormal = anomaly.anomaly_attention(abnormal)
    expected_normal = anomaly.anomaly_attention(normal)
    gate_input = None
    expected_gate = None
    if mode == "multiscale":
        gate_input = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
        expected_gate = anomaly.fusion_gate(gate_input).detach().clone()
    rng_before = torch.get_rng_state().clone()
    callback = Stage2FActivationDiagnosticCallback(
        model=model,
        output_path=str(output_path),
        mode=mode,
        expected_steps=1,
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    state = SimpleNamespace(global_step=0)
    callback.on_train_begin(None, state, None)
    callback.on_step_begin(None, state, None)
    observed_abnormal = anomaly.anomaly_attention(abnormal)
    observed_normal = anomaly.anomaly_attention(normal)
    assert torch.equal(observed_abnormal[0], expected_abnormal[0])
    assert torch.equal(observed_abnormal[1], expected_abnormal[1])
    assert torch.equal(observed_normal[0], expected_normal[0])
    assert torch.equal(observed_normal[1], expected_normal[1])
    if mode == "multiscale":
        observed_gate = anomaly.fusion_gate(gate_input)
        assert torch.equal(observed_gate, expected_gate)
    callback.on_pre_optimizer_step(None, state, None)
    callback.on_train_end(None, SimpleNamespace(global_step=1), None)
    assert torch.equal(rng_before, torch.get_rng_state())
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    step = [row for row in rows if row["event"] == "activation_diagnostic_step"][0]
    assert step["attention_delta_observations"]
    if mode == "multiscale":
        assert step["lambda_observations"]
    else:
        assert not step["lambda_observations"]
    return observed_normal


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exercise("multiscale", root / "a3.jsonl")
        exercise("single", root / "b0.jsonl")
    print("training diagnostics output/RNG invariance: PASS")


if __name__ == "__main__":
    main()
