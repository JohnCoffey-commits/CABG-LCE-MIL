import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer, TrainerCallback

from reproduction.stage2h.evidence import compute_evidence_margin_loss


class AnomalyEvidenceTrainer(Trainer):
    """One-forward Trainer shared by B0-AS and LAD-MIL v2."""

    def __init__(
        self,
        *args,
        evidence_loss_weight: float,
        evidence_trace_output: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evidence_loss_weight = float(evidence_loss_weight)
        if self.evidence_loss_weight < 0 or not math.isfinite(self.evidence_loss_weight):
            raise ValueError("evidence_loss_weight must be finite and non-negative.")
        self.evidence_trace_output = evidence_trace_output

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = dict(inputs)
        anomaly_labels = inputs.pop("anomaly_labels", None)
        if anomaly_labels is None:
            raise RuntimeError("AnomalyEvidenceTrainer requires anomaly_labels.")
        inputs["tune_mode"] = "default"
        inputs["return_anomaly_evidence"] = True
        started = time.perf_counter()
        outputs = model(**inputs)
        forward_seconds = time.perf_counter() - started
        lm_loss = outputs.loss
        if lm_loss is None or not torch.isfinite(lm_loss):
            raise RuntimeError("Stage 2H LM loss is missing or non-finite.")
        evidence_result = compute_evidence_margin_loss(outputs.anomaly_evidence, anomaly_labels)
        evidence_loss = evidence_result["loss"]
        total_loss = (
            lm_loss
            if self.evidence_loss_weight == 0.0
            else lm_loss + self.evidence_loss_weight * evidence_loss
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError("Stage 2H total loss is non-finite.")

        saturation_count = int(evidence_result["saturation_count"].detach().cpu().item())
        saturation_total = int(evidence_result["saturation_total"].detach().cpu().item())
        record = {
            "global_step_before_update": int(self.state.global_step),
            "evidence_loss_weight": self.evidence_loss_weight,
            "lm_loss": float(lm_loss.detach().float().cpu().item()),
            "evidence_loss": float(evidence_loss.detach().float().cpu().item()),
            "total_loss": float(total_loss.detach().float().cpu().item()),
            "evidence_k": int(evidence_result["k"]),
            "evidence_positions": int(evidence_result["positions"]),
            "evidence_min": float(outputs.anomaly_evidence.detach().float().min().cpu().item()),
            "evidence_max": float(outputs.anomaly_evidence.detach().float().max().cpu().item()),
            "evidence_mean": float(outputs.anomaly_evidence.detach().float().mean().cpu().item()),
            "evidence_saturation_count": saturation_count,
            "evidence_saturation_total": saturation_total,
            "evidence_saturation_proportion": saturation_count / saturation_total,
            "forward_wall_seconds": forward_seconds,
        }
        self.log(
            {
                "train_lm_loss": record["lm_loss"],
                "train_evidence_loss": record["evidence_loss"],
                "train_total_loss": record["total_loss"],
            }
        )
        if self.evidence_trace_output and model.training and self.is_world_process_zero():
            output_path = Path(self.evidence_trace_output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return (total_loss, outputs) if return_outputs else total_loss


class Stage2HStepTimingCallback(TrainerCallback):
    def __init__(self, output_path: str):
        self.output_path = Path(output_path).expanduser()
        self.started = None

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is None:
            raise RuntimeError("Stage 2H step timer has no matching start.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"global_step": int(state.global_step), "wall_seconds": elapsed},
                    sort_keys=True,
                )
                + "\n"
            )
        self.started = None
