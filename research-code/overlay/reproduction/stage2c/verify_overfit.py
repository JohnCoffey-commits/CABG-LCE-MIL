#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def main() -> None:
    output_dir = Path(
        os.environ.get(
            "MEDIC_AD_OVERFIT_OUTPUT",
            "/home/outputs/medic-ad/training-overfit/stage1-single-abnormal-20step-v1",
        )
    )
    audit_path = Path(
        os.environ.get(
            "MEDIC_AD_GRADIENT_AUDIT",
            "/home/logs/medic-ad/training-overfit/stage1-gradient-audit.jsonl",
        )
    )
    loss_trace_path = Path(
        os.environ.get(
            "MEDIC_AD_LOSS_TRACE",
            "/home/logs/medic-ad/training-overfit/stage1-loss-trace.jsonl",
        )
    )
    trainer_state_path = output_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise FileNotFoundError(trainer_state_path)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if not loss_trace_path.is_file():
        raise FileNotFoundError(loss_trace_path)

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != 20:
        raise RuntimeError(f"Expected global_step=20, got {state.get('global_step')}")

    step_records = [
        record
        for record in state.get("log_history", [])
        if "loss" in record and "grad_norm" in record and "step" in record
    ]
    if [record["step"] for record in step_records] != list(range(1, 21)):
        raise RuntimeError("Expected one loss/gradient record for every optimizer step 1..20.")

    rounded_losses = [float(record["loss"]) for record in step_records]
    grad_norms = [float(record["grad_norm"]) for record in step_records]
    if not all(math.isfinite(value) for value in rounded_losses + grad_norms):
        raise RuntimeError("Loss or gradient norm contains a non-finite value.")
    if not all(value > 0.0 for value in grad_norms):
        raise RuntimeError("At least one optimizer step has a zero gradient norm.")

    loss_trace = [
        json.loads(line)
        for line in loss_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record.get("global_step_before_update") for record in loss_trace] != list(range(20)):
        raise RuntimeError("Expected one unrounded loss record before every optimizer step 0..19.")
    losses = [float(record["loss"]) for record in loss_trace]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Unrounded loss trace contains a non-finite value.")

    first_window_mean = mean(losses[:5])
    last_window_mean = mean(losses[-5:])
    if not last_window_mean < first_window_mean:
        raise RuntimeError(
            f"Overfit loss did not improve: first5={first_window_mean}, last5={last_window_mean}"
        )

    audit_records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start_records = [record for record in audit_records if record.get("event") == "gradient_audit_start"]
    gradient_steps = [record for record in audit_records if record.get("event") == "gradient_audit_step"]
    end_records = [record for record in audit_records if record.get("event") == "gradient_audit_end"]
    if len(start_records) != 1 or len(end_records) != 1:
        raise RuntimeError("Gradient audit must contain exactly one start and one end record.")
    if [record.get("optimizer_step") for record in gradient_steps] != [5, 10, 15, 20]:
        raise RuntimeError("Expected gradient audits at optimizer steps 5, 10, 15, and 20.")

    expected_tensors = 21
    expected_elements = 29_561_345
    final_audit = end_records[0]
    if final_audit.get("status") != "SUCCESS":
        raise RuntimeError(f"Gradient audit failed: {final_audit}")
    if final_audit.get("trainable_parameter_tensors") != expected_tensors:
        raise RuntimeError(f"Unexpected trainable tensor count: {final_audit}")
    if final_audit.get("trainable_parameter_elements") != expected_elements:
        raise RuntimeError(f"Unexpected trainable element count: {final_audit}")

    forbidden_model_files = sorted(output_dir.glob("*.safetensors"))
    if forbidden_model_files:
        raise RuntimeError(f"Diagnostic run unexpectedly saved full model weights: {forbidden_model_files}")

    result = {
        "status": "SUCCESS",
        "global_step": state["global_step"],
        "losses": losses,
        "rounded_trainer_losses": rounded_losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "first_5_step_mean_loss": first_window_mean,
        "last_5_step_mean_loss": last_window_mean,
        "last_to_first_window_ratio": last_window_mean / first_window_mean,
        "gradient_norm_min": min(grad_norms),
        "gradient_norm_max": max(grad_norms),
        "gradient_audit_steps": [record["optimizer_step"] for record in gradient_steps],
        "gradient_audit_status": final_audit["status"],
        "trainable_parameter_tensors": final_audit["trainable_parameter_tensors"],
        "trainable_parameter_elements": final_audit["trainable_parameter_elements"],
        "full_model_saved": False,
        "output_dir": str(output_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
