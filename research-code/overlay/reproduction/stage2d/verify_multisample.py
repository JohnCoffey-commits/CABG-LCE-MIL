#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path


def main() -> None:
    run_id = os.environ.get("MEDIC_AD_RUN_ID", "run-a")
    output_dir = Path(os.environ["MEDIC_AD_MULTISAMPLE_OUTPUT"])
    gradient_audit_path = Path(os.environ["MEDIC_AD_GRADIENT_AUDIT"])
    loss_trace_path = Path(os.environ["MEDIC_AD_LOSS_TRACE"])
    trainer_state_path = output_dir / "trainer_state.json"
    for required in (trainer_state_path, gradient_audit_path, loss_trace_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != 8:
        raise RuntimeError(f"Expected global_step=8, got {state.get('global_step')}")
    train_records = [
        record
        for record in state.get("log_history", [])
        if "loss" in record and "grad_norm" in record and "step" in record
    ]
    if [record["step"] for record in train_records] != list(range(1, 9)):
        raise RuntimeError("Expected train records for optimizer steps 1..8.")
    gradient_norms = [float(record["grad_norm"]) for record in train_records]
    if not all(math.isfinite(value) and value > 0.0 for value in gradient_norms):
        raise RuntimeError("Training gradient norms must be finite and positive.")

    eval_records = [record for record in state.get("log_history", []) if "eval_loss" in record]
    if [record.get("step") for record in eval_records] != [4, 8]:
        raise RuntimeError(f"Expected evaluation at steps 4 and 8; got {eval_records}")
    eval_losses = [float(record["eval_loss"]) for record in eval_records]
    if not all(math.isfinite(value) for value in eval_losses):
        raise RuntimeError("Evaluation loss contains a non-finite value.")

    loss_trace = [
        json.loads(line)
        for line in loss_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record.get("global_step_before_update") for record in loss_trace] != list(range(8)):
        raise RuntimeError("Expected one unrounded training loss before steps 0..7.")
    losses = [float(record["loss"]) for record in loss_trace]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Training loss trace contains a non-finite value.")

    audit_records = [
        json.loads(line)
        for line in gradient_audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_steps = [record for record in audit_records if record.get("event") == "gradient_audit_step"]
    audit_end = [record for record in audit_records if record.get("event") == "gradient_audit_end"]
    if [record.get("optimizer_step") for record in audit_steps] != [4, 8]:
        raise RuntimeError("Expected gradient audits at optimizer steps 4 and 8.")
    if len(audit_end) != 1 or audit_end[0].get("status") != "SUCCESS":
        raise RuntimeError(f"Gradient audit failed: {audit_end}")
    if audit_end[0].get("trainable_parameter_tensors") != 21:
        raise RuntimeError("Unexpected trainable tensor count.")
    if audit_end[0].get("trainable_parameter_elements") != 29_561_345:
        raise RuntimeError("Unexpected trainable parameter element count.")

    model_files = sorted(output_dir.glob("*.safetensors"))
    if model_files:
        raise RuntimeError(f"Diagnostic run unexpectedly saved full weights: {model_files}")

    result = {
        "status": "SUCCESS",
        "run_id": run_id,
        "full_determinism": os.environ.get("MEDIC_AD_FULL_DETERMINISM", "False") == "True",
        "global_step": state["global_step"],
        "training_losses": losses,
        "rounded_training_losses": [float(record["loss"]) for record in train_records],
        "gradient_norms": gradient_norms,
        "eval_steps": [record["step"] for record in eval_records],
        "eval_losses": eval_losses,
        "first_eval_loss": eval_losses[0],
        "final_eval_loss": eval_losses[-1],
        "gradient_audit_steps": [record["optimizer_step"] for record in audit_steps],
        "gradient_audit_status": audit_end[0]["status"],
        "trainable_parameter_tensors": audit_end[0]["trainable_parameter_tensors"],
        "trainable_parameter_elements": audit_end[0]["trainable_parameter_elements"],
        "full_model_saved": False,
        "output_dir": str(output_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
