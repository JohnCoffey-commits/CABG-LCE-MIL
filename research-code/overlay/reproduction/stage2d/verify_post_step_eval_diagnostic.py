#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path


def main() -> None:
    output_dir = Path(os.environ["MEDIC_AD_POST_STEP_DIAGNOSTIC_OUTPUT"])
    state_path = output_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != 1:
        raise RuntimeError(f"Expected global_step=1, got {state.get('global_step')}")
    train_records = [
        record
        for record in state.get("log_history", [])
        if "loss" in record and "grad_norm" in record
    ]
    eval_records = [record for record in state.get("log_history", []) if "eval_loss" in record]
    if len(train_records) != 1 or train_records[0].get("step") != 1:
        raise RuntimeError(f"Unexpected training records: {train_records}")
    if len(eval_records) != 1 or eval_records[0].get("step") != 1:
        raise RuntimeError(f"Unexpected evaluation records: {eval_records}")
    loss = float(train_records[0]["loss"])
    gradient_norm = float(train_records[0]["grad_norm"])
    eval_loss = float(eval_records[0]["eval_loss"])
    if not all(math.isfinite(value) for value in (loss, gradient_norm, eval_loss)):
        raise RuntimeError("A diagnostic metric is non-finite.")
    if gradient_norm <= 0:
        raise RuntimeError("Expected a positive gradient norm.")
    model_files = sorted(output_dir.glob("*.safetensors"))
    if model_files:
        raise RuntimeError(f"Unexpected full model files: {model_files}")
    result = {
        "status": "SUCCESS",
        "global_step": 1,
        "optimizer_step_performed": True,
        "training_loss": loss,
        "gradient_norm": gradient_norm,
        "eval_step": 1,
        "eval_loss": eval_loss,
        "full_model_saved": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
