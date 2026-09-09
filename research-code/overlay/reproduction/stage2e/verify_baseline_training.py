#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path


EXPECTED_TENSORS = 21
EXPECTED_ELEMENTS = 29_561_345


def main() -> None:
    output_dir = Path(os.environ["MEDIC_AD_BASELINE_OUTPUT"])
    log_dir = Path(os.environ["MEDIC_AD_BASELINE_LOG_DIR"])
    run_id = os.environ["MEDIC_AD_RUN_ID"]
    adapter_path = Path(os.environ["MEDIC_AD_TRAINABLE_STATE_OUTPUT"])
    manifest_path = adapter_path.with_suffix(".manifest.json")
    state_path = output_dir / "trainer_state.json"
    audit_path = log_dir / f"{run_id}-gradient-audit.jsonl"
    loss_trace_path = log_dir / f"{run_id}-loss-trace.jsonl"
    for required in (state_path, audit_path, loss_trace_path, adapter_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != 32:
        raise RuntimeError(f"Expected global_step=32, got {state.get('global_step')}")
    train_records = [record for record in state.get("log_history", []) if "loss" in record and "grad_norm" in record]
    if [record.get("step") for record in train_records] != list(range(1, 33)):
        raise RuntimeError("Expected training records for steps 1..32.")
    eval_records = [record for record in state.get("log_history", []) if "eval_loss" in record]
    if [record.get("step") for record in eval_records] != [16, 32]:
        raise RuntimeError(f"Expected evaluation at steps 16 and 32: {eval_records}")
    if not all(math.isfinite(float(record["eval_loss"])) for record in eval_records):
        raise RuntimeError("Evaluation loss contains a non-finite value.")

    loss_trace = [json.loads(line) for line in loss_trace_path.read_text().splitlines() if line.strip()]
    if [record.get("global_step_before_update") for record in loss_trace] != list(range(32)):
        raise RuntimeError("Expected unrounded loss records before steps 0..31.")
    if not all(math.isfinite(float(record["loss"])) for record in loss_trace):
        raise RuntimeError("Training loss trace contains a non-finite value.")

    audits = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    audit_steps = [record for record in audits if record.get("event") == "gradient_audit_step"]
    audit_end = [record for record in audits if record.get("event") == "gradient_audit_end"]
    if [record.get("optimizer_step") for record in audit_steps] != [16, 32]:
        raise RuntimeError("Expected gradient audits at steps 16 and 32.")
    if len(audit_end) != 1 or audit_end[0].get("status") != "SUCCESS":
        raise RuntimeError("Gradient audit did not succeed.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCESS":
        raise RuntimeError("Trainable checkpoint manifest is not successful.")
    if manifest.get("trainable_parameter_tensors") != EXPECTED_TENSORS:
        raise RuntimeError("Unexpected trainable checkpoint tensor count.")
    if manifest.get("trainable_parameter_elements") != EXPECTED_ELEMENTS:
        raise RuntimeError("Unexpected trainable checkpoint element count.")
    if manifest.get("metadata", {}).get("global_step") != 32:
        raise RuntimeError("Trainable checkpoint global step mismatch.")
    if list(output_dir.glob("*.safetensors")):
        raise RuntimeError("Output directory unexpectedly contains full-model safetensors.")

    result = {
        "status": "SUCCESS",
        "run_id": run_id,
        "seed": manifest["metadata"]["seed"],
        "global_step": 32,
        "training_losses": [float(record["loss"]) for record in loss_trace],
        "eval_steps": [16, 32],
        "eval_losses": [float(record["eval_loss"]) for record in eval_records],
        "gradient_audit_status": "SUCCESS",
        "trainable_parameter_tensors": EXPECTED_TENSORS,
        "trainable_parameter_elements": EXPECTED_ELEMENTS,
        "adapter_path": str(adapter_path),
        "adapter_bytes": manifest["checkpoint_bytes"],
        "adapter_sha256": manifest["checkpoint_sha256"],
        "full_model_saved": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
