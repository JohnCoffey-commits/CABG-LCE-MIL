import hashlib
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

from reproduction.stage2f.adapter_schema import stage1_architecture_parameter_names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def trainable_state_record(model) -> Dict:
    names = stage1_architecture_parameter_names(model, "single")
    parameters = dict(model.named_parameters())
    records = []
    for name in names:
        tensor = parameters[name].detach()
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "elements": int(tensor.numel()),
                "tensor_sha256": tensor_sha256(tensor),
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    anomaly = model.model.visual.anomaly_qformer
    return {
        "status": "SUCCESS",
        "anomaly_query_mode": "single",
        "num_pooling_size": 4,
        "output_token_count": 16,
        "fusion_gate": None,
        "trainable_parameter_tensors": len(records),
        "trainable_parameter_elements": sum(record["elements"] for record in records),
        "trainable_state_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "gate_scale": float(anomaly.gate_scale.detach().float().cpu().item()),
        "parameters": records,
    }


def save_initial_trainable_state(model, checkpoint_path: str) -> Dict:
    path = Path(checkpoint_path).expanduser()
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 2H step-0 state: {path}")
    names = stage1_architecture_parameter_names(model, "single")
    parameters = dict(model.named_parameters())
    state = {name: parameters[name].detach().cpu().contiguous().clone() for name in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(state.items())), str(path))
    record = trainable_state_record(model)
    record.update(
        {
            "checkpoint_path": str(path),
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": sha256_file(path),
        }
    )
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def compare_initial_and_final(initial_path: Path, final_path: Path) -> Dict:
    initial = load_file(str(initial_path), device="cpu")
    final = load_file(str(final_path), device="cpu")
    if set(initial) != set(final):
        raise RuntimeError("Stage 2H initial/final adapter key sets differ.")
    parameters = []
    for name in sorted(initial):
        if initial[name].shape != final[name].shape:
            raise RuntimeError(f"Stage 2H initial/final shape mismatch: {name}")
        difference = final[name].float() - initial[name].float()
        parameters.append(
            {
                "name": name,
                "changed": not torch.equal(initial[name], final[name]),
                "max_abs_change": float(difference.abs().max().item()),
                "l2_change": float(difference.norm().item()),
            }
        )
    target_prefixes = (
        "model.visual.anomaly_qformer.abnormal_prompt",
        "model.visual.anomaly_qformer.normal_prompt",
        "model.visual.anomaly_qformer.anomaly_attention.query_proj.",
        "model.visual.anomaly_qformer.anomaly_attention.key_proj.",
    )
    targets = [row for row in parameters if row["name"].startswith(target_prefixes)]
    gate_row = next(row for row in parameters if row["name"].endswith(".gate_scale"))
    gate_name = gate_row["name"]
    gate_initial = float(initial[gate_name].float().item())
    gate_final = float(final[gate_name].float().item())
    gate_row.update(
        {
            "initial_value": gate_initial,
            "final_value": gate_final,
            "signed_change": gate_final - gate_initial,
            "absolute_change": abs(gate_final - gate_initial),
            "initial_finite_positive": bool(torch.isfinite(initial[gate_name].float()).all().item())
            and gate_initial > 0,
            "final_finite_positive": bool(torch.isfinite(final[gate_name].float()).all().item())
            and gate_final > 0,
        }
    )
    return {
        "status": "SUCCESS",
        "parameter_count": len(parameters),
        "changed_parameter_count": sum(row["changed"] for row in parameters),
        "target_parameter_count": len(targets),
        "changed_target_parameter_count": sum(row["changed"] for row in targets),
        "optimizer_changed_at_least_one_target": any(row["changed"] for row in targets),
        "gate_scale_change": gate_row,
        "parameters": parameters,
    }
