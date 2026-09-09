import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file


ADAPTER_SCHEMA_VERSION = 2


def _distributed_is_initialized() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _rank() -> int:
    return torch.distributed.get_rank() if _distributed_is_initialized() else 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".manifest.json")


def _parameter_fingerprint(parameter_records) -> str:
    canonical = [
        {
            "name": record["name"],
            "shape": list(record["shape"]),
            "dtype": record["dtype"],
            "elements": int(record["elements"]),
        }
        for record in sorted(parameter_records, key=lambda item: item["name"])
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gather_context(param: torch.nn.Parameter):
    if not hasattr(param, "ds_status"):
        return nullcontext()
    try:
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except Exception as exc:  # pragma: no cover - requires DeepSpeed runtime
        raise RuntimeError("DeepSpeed is required to gather partitioned trainable parameters.") from exc
    gather_needed = getattr(param, "ds_status", None) != ZeroParamStatus.AVAILABLE
    return deepspeed.zero.GatheredParameters(
        [param],
        modifier_rank=0,
        enabled=gather_needed,
    )


def save_trainable_checkpoint(
    trainer,
    output_path: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    checkpoint_path = Path(output_path).expanduser()
    manifest_path = _manifest_path(checkpoint_path)
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite trainable checkpoint evidence: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = trainer.model
    state = {}
    expected_names = [
        name
        for name, param in model.named_parameters()
        if isinstance(param, torch.nn.Parameter) and param.requires_grad
    ]
    if not expected_names:
        raise RuntimeError("No trainable parameters were found for checkpoint export.")

    for name, param in model.named_parameters():
        if name not in expected_names:
            continue
        with _gather_context(param):
            if _rank() == 0:
                if param.numel() == 0:
                    raise RuntimeError(f"Gathered trainable parameter is empty: {name}")
                state[name] = param.detach().cpu().contiguous().clone()

    if _distributed_is_initialized():
        torch.distributed.barrier()

    manifest = {}
    if _rank() == 0:
        if set(state) != set(expected_names):
            raise RuntimeError("Trainable checkpoint export did not collect every expected parameter.")
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        save_file(dict(sorted(state.items())), str(tmp_path))
        os.replace(tmp_path, checkpoint_path)
        parameter_records = [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "elements": int(tensor.numel()),
            }
            for name, tensor in sorted(state.items())
        ]
        manifest = {
            "status": "SUCCESS",
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "format": "safetensors",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "trainable_parameter_tensors": len(parameter_records),
            "trainable_parameter_elements": sum(record["elements"] for record in parameter_records),
            "trainable_key_fingerprint": _parameter_fingerprint(parameter_records),
            "parameters": parameter_records,
            "metadata": metadata or {},
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def load_trainable_checkpoint(
    model,
    checkpoint_path: str,
    *,
    strict_schema: bool = False,
    expected_metadata: Optional[Dict] = None,
    expected_parameter_names=None,
) -> Dict:
    path = Path(checkpoint_path).expanduser()
    manifest_path = _manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing trainable checkpoint or manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCESS":
        raise RuntimeError("Trainable checkpoint manifest is not successful.")
    if manifest.get("format") != "safetensors":
        raise RuntimeError("Trainable checkpoint manifest has an unsupported format.")
    if path.stat().st_size != manifest.get("checkpoint_bytes"):
        raise RuntimeError("Trainable checkpoint byte-size mismatch.")
    if _sha256_file(path) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("Trainable checkpoint SHA-256 mismatch.")

    state = load_file(str(path), device="cpu")
    parameter_records = manifest.get("parameters", [])
    records = {record["name"]: record for record in parameter_records}
    if len(records) != len(parameter_records):
        raise RuntimeError("Trainable checkpoint manifest contains duplicate parameter names.")
    if set(state) != set(records):
        raise RuntimeError("Checkpoint tensor names do not match the manifest.")
    if manifest.get("trainable_parameter_tensors") != len(parameter_records):
        raise RuntimeError("Trainable checkpoint tensor-count mismatch.")
    observed_elements = sum(int(tensor.numel()) for tensor in state.values())
    if manifest.get("trainable_parameter_elements") != observed_elements:
        raise RuntimeError("Trainable checkpoint element-count mismatch.")
    for name, tensor in state.items():
        record = records[name]
        observed_dtype = str(tensor.dtype).removeprefix("torch.")
        if list(tensor.shape) != record.get("shape"):
            raise RuntimeError(f"Manifest shape mismatch: {name}")
        if observed_dtype != record.get("dtype"):
            raise RuntimeError(f"Manifest dtype mismatch: {name}")
        if int(tensor.numel()) != record.get("elements"):
            raise RuntimeError(f"Manifest element-count mismatch: {name}")

    schema_version = manifest.get("schema_version")
    fingerprint = _parameter_fingerprint(parameter_records)
    recorded_fingerprint = manifest.get("trainable_key_fingerprint")
    if recorded_fingerprint is not None and fingerprint != recorded_fingerprint:
        raise RuntimeError("Trainable checkpoint key fingerprint mismatch.")
    if strict_schema:
        if schema_version != ADAPTER_SCHEMA_VERSION:
            raise RuntimeError(
                f"Strict adapter schema requires version {ADAPTER_SCHEMA_VERSION}, got {schema_version!r}."
            )
        if recorded_fingerprint is None:
            raise RuntimeError("Strict adapter schema requires a trainable-key fingerprint.")
        if expected_parameter_names is None:
            raise ValueError("Strict adapter loading requires expected_parameter_names.")
        expected_names = set(expected_parameter_names)
        if set(state) != expected_names:
            missing = sorted(expected_names - set(state))
            unexpected = sorted(set(state) - expected_names)
            raise RuntimeError(
                f"Strict adapter parameter-set mismatch: missing={missing}, unexpected={unexpected}"
            )
        metadata = manifest.get("metadata", {})
        for key, expected in (expected_metadata or {}).items():
            if key not in metadata:
                raise RuntimeError(f"Strict adapter metadata is missing: {key}")
            if metadata[key] != expected:
                raise RuntimeError(
                    f"Strict adapter metadata mismatch for {key}: "
                    f"expected {expected!r}, got {metadata[key]!r}"
                )

    model_parameters = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in model_parameters:
            raise RuntimeError(f"Checkpoint parameter does not exist in model: {name}")
        if tuple(tensor.shape) != tuple(model_parameters[name].shape):
            raise RuntimeError(f"Model shape mismatch: {name}")

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected trainable checkpoint keys: {incompatible.unexpected_keys}")
    for name, expected in state.items():
        observed = model_parameters[name].detach().cpu().to(dtype=expected.dtype)
        if not torch.equal(observed, expected):
            raise RuntimeError(f"Loaded trainable parameter verification failed: {name}")
    return manifest
