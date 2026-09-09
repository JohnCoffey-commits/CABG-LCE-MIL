import hashlib
import json
from typing import Dict, Iterable, List

import torch


VALID_QUERY_MODES = frozenset({"single", "multiscale"})
BASE_MODEL_REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
BASE_CHECKPOINT_FINGERPRINT = "9bc1b0de134f6a94f3ddaa6e6b0f3566c4ee55a2dbdccea27e2cb8432a02e16d"
DATASET_MANIFEST_SHA256 = "8c5ebce131590351758a5bb6f1b7d41d894ba9d584df5f6ea5a67c61130ad118"
SOURCE_BASE_COMMIT = "580abb4"

EXPECTED_SCOPE = {
    "single": {"tensors": 21, "elements": 29_561_345},
    "multiscale": {"tensors": 23, "elements": 29_563_906},
}


def normalize_query_mode(mode: str) -> str:
    normalized = str(mode or "single").strip().lower()
    if normalized not in VALID_QUERY_MODES:
        raise ValueError(f"Unsupported anomaly query mode: {mode!r}")
    return normalized


def _is_stage1_parameter(name: str) -> bool:
    return name.startswith(
        (
            "model.visual.deep_prompt_embeddings.",
            "model.visual.anomaly_qformer.",
        )
    ) and not name.startswith(
        (
            "model.visual.anomaly_qformer.diff_q_former.",
            "model.visual.anomaly_qformer.diff_post_ffn.",
        )
    )


def stage1_architecture_parameter_names(model, mode: str) -> List[str]:
    mode = normalize_query_mode(mode)
    model_parameters = dict(model.named_parameters())
    names = sorted(name for name in model_parameters if _is_stage1_parameter(name))
    gate_names = sorted(name for name in names if ".fusion_gate." in name)
    expected_gate_names = [] if mode == "single" else [
        "model.visual.anomaly_qformer.fusion_gate.bias",
        "model.visual.anomaly_qformer.fusion_gate.weight",
    ]
    if gate_names != expected_gate_names:
        raise RuntimeError(
            f"Fusion-gate architecture mismatch for {mode}: "
            f"expected {expected_gate_names}, got {gate_names}"
        )
    expected = EXPECTED_SCOPE[mode]
    elements = sum(int(model_parameters[name].numel()) for name in names)
    if len(names) != expected["tensors"] or elements != expected["elements"]:
        raise RuntimeError(
            f"Stage 1 architecture scope mismatch for {mode}: "
            f"tensors={len(names)}/{expected['tensors']}, "
            f"elements={elements}/{expected['elements']}"
        )
    return names


def audit_trainable_scope(model, mode: str) -> Dict:
    mode = normalize_query_mode(mode)
    expected_names = set(stage1_architecture_parameter_names(model, mode))
    observed_names = {
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, torch.nn.Parameter) and parameter.requires_grad
    }
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    records = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype).removeprefix("torch."),
            "elements": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in sorted(model.named_parameters())
        if name in expected_names
    ]
    result = {
        "status": "SUCCESS" if not missing and not unexpected else "FAILED",
        "anomaly_query_mode": mode,
        "expected_names": sorted(expected_names),
        "observed_names": sorted(observed_names),
        "missing": missing,
        "unexpected": unexpected,
        "trainable_parameter_tensors": len(observed_names),
        "trainable_parameter_elements": sum(
            int(parameter.numel())
            for name, parameter in model.named_parameters()
            if name in observed_names
        ),
        "parameters": records,
    }
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["scope_fingerprint"] = hashlib.sha256(payload).hexdigest()
    if result["status"] != "SUCCESS":
        raise RuntimeError(
            f"Stage 1 trainable-scope mismatch: missing={missing}, unexpected={unexpected}"
        )
    return result


def strict_adapter_metadata(mode: str) -> Dict:
    mode = normalize_query_mode(mode)
    return {
        "anomaly_query_mode": mode,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_checkpoint_fingerprint": BASE_CHECKPOINT_FINGERPRINT,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_base_commit": SOURCE_BASE_COMMIT,
        "num_pooling_size": 4,
        "output_token_count": 16,
        "method_id": "fb-maq-stage2f" if mode == "multiscale" else "medic-ad-b0",
        "gate_type": "spatial_conv1x1" if mode == "multiscale" else "none",
        "gate_initial_lambda": 0.9 if mode == "multiscale" else None,
        "tune_mm_vpt": True,
        "tune_mm_anomaly": True,
    }


def require_metadata_fields(metadata: Dict, fields: Iterable[str]) -> None:
    missing = sorted(field for field in fields if field not in metadata or metadata[field] is None)
    if missing:
        raise RuntimeError(f"Adapter metadata is missing required non-null fields: {missing}")
