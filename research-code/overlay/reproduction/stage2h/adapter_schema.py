from typing import Dict

from qwenvl.train.trainable_checkpoint import load_trainable_checkpoint

from reproduction.stage2f.adapter_schema import stage1_architecture_parameter_names


STAGE = "2H-E"
VALID_METHODS = frozenset({"medic-ad-b0-as", "lad-mil-v2"})


def expected_stage2h_metadata(
    *,
    method_id: str,
    evidence_loss_weight: float,
    base_model_revision: str,
    base_checkpoint_fingerprint: str,
    dataset_manifest_sha256: str,
    training_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    implementation_source_fingerprint: str,
    run_id: str,
    protocol_version: str = "2.1",
) -> Dict:
    if method_id not in VALID_METHODS:
        raise ValueError(f"Unsupported Stage 2H method_id: {method_id!r}")
    weight = float(evidence_loss_weight)
    if method_id == "medic-ad-b0-as" and weight != 0.0:
        raise ValueError("B0-AS requires evidence_loss_weight=0.")
    if method_id == "lad-mil-v2" and weight <= 0.0:
        raise ValueError("LAD-MIL v2 requires a positive evidence_loss_weight.")
    return {
        "stage": STAGE,
        "protocol_version": protocol_version,
        "method_id": method_id,
        "train_type": "anomaly_evidence",
        "evidence_loss_weight": weight,
        "evidence_definition": "pre_gate_sigmoid_difference",
        "evidence_margin": 0.1,
        "evidence_topk_fraction": 0.01,
        "evidence_topk_k": 11,
        "anomaly_query_mode": "single",
        "num_pooling_size": 4,
        "output_token_count": 16,
        "gate_type": "none",
        "base_model_revision": base_model_revision,
        "base_checkpoint_fingerprint": base_checkpoint_fingerprint,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "implementation_source_fingerprint": implementation_source_fingerprint,
        "run_id": run_id,
        "claim_scope": "image_level_internal_engineering_only",
        "tune_mm_vpt": True,
        "tune_mm_anomaly": True,
        "seed": 42,
        "data_seed": 42,
        "global_step": 2,
        "full_determinism": False,
    }


def assert_stage2h_architecture(model) -> Dict:
    visual = getattr(model, "visual", None)
    if visual is None and hasattr(model, "model"):
        visual = getattr(model.model, "visual", None)
    if visual is None:
        raise RuntimeError("Stage 2H model is missing its visual module.")
    anomaly = getattr(visual, "anomaly_qformer", None)
    if anomaly is None:
        raise RuntimeError("Stage 2H model is missing AnomalyQformer.")
    mode = str(getattr(visual, "anomaly_query_mode", "")).strip().lower()
    pooling = int(getattr(visual, "num_pooling_size", 0))
    if mode != "single" or pooling != 4:
        raise RuntimeError(
            f"Stage 2H requires single/4, got mode={mode!r}, pooling={pooling}."
        )
    if getattr(anomaly, "fusion_gate", None) is not None:
        raise RuntimeError("Stage 2H found a forbidden fusion_gate.")
    names = stage1_architecture_parameter_names(model, "single")
    parameters = dict(model.named_parameters())
    elements = sum(int(parameters[name].numel()) for name in names)
    if len(names) != 21 or elements != 29_561_345:
        raise RuntimeError("Stage 2H B0-compatible parameter schema changed.")
    return {
        "status": "SUCCESS",
        "anomaly_query_mode": mode,
        "num_pooling_size": pooling,
        "output_token_count": pooling**2,
        "fusion_gate": None,
        "trainable_parameter_tensors": len(names),
        "trainable_parameter_elements": elements,
        "parameter_names": names,
    }


def load_stage2h_adapter(
    model,
    checkpoint_path: str,
    *,
    expected_metadata: Dict,
) -> Dict:
    architecture = assert_stage2h_architecture(model)
    manifest = load_trainable_checkpoint(
        model,
        checkpoint_path,
        strict_schema=True,
        expected_metadata=expected_metadata,
        expected_parameter_names=architecture["parameter_names"],
    )
    return manifest
