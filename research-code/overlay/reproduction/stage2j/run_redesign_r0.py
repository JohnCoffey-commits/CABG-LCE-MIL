#!/usr/bin/env python3
"""Exploratory, read-only CABG-MIL v1.2 R0 real-model diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import (
    ABNORMAL_LABEL,
    EXPECTED_TRAINABLE_ELEMENTS,
    EXPECTED_TRAINABLE_TENSORS,
    NORMAL_LABEL,
    SHARED_SUPPORT_NAMES,
)
from reproduction.stage2i.d3_mechanisms import MECHANISM_IDS, compute_mechanisms
from reproduction.stage2i.fingerprint import (
    canonical_json_sha256,
    git_identity,
    implementation_source_record,
    sha256_file,
    tensor_sha256,
)
from reproduction.stage2i.gate_d3_preflight import _gpu_snapshot, checkpoint_shard_audit
from reproduction.stage2i.rng_state import restore_rng_state, rng_state_fingerprint, snapshot_rng_state
from reproduction.stage2i.run_gate_d3 import (
    MedicalImageOpenAudit,
    NvidiaSmiMonitor,
    PhaseMarkers,
    _exclusive_json,
    _load_json,
    _load_jsonl,
    _state_grad_audit,
)
from reproduction.stage2j import SCHEMA_VERSION
from reproduction.stage2j.candidate_math import CANDIDATE_ID, compute_softsign_logit_contrast
from reproduction.stage2j.raw_observer import RawAnomalyAttentionObserver
from reproduction.stage2j.statistics import (
    distribution_record,
    gradient_rows,
    layer_distribution_records,
    named_global_norm,
    support_intersection,
)


EXPECTED_D3_VERIFICATION_SHA256 = "35727163c23b6ee7cd91e0cc8ad3c9f1a51eabcb8f0282b2f46df7792808237c"
EXPECTED_D3_EVIDENCE_FINGERPRINT = "7a00b261dc86fc0ae8f68847de9e8b026a4cac6b3581fa6a295669a63153df37"
EXPECTED_MANIFEST_SHA256 = "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c"
EXPECTED_PARENT_COMMIT = "f8bd0130bb9a2ba0324e73148a4ecd0c331ec361"
EXTERNAL_PEAK_LIMIT_MIB = 22500


SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "reproduction/stage2h/runtime.py",
    "reproduction/stage2h/state_audit.py",
    "reproduction/stage2i/constants.py",
    "reproduction/stage2i/d3_mechanisms.py",
    "reproduction/stage2i/run_gate_d3.py",
    "reproduction/stage2j/__init__.py",
    "reproduction/stage2j/candidate_math.py",
    "reproduction/stage2j/raw_observer.py",
    "reproduction/stage2j/run_redesign_r0_v2_l4.sh",
    "reproduction/stage2j/run_unit_tests.py",
    "reproduction/stage2j/statistics.py",
    "reproduction/stage2j/tests/test_static_contracts.py",
    "reproduction/stage2j/tests/test_statistics.py",
    "reproduction/stage2j/run_redesign_r0.py",
    "reproduction/stage2j/verify_redesign_r0.py",
)


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for row in rows:
            raw = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            os.write(descriptor, (raw + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _spatial_record(mechanism: dict[str, object]) -> dict[str, object]:
    nonzero = int(mechanism["nonzero_pooling_weights"][0].detach().cpu())
    return {
        "entropy": float(mechanism["spatial_entropy"][0].detach().cpu()),
        "effective_support": float(mechanism["effective_support"][0].detach().cpu()),
        "top11_mass": float(mechanism["top11_mass"][0].detach().cpu()),
        "max_pooling_weight": float(mechanism["max_pooling_weight"][0].detach().cpu()),
        "min_nonzero_pooling_weight": float(mechanism["min_nonzero_pooling_weight"][0].detach().cpu()),
        "nonzero_pooling_weights": nonzero,
        "exact_one_position_collapse": nonzero == 1,
    }


def _branch_record(observed: dict[str, torch.Tensor]) -> dict[str, object]:
    specifications = {
        "abnormal_raw": "raw_logit",
        "normal_raw": "raw_logit",
        "raw_gap": "raw_gap",
        "abnormal_probability": "probability",
        "normal_probability": "probability",
        "evidence": "evidence",
    }
    all_values = {
        name: distribution_record(observed[name], kind=kind)
        for name, kind in specifications.items()
    }
    layerwise = {
        name: layer_distribution_records(observed[name], kind=kind)
        for name, kind in specifications.items()
    }
    patch_layer_mean = {
        name: distribution_record(observed[name].mean(dim=1), kind=kind)
        for name, kind in specifications.items()
    }
    return {
        "tensor_shape": list(observed["evidence"].shape),
        "raw_payload": {
            "dtype": "float32_serialized_from_bfloat16_forward",
            "abnormal_raw_flat": observed["abnormal_raw"].detach().float().cpu().reshape(-1).tolist(),
            "normal_raw_flat": observed["normal_raw"].detach().float().cpu().reshape(-1).tolist(),
            "abnormal_probability_flat": observed["abnormal_probability"].detach().float().cpu().reshape(-1).tolist(),
            "normal_probability_flat": observed["normal_probability"].detach().float().cpu().reshape(-1).tolist(),
        },
        "all_layer_patch_values": all_values,
        "per_selected_layer": layerwise,
        "patch_after_layer_mean": patch_layer_mean,
        "reconstruction_exact": bool(torch.equal(observed["evidence"], observed["abnormal_probability"] - observed["normal_probability"])),
    }


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    if sha256_file(args.d3_verification) != EXPECTED_D3_VERIFICATION_SHA256:
        raise CABGContractError("R0 source D3 verification SHA-256 changed.")
    d3 = _load_json(args.d3_verification)
    if (
        d3.get("status") != "SUCCESS"
        or d3.get("decision") != "PAUSE"
        or d3.get("evidence_bundle_fingerprint") != EXPECTED_D3_EVIDENCE_FINGERPRINT
        or d3.get("claim_scope") != "read_only_mechanism_diagnostic_only"
    ):
        raise CABGContractError("R0 source D3 decision/evidence contract changed.")
    preflight = _load_json(args.d3_preflight)
    if preflight.get("status") != "SUCCESS" or preflight.get("decision") != "READY_FOR_GATE_D3_MODEL_LOAD":
        raise CABGContractError("R0 source D3 preflight is invalid.")
    observed_checkpoint = checkpoint_shard_audit(args.base_model)
    locked_checkpoint = preflight.get("base_checkpoint", {})
    for key in ("index_sha256", "checkpoint_shard_list_fingerprint", "physical_shard_bytes", "shard_count"):
        if observed_checkpoint.get(key) != locked_checkpoint.get(key):
            raise CABGContractError(f"R0 base checkpoint mismatch: {key}")
    dataset = _load_json(args.dataset_audit)
    manifest_path = Path(str(dataset["locked_manifest"]))
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CABGContractError("R0 locked 12-image manifest changed.")
    manifest = _load_jsonl(manifest_path)
    labels = [normalize_label(row["label"]) for row in manifest]
    if len(manifest) != 12 or Counter(labels) != Counter({NORMAL_LABEL: 6, ABNORMAL_LABEL: 6}):
        raise CABGContractError("R0 requires the exact balanced 12-image D3 subset.")
    if dataset.get("internal_test_outputs_read") != 0:
        raise CABGContractError("R0 dataset audit indicates internal-test access.")
    return d3, dataset, manifest


def run(args: argparse.Namespace) -> dict[str, object]:
    outputs = (
        args.records_output,
        args.model_load_audit_output,
        args.state_before_output,
        args.state_after_output,
        args.file_open_audit_output,
        args.rng_audit_output,
        args.monitor_summary_output,
        args.source_inventory_output,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite CABG-MIL v1.2 R0 evidence.")
    d3, dataset_audit, manifest = _validate_inputs(args)
    labels = [normalize_label(row["label"]) for row in manifest]
    repository = git_identity(args.repo_root)
    if repository["dirty"]:
        raise CABGContractError("R0 remote diagnostic worktree must be clean.")
    if subprocess_result(args.repo_root, ["git", "merge-base", "--is-ancestor", EXPECTED_PARENT_COMMIT, "HEAD"]) != 0:
        raise CABGContractError("R0 source is not descended from the locked Gate D3 commit.")
    source = implementation_source_record(args.repo_root, SOURCE_FILES)
    _exclusive_json(args.source_inventory_output, source)
    disk = shutil.disk_usage("/home")
    if disk.free < 5 * 1024**3:
        raise CABGContractError("R0 requires at least 5 GiB free on /home.")
    pre_model_gpu = _gpu_snapshot()

    os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = str(dataset_audit["annotation"])
    os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = str(dataset_audit["image_root"])
    from reproduction.stage2h.runtime import load_stage2h_runtime, make_stage2h_dataset, move_batch_to_device

    monitor = NvidiaSmiMonitor(args.monitor_output)
    phases = PhaseMarkers(args.phase_markers_output)
    monitor.start()
    phases.mark("monitor_started")
    wrapper = None
    outer_rng = None
    records: list[dict[str, object]] = []
    file_audit = MedicalImageOpenAudit(Path(str(dataset_audit["image_root"])), list(dataset_audit["allowed_image_paths"]))
    try:
        phases.mark("model_load_started")
        wrapper, runtime_audit = load_stage2h_runtime(args.base_model, seed=42, training=True, gradient_checkpointing=True)
        model = wrapper.llm
        phases.mark("model_load_completed")
        scope = runtime_audit["trainable_scope"]
        trainable_names = list(scope["observed_names"])
        if len(trainable_names) != EXPECTED_TRAINABLE_TENSORS or int(scope["trainable_parameter_elements"]) != EXPECTED_TRAINABLE_ELEMENTS:
            raise CABGContractError("R0 trainable scope differs from the locked 21-tensor scope.")
        parameters = dict(model.named_parameters())
        trainable_parameters = [parameters[name] for name in trainable_names]
        state_before = trainable_state_record(model)
        if state_before != runtime_audit["step0_state"]:
            raise CABGContractError("R0 state changed immediately after model reset.")
        _exclusive_json(args.state_before_output, state_before)
        outer_rng = snapshot_rng_state()
        rng_before = rng_state_fingerprint(outer_rng)
        model_load = {
            "status": "SUCCESS",
            "schema_version": "cabg-mil-v1.2-redesign-r0-model-load-1",
            "source_commit": repository["commit"],
            "parent_gate_d3_commit": EXPECTED_PARENT_COMMIT,
            "source_fingerprint": source["fingerprint"],
            "source_d3_verification_sha256": EXPECTED_D3_VERIFICATION_SHA256,
            "source_d3_evidence_fingerprint": EXPECTED_D3_EVIDENCE_FINGERPRINT,
            "runtime_audit": runtime_audit,
            "model_dtype": "bfloat16",
            "model_training": bool(model.training),
            "gradient_checkpointing": bool(runtime_audit["gradient_checkpointing"]),
            "use_cache": bool(model.config.use_cache),
            "anomaly_query_mode": model.visual.anomaly_query_mode,
            "fusion_gate_present": getattr(model.visual.anomaly_qformer, "fusion_gate", None) is not None,
            "optimizer_created": False,
            "scheduler_created": False,
            "parameter_update_performed": False,
            "official_inference_path_modified": False,
            "quantization": False,
            "offloading": False,
            "pre_model_gpu": pre_model_gpu,
            "home_free_bytes_before_model_load": disk.free,
        }
        if not model_load["model_training"] or not model_load["gradient_checkpointing"] or model_load["use_cache"] or model_load["anomaly_query_mode"] != "single" or model_load["fusion_gate_present"]:
            raise CABGContractError(f"R0 runtime contract changed: {model_load}")
        _exclusive_json(args.model_load_audit_output, model_load)
        _, dataset, collator = make_stage2h_dataset(args.base_model, "medic_ad_stage2h_calibration", shuffle=False)
        expected_rows = [(f"cabg-d3-{row['sample_id']}", str(row["sample_id"]), str(row["relative_path"]), str(row["sha256"])) for row in manifest]
        observed_rows = [(str(row.get("id")), str(row.get("cabg_sample_id")), str(row.get("image")), str(row.get("image_sha256"))) for row in dataset.list_data_dict]
        if len(dataset) != 12 or observed_rows != expected_rows:
            raise CABGContractError("R0 runtime dataset order or identity differs from Gate D3.")
        file_audit.install()

        for image_index, (manifest_row, label) in enumerate(zip(manifest, labels)):
            started = time.perf_counter()
            phases.mark("image_started", image_index=image_index, sample_id=manifest_row["sample_id"])
            batch = move_batch_to_device(collator([dataset[image_index]]), wrapper.device)
            anomaly_labels = batch.pop("anomaly_labels")
            expected_label = 1 if label == ABNORMAL_LABEL else 0
            if anomaly_labels.numel() != 1 or int(anomaly_labels.item()) != expected_label:
                raise CABGContractError("R0 runtime label mismatch.")
            batch["tune_mode"] = "default"
            batch["return_anomaly_evidence"] = True
            rng_image_before = rng_state_fingerprint(snapshot_rng_state())
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            forward_started = time.perf_counter()
            observer_module = model.visual.anomaly_qformer.anomaly_attention
            with RawAnomalyAttentionObserver(observer_module) as observer:
                result = model(**batch)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            if result.loss is None or result.anomaly_evidence is None:
                raise CABGContractError("R0 forward did not return LM loss and anomaly evidence.")
            evidence = result.anomaly_evidence
            observed = observer.finalize(evidence)
            mechanisms = compute_mechanisms(evidence, [label])
            mechanisms[CANDIDATE_ID] = compute_softsign_logit_contrast(
                observed["abnormal_raw"], observed["normal_raw"], [label]
            )
            lm_gradients = torch.autograd.grad(result.loss, trainable_parameters, retain_graph=True, allow_unused=True)
            lm_rows = gradient_rows(trainable_names, lm_gradients)
            mechanism_records = {}
            diagnostic_mechanism_ids = (*MECHANISM_IDS, CANDIDATE_ID)
            for mechanism_index, mechanism_id in enumerate(diagnostic_mechanism_ids):
                auxiliary_gradients = torch.autograd.grad(
                    mechanisms[mechanism_id]["per_image_loss"][0],
                    trainable_parameters,
                    retain_graph=mechanism_index < len(diagnostic_mechanism_ids) - 1,
                    allow_unused=True,
                )
                auxiliary_rows = gradient_rows(trainable_names, auxiliary_gradients)
                support = support_intersection(lm_rows, auxiliary_rows)
                legacy_support = list(SHARED_SUPPORT_NAMES)
                mechanism_records[mechanism_id] = {
                    "name": str(mechanisms[mechanism_id]["mechanism"]),
                    "score": float(mechanisms[mechanism_id]["score"][0].detach().cpu()),
                    "loss": float(mechanisms[mechanism_id]["per_image_loss"][0].detach().cpu()),
                    "active_hinge": bool(mechanism_id == "M0" and float(mechanisms[mechanism_id]["per_image_loss"][0].detach().cpu()) > 0.0),
                    "auxiliary_gradients_all_trainable": auxiliary_rows,
                    "support": support,
                    "auxiliary_global_norm_all_effective": named_global_norm(auxiliary_rows, support["auxiliary_effective"]),
                    "auxiliary_global_norm_actual_intersection": named_global_norm(auxiliary_rows, support["actual_intersection"]),
                    "auxiliary_global_norm_legacy_six": named_global_norm(auxiliary_rows, legacy_support),
                    "spatial": _spatial_record(mechanisms[mechanism_id]) if mechanism_id == "M2" else None,
                }
            branch = _branch_record(observed)
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "SUCCESS",
                "claim_scope": "exploratory_read_only_mechanism_diagnostic_only",
                "sample_id": str(manifest_row["sample_id"]),
                "image_sha256": str(manifest_row["sha256"]),
                "relative_path": str(manifest_row["relative_path"]),
                "label": label,
                "forward_id": f"r0-forward-{image_index:03d}",
                "evidence_fingerprint": tensor_sha256(evidence),
                "lm_loss": float(result.loss.detach().float().cpu()),
                "lm_gradients_all_trainable": lm_rows,
                "lm_global_norm_all_effective": named_global_norm(lm_rows, [row["name"] for row in lm_rows if row["effective"]]),
                "branches": branch,
                "mechanisms": mechanism_records,
                "memory": {
                    "image_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024**2),
                    "image_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024**2),
                    "run_external_peak_mib": 0.0,
                },
                "runtime": {"forward_seconds": forward_seconds, "image_seconds": 0.0},
                "rng": {"before_image": rng_image_before, "after_image": ""},
                "provenance": {
                    "source_commit": repository["commit"],
                    "source_fingerprint": source["fingerprint"],
                    "source_d3_evidence_fingerprint": EXPECTED_D3_EVIDENCE_FINGERPRINT,
                    "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "legacy_shared_support_fingerprint": canonical_json_sha256(list(SHARED_SUPPORT_NAMES)),
                },
            }
            del auxiliary_gradients, lm_gradients, result, mechanisms, evidence, observed, batch, anomaly_labels
            gc.collect()
            torch.cuda.synchronize()
            record["runtime"]["image_seconds"] = time.perf_counter() - started
            record["rng"]["after_image"] = rng_state_fingerprint(snapshot_rng_state())
            records.append(record)
            phases.mark("image_completed", image_index=image_index, sample_id=manifest_row["sample_id"])

        state_after = trainable_state_record(model)
        grad_audit = _state_grad_audit(model, trainable_names)
        state_after["grad_audit"] = grad_audit
        state_after["matches_before"] = state_after["trainable_state_fingerprint"] == state_before["trainable_state_fingerprint"]
        if not state_after["matches_before"] or not grad_audit["all_grad_none"]:
            raise CABGContractError("R0 read-only parameter-state or .grad contract failed.")
        _exclusive_json(args.state_after_output, state_after)
        file_result = file_audit.result()
        if file_result["status"] != "SUCCESS":
            raise CABGContractError("R0 file-open isolation failed.")
        file_result["schema_version"] = "cabg-mil-v1.2-redesign-r0-file-open-1"
        _exclusive_json(args.file_open_audit_output, file_result)
        restore_rng_state(outer_rng)
        rng_after_restore = rng_state_fingerprint(snapshot_rng_state())
        rng_result = {
            "status": "SUCCESS" if rng_after_restore == rng_before else "FAILED",
            "schema_version": "cabg-mil-v1.2-redesign-r0-rng-1",
            "outer_before": rng_before,
            "after_restore": rng_after_restore,
            "restored_exactly": rng_after_restore == rng_before,
            "natural_sequential_stream": True,
            "per_mechanism_reseeded": False,
        }
        if rng_result["status"] != "SUCCESS":
            raise CABGContractError("R0 RNG restoration failed.")
        _exclusive_json(args.rng_audit_output, rng_result)
        phases.mark("diagnostic_completed")
        del model, wrapper, dataset, collator, parameters, trainable_parameters
        wrapper = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        phases.mark("model_released")
    finally:
        if outer_rng is not None:
            restore_rng_state(outer_rng)
        if wrapper is not None:
            del wrapper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        phases.mark("monitor_stopping")
        phases.close()
        monitor_result = monitor.stop()

    _exclusive_json(args.monitor_summary_output, monitor_result)
    peak = float(monitor_result["peak_memory_used_mib"])
    for record in records:
        record["memory"]["run_external_peak_mib"] = peak
    if len(records) != 12 or len({row["forward_id"] for row in records}) != 12:
        raise CABGContractError("R0 did not produce exactly 12 one-forward records.")
    _write_jsonl_exclusive(args.records_output, records)
    summary = {
        "status": "SUCCESS",
        "claim_scope": "exploratory_read_only_mechanism_diagnostic_only",
        "images": 12,
        "forwards": 12,
        "records": 12,
        "external_peak_mib": peak,
        "external_peak_limit_mib": EXTERNAL_PEAK_LIMIT_MIB,
        "records_sha256": sha256_file(args.records_output),
        "independent_evaluator_required": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def subprocess_result(repo_root: Path, command: list[str]) -> int:
    import subprocess

    return subprocess.run(command, cwd=repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--d3-preflight", type=Path, required=True)
    parser.add_argument("--d3-verification", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--model-load-audit-output", type=Path, required=True)
    parser.add_argument("--state-before-output", type=Path, required=True)
    parser.add_argument("--state-after-output", type=Path, required=True)
    parser.add_argument("--file-open-audit-output", type=Path, required=True)
    parser.add_argument("--rng-audit-output", type=Path, required=True)
    parser.add_argument("--monitor-output", type=Path, required=True)
    parser.add_argument("--monitor-summary-output", type=Path, required=True)
    parser.add_argument("--phase-markers-output", type=Path, required=True)
    parser.add_argument("--source-inventory-output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
