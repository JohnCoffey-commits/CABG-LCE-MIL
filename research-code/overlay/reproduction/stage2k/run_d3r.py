#!/usr/bin/env python3
"""Read-only real-model producer for CABG-LCE-MIL v1.2 D3R."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from reproduction.stage2h.state_audit import trainable_state_record
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
from reproduction.stage2k import RUN_ID, SCHEMA_VERSION
from reproduction.stage2k.constants import (
    ABNORMAL_LABEL,
    EXPECTED_APPROVED_R0_TREE,
    EXPECTED_FORMAL_R0_COMMIT,
    EXPECTED_GATE_D3_COMMIT,
    EXPECTED_PARENT_COMMIT,
    EXPECTED_TRAINABLE_ELEMENTS,
    EXPECTED_TRAINABLE_TENSORS,
    EXTERNAL_PEAK_LIMIT_MIB,
    NORMAL_LABEL,
    S9_NAMES,
    SOURCE_FILES,
    STRUCTURAL_ZERO_NAME,
    TRAINABLE_NAMES,
    TRAINABLE_SCHEMA,
)
from reproduction.stage2k.evidence import (
    RawAttentionObserver,
    global_norm,
    gradient_rows,
    tensor_payload,
)
from reproduction.stage2k.manifest import ManifestContractError, normalize_label
from reproduction.stage2k.producer_math import compute_lce


EXPECTED_MANIFEST_SHA256 = "7e581453de5f4abba2eafa87b982eccb751264f66e86ac10f3b92a160ae20db1"


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for row in rows:
            raw = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            os.write(descriptor, (raw + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_schema(scope: dict[str, object]) -> None:
    if list(scope.get("observed_names", [])) != list(TRAINABLE_NAMES):
        raise ManifestContractError("D3R trainable names differ from the locked 21-tensor schema.")
    if int(scope.get("trainable_parameter_tensors", -1)) != EXPECTED_TRAINABLE_TENSORS:
        raise ManifestContractError("D3R trainable tensor count changed.")
    if int(scope.get("trainable_parameter_elements", -1)) != EXPECTED_TRAINABLE_ELEMENTS:
        raise ManifestContractError("D3R trainable element count changed.")
    observed = [(row["name"], tuple(row["shape"]), int(row["elements"])) for row in scope.get("parameters", [])]
    if observed != list(TRAINABLE_SCHEMA):
        raise ManifestContractError("D3R trainable name/shape/element schema changed.")


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    dataset = _load_json(args.dataset_audit)
    if dataset.get("status") != "SUCCESS" or dataset.get("decision") != "D3R_SELECTED_IMAGE_BYTES_VERIFIED":
        raise ManifestContractError("D3R runtime dataset audit is invalid.")
    if dataset.get("locked_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ManifestContractError("D3R runtime manifest fingerprint changed.")
    if dataset.get("protected_internal_test_image_files_opened") != 0 or dataset.get("protected_internal_test_outputs_read") != 0:
        raise ManifestContractError("D3R runtime dataset audit crossed the protected boundary.")
    manifest_path = Path(str(dataset["locked_manifest"]))
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ManifestContractError("D3R locked manifest bytes changed.")
    manifest = _load_jsonl(manifest_path)
    labels = [normalize_label(row["d3r_label"]) for row in manifest]
    if len(manifest) != 24 or Counter(labels) != Counter({ABNORMAL_LABEL: 12, NORMAL_LABEL: 12}):
        raise ManifestContractError("D3R runtime manifest is not balanced 24.")
    old_preflight = _load_json(args.approved_preflight)
    if old_preflight.get("status") != "SUCCESS" or old_preflight.get("decision") != "READY_FOR_GATE_D3_MODEL_LOAD":
        raise ManifestContractError("Approved checkpoint preflight is invalid.")
    checkpoint = checkpoint_shard_audit(args.base_model)
    locked = old_preflight.get("base_checkpoint", {})
    for key in ("index_sha256", "checkpoint_shard_list_fingerprint", "physical_shard_bytes", "shard_count"):
        if checkpoint.get(key) != locked.get(key):
            raise ManifestContractError(f"Approved checkpoint mismatch: {key}")
    return dataset, manifest, checkpoint


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
        raise FileExistsError("Refusing to overwrite D3R evidence.")
    dataset_audit, manifest, checkpoint = _validate_inputs(args)
    labels = [normalize_label(row["d3r_label"]) for row in manifest]
    repository = git_identity(args.repo_root)
    if repository["dirty"]:
        raise ManifestContractError("D3R execution worktree must be clean.")
    for parent in (EXPECTED_GATE_D3_COMMIT, EXPECTED_PARENT_COMMIT):
        status = subprocess.run(
            ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
            cwd=args.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if status != 0:
            raise ManifestContractError(f"D3R source does not descend from {parent}.")
    approved_tree = subprocess.check_output(
        ["git", "rev-parse", f"{EXPECTED_PARENT_COMMIT}^{{tree}}"], cwd=args.repo_root, text=True
    ).strip()
    formal_tree = subprocess.check_output(
        ["git", "rev-parse", f"{EXPECTED_FORMAL_R0_COMMIT}^{{tree}}"], cwd=args.repo_root, text=True
    ).strip()
    if approved_tree != EXPECTED_APPROVED_R0_TREE or formal_tree != EXPECTED_APPROVED_R0_TREE:
        raise ManifestContractError("D3R approved local/formal R0 tree equivalence failed.")
    source = implementation_source_record(args.repo_root, SOURCE_FILES)
    source["repository"] = repository
    _exclusive_json(args.source_inventory_output, source)
    disk = shutil.disk_usage("/home")
    if disk.free < 5 * 1024**3:
        raise ManifestContractError("D3R requires at least 5 GiB free on /home.")
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
    post_load_started: float | None = None
    file_audit = MedicalImageOpenAudit(
        Path(str(dataset_audit["image_root"])), list(dataset_audit["allowed_image_paths"])
    )
    try:
        phases.mark("model_load_started")
        wrapper, runtime_audit = load_stage2h_runtime(
            args.base_model, seed=42, training=True, gradient_checkpointing=True
        )
        model = wrapper.llm
        phases.mark("model_load_completed")
        post_load_started = time.perf_counter()
        scope = runtime_audit["trainable_scope"]
        _validate_schema(scope)
        trainable_names = list(scope["observed_names"])
        parameters = dict(model.named_parameters())
        trainable_parameters = [parameters[name] for name in trainable_names]
        state_before = trainable_state_record(model)
        if state_before != runtime_audit["step0_state"]:
            raise ManifestContractError("D3R state changed immediately after reset.")
        _exclusive_json(args.state_before_output, state_before)
        outer_rng = snapshot_rng_state()
        rng_before = rng_state_fingerprint(outer_rng)
        model_load = {
            "status": "SUCCESS",
            "schema_version": "cabg-lce-mil-v1.2-d3r-model-load-1",
            "run_id": RUN_ID,
            "source_commit": repository["commit"],
            "source_fingerprint": source["fingerprint"],
            "source_ancestry": [EXPECTED_GATE_D3_COMMIT, EXPECTED_PARENT_COMMIT],
            "approved_r0_equivalence": {
                "local_commit": EXPECTED_PARENT_COMMIT,
                "formal_remote_commit": EXPECTED_FORMAL_R0_COMMIT,
                "approved_tree": EXPECTED_APPROVED_R0_TREE,
                "local_tree": approved_tree,
                "formal_remote_tree": formal_tree,
                "trees_equal": approved_tree == formal_tree == EXPECTED_APPROVED_R0_TREE,
            },
            "checkpoint_audit": checkpoint,
            "checkpoint_fingerprint": canonical_json_sha256(checkpoint),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
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
            "checkpoint_saved": False,
            "official_inference_path_modified": False,
            "quantization": False,
            "offloading": False,
            "pre_model_gpu": pre_model_gpu,
            "home_free_bytes_before_model_load": disk.free,
        }
        if (
            not model_load["model_training"]
            or not model_load["gradient_checkpointing"]
            or model_load["use_cache"]
            or model_load["anomaly_query_mode"] != "single"
            or model_load["fusion_gate_present"]
        ):
            raise ManifestContractError("D3R runtime contract changed.")
        _exclusive_json(args.model_load_audit_output, model_load)
        _, dataset, collator = make_stage2h_dataset(
            args.base_model, "medic_ad_stage2h_calibration", shuffle=False
        )
        expected_rows = [
            (f"cabg-d3r-{row['sample_id']}", str(row["sample_id"]), str(row["relative_path"]), str(row["sha256"]), str(row["d3r_selection_key"]))
            for row in manifest
        ]
        observed_rows = [
            (str(row.get("id")), str(row.get("cabg_sample_id")), str(row.get("image")), str(row.get("image_sha256")), str(row.get("d3r_selection_key")))
            for row in dataset.list_data_dict
        ]
        if len(dataset) != 24 or observed_rows != expected_rows:
            raise ManifestContractError("D3R runtime dataset order or identity changed.")
        file_audit.install()

        for image_index, (manifest_row, label) in enumerate(zip(manifest, labels)):
            started = time.perf_counter()
            phases.mark("image_started", image_index=image_index, sample_id=manifest_row["sample_id"])
            batch = move_batch_to_device(collator([dataset[image_index]]), wrapper.device)
            anomaly_labels = batch.pop("anomaly_labels")
            expected_label = 1 if label == ABNORMAL_LABEL else 0
            if anomaly_labels.numel() != 1 or int(anomaly_labels.item()) != expected_label:
                raise ManifestContractError("D3R runtime label mismatch.")
            batch["tune_mode"] = "default"
            batch["return_anomaly_evidence"] = True
            rng_image_before = rng_state_fingerprint(snapshot_rng_state())
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            forward_started = time.perf_counter()
            with RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
                result = model(**batch)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            if result.loss is None or result.anomaly_evidence is None:
                raise ManifestContractError("D3R forward did not return LM loss and evidence.")
            evidence = result.anomaly_evidence
            observed = observer.finalize(evidence)
            lce = compute_lce(observed["abnormal_raw"], observed["normal_raw"], [label])
            lm_gradients = torch.autograd.grad(
                result.loss, trainable_parameters, retain_graph=True, allow_unused=True
            )
            lce_gradients = torch.autograd.grad(
                lce["per_image_loss"][0], trainable_parameters, retain_graph=False, allow_unused=True
            )
            lm_rows = gradient_rows(trainable_names, lm_gradients)
            lce_rows = gradient_rows(trainable_names, lce_gradients)
            nonzero = int(lce["nonzero_pooling_weights"][0].detach().cpu())
            record = {
                "schema_version": SCHEMA_VERSION,
                "run_id": RUN_ID,
                "status": "SUCCESS",
                "claim_scope": "confirmatory_read_only_mechanism_diagnostic_only",
                "manifest_index": image_index,
                "sample_id": str(manifest_row["sample_id"]),
                "image_sha256": str(manifest_row["sha256"]),
                "relative_path": str(manifest_row["relative_path"]),
                "selection_key": str(manifest_row["d3r_selection_key"]),
                "label": label,
                "forward_id": f"d3r-forward-{image_index:03d}",
                "evidence_fingerprint": tensor_sha256(evidence),
                "lm_loss": float(result.loss.detach().float().cpu()),
                "lm_gradients": lm_rows,
                "lm_global_norm": global_norm(lm_rows, trainable_names),
                "lce": {
                    "mechanism": lce["mechanism"],
                    "score": float(lce["score"][0].detach().cpu()),
                    "loss": float(lce["per_image_loss"][0].detach().cpu()),
                    "gradients": lce_rows,
                    "s9_global_norm": global_norm(lce_rows, S9_NAMES),
                    "structural_zero_name": STRUCTURAL_ZERO_NAME,
                    "spatial": {
                        "entropy": float(lce["spatial_entropy"][0].detach().cpu()),
                        "effective_support": float(lce["effective_support"][0].detach().cpu()),
                        "top11_mass": float(lce["top11_mass"][0].detach().cpu()),
                        "max_pooling_weight": float(lce["max_pooling_weight"][0].detach().cpu()),
                        "min_nonzero_pooling_weight": float(lce["min_nonzero_pooling_weight"][0].detach().cpu()),
                        "nonzero_pooling_weights": nonzero,
                    },
                },
                "formula_payload": tensor_payload(observed, lce),
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
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "checkpoint_fingerprint": model_load["checkpoint_fingerprint"],
                },
            }
            del lce_gradients, lm_gradients, result, lce, evidence, observed, batch, anomaly_labels
            gc.collect()
            torch.cuda.synchronize()
            record["runtime"]["image_seconds"] = time.perf_counter() - started
            record["rng"]["after_image"] = rng_state_fingerprint(snapshot_rng_state())
            records.append(record)
            phases.mark("image_completed", image_index=image_index, sample_id=manifest_row["sample_id"])

        post_load_seconds = time.perf_counter() - post_load_started
        state_after = trainable_state_record(model)
        grad_audit = _state_grad_audit(model, trainable_names)
        state_after["grad_audit"] = grad_audit
        state_after["matches_before"] = state_after["trainable_state_fingerprint"] == state_before["trainable_state_fingerprint"]
        if not state_after["matches_before"] or not grad_audit["all_grad_none"]:
            raise ManifestContractError("D3R parameter-state or .grad contract failed.")
        _exclusive_json(args.state_after_output, state_after)
        file_result = file_audit.result()
        file_result.update(
            {
                "schema_version": "cabg-lce-mil-v1.2-d3r-file-open-1",
                "protected_internal_test_outputs_read": 0,
            }
        )
        if file_result["status"] != "SUCCESS":
            raise ManifestContractError("D3R file-open isolation failed.")
        _exclusive_json(args.file_open_audit_output, file_result)
        restore_rng_state(outer_rng)
        rng_after_restore = rng_state_fingerprint(snapshot_rng_state())
        rng_result = {
            "status": "SUCCESS" if rng_after_restore == rng_before else "FAILED",
            "schema_version": "cabg-lce-mil-v1.2-d3r-rng-1",
            "outer_before": rng_before,
            "after_restore": rng_after_restore,
            "restored_exactly": rng_after_restore == rng_before,
            "natural_sequential_stream": True,
            "per_image_reseeded": False,
        }
        if rng_result["status"] != "SUCCESS":
            raise ManifestContractError("D3R RNG restoration failed.")
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
        record["runtime"]["post_load_diagnostic_seconds"] = post_load_seconds
    if len(records) != 24 or len({row["forward_id"] for row in records}) != 24:
        raise ManifestContractError("D3R did not produce exactly 24 one-forward records.")
    _write_jsonl_exclusive(args.records_output, records)
    result = {
        "status": "SUCCESS",
        "decision": "D3R_PRODUCER_COMPLETE_AWAITING_INDEPENDENT_EVALUATOR",
        "run_id": RUN_ID,
        "claim_scope": "confirmatory_read_only_mechanism_diagnostic_only",
        "images": 24,
        "forwards": 24,
        "records": 24,
        "external_peak_mib": peak,
        "external_peak_limit_mib": EXTERNAL_PEAK_LIMIT_MIB,
        "post_load_diagnostic_seconds": post_load_seconds,
        "records_sha256": sha256_file(args.records_output),
        "independent_evaluator_required": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--approved-preflight", type=Path, required=True)
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
