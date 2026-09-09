#!/usr/bin/env python3
"""Read-only, one-forward-per-image CABG-MIL v1.1 Gate D3 diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

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
from reproduction.stage2i.d3_artifact_schema import write_records_exclusive
from reproduction.stage2i.d3_mechanisms import (
    MECHANISM_IDS,
    cancellation_ratio,
    compute_mechanisms,
    global_norm,
    gradient_cosine,
    gradient_statistics,
)
from reproduction.stage2i.d3_observer import AnomalyAttentionObserver
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, tensor_sha256
from reproduction.stage2i.rng_state import (
    restore_rng_state,
    rng_state_fingerprint,
    snapshot_rng_state,
)


SATURATION_LIMIT = 0.10
GRADIENT_FLOOR = 1e-12
EXTERNAL_PEAK_LIMIT_MIB = 22500


def _exclusive_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        payload = json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8") + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CABGContractError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CABGContractError(f"Expected non-empty JSONL objects: {path}")
    return rows


class MedicalImageOpenAudit:
    def __init__(self, image_root: Path, allowed_paths: list[str]):
        self.image_root = image_root.resolve()
        self.allowed = {str(Path(value).resolve()) for value in allowed_paths}
        self.events: list[str] = []
        self.installed = False

    def _hook(self, event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(raw)).resolve()
            path.relative_to(self.image_root)
        except (OSError, ValueError):
            return
        canonical = str(path)
        self.events.append(canonical)
        if canonical not in self.allowed:
            raise CABGContractError(f"D3 attempted to open a non-allowlisted medical image: {canonical}")

    def install(self) -> None:
        if self.installed:
            raise CABGContractError("D3 file-open audit cannot be installed twice.")
        sys.addaudithook(self._hook)
        self.installed = True

    def result(self) -> dict[str, object]:
        opened = sorted(set(self.events))
        return {
            "status": "SUCCESS" if set(opened) == self.allowed else "FAILED",
            "schema_version": "cabg-v1.1-gate-d3-file-open-1",
            "image_root": str(self.image_root),
            "allowed_paths": sorted(self.allowed),
            "opened_paths": opened,
            "open_event_count": len(self.events),
            "unique_opened_count": len(opened),
            "non_allowlisted_paths": sorted(set(opened) - self.allowed),
            "unopened_allowlisted_paths": sorted(self.allowed - set(opened)),
            "internal_test_outputs_read": 0,
        }


class NvidiaSmiMonitor:
    def __init__(self, output: Path):
        self.output = output
        self._stream = None
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output.open("x", encoding="utf-8")
        self._stream.write("timestamp,memory_used_mib,memory_total_mib,utilization_gpu_percent\n")
        self._stream.flush()
        command = (
            "while true; do "
            "nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader,nounits; sleep 0.1; done"
        )
        self._process = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def stop(self) -> dict[str, object]:
        if self._process is None or self._stream is None:
            raise CABGContractError("D3 external monitor was not started.")
        if self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=10)
        self._stream.flush()
        self._stream.close()
        samples = []
        invalid = []
        for index, line in enumerate(self.output.read_text(encoding="utf-8").splitlines()[1:], 2):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                invalid.append({"line": index, "value": line})
                continue
            try:
                samples.append(
                    {
                        "timestamp": parts[0],
                        "memory_used_mib": int(parts[1]),
                        "memory_total_mib": int(parts[2]),
                        "utilization_gpu_percent": int(parts[3]),
                    }
                )
            except ValueError:
                invalid.append({"line": index, "value": line})
        if not samples or invalid:
            raise CABGContractError(
                f"D3 external GPU monitor is incomplete: samples={len(samples)}, invalid={invalid[:3]}"
            )
        return {
            "status": "SUCCESS",
            "schema_version": "cabg-v1.1-gate-d3-monitor-1",
            "sample_count": len(samples),
            "sampling_target_seconds": 0.1,
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in samples),
            "memory_total_mib": max(row["memory_total_mib"] for row in samples),
            "max_utilization_gpu_percent": max(row["utilization_gpu_percent"] for row in samples),
            "csv_path": str(self.output.resolve()),
            "csv_sha256": sha256_file(self.output),
        }


class PhaseMarkers:
    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)

    def mark(self, phase: str, **fields: object) -> None:
        payload = {
            "phase": phase,
            "unix_time": time.time(),
            "monotonic_seconds": time.monotonic(),
            **fields,
        }
        raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        os.write(self._descriptor, raw.encode("utf-8"))
        os.fsync(self._descriptor)

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None


def _tensor_metrics(value: torch.Tensor, *, saturation: str) -> dict[str, float]:
    working = value.detach().float()
    if not torch.isfinite(working).all():
        raise CABGContractError("D3 metric tensor contains NaN or Inf.")
    if saturation == "evidence":
        fraction = (working.abs() >= 0.95).float().mean()
    elif saturation == "probability":
        fraction = ((working <= 0.01) | (working >= 0.99)).float().mean()
    else:
        raise CABGContractError(f"Unknown D3 saturation metric: {saturation}")
    return {
        "min": float(working.min().cpu().item()),
        "max": float(working.max().cpu().item()),
        "mean": float(working.mean().cpu().item()),
        "saturation_fraction": float(fraction.cpu().item()),
    }


def _state_grad_audit(model: torch.nn.Module, names: list[str]) -> dict[str, object]:
    parameters = dict(model.named_parameters())
    rows = [
        {"name": name, "grad_is_none": parameters[name].grad is None}
        for name in names
    ]
    return {
        "status": "SUCCESS" if all(row["grad_is_none"] for row in rows) else "FAILED",
        "parameter_count": len(rows),
        "all_grad_none": all(row["grad_is_none"] for row in rows),
        "parameters": rows,
    }


def _class_aggregate(
    labels: list[str],
    mechanism_gradients: Mapping[str, list[Mapping[str, torch.Tensor]]],
    records: list[dict[str, object]],
) -> dict[str, object]:
    by_mechanism = {}
    for mechanism_id in MECHANISM_IDS:
        rows = [row for row in records if row["mechanism_id"] == mechanism_id]
        classes = {}
        for label in (NORMAL_LABEL, ABNORMAL_LABEL):
            indices = [index for index, value in enumerate(labels) if value == label]
            class_gradients = [mechanism_gradients[mechanism_id][index] for index in indices]
            class_rows = [row for row in rows if row["label"] == label]
            squared_norms = [
                float(row["gradients"]["auxiliary_shared_global_norm"]) ** 2
                for row in class_rows
            ]
            classes[label] = {
                "image_count": len(indices),
                "shared_support_rms": math.sqrt(sum(squared_norms) / len(squared_norms)),
                "cancellation": cancellation_ratio(class_gradients),
                "effective_images": sum(
                    float(row["gradients"]["auxiliary_shared_global_norm"]) > GRADIENT_FLOOR
                    for row in class_rows
                ),
                "effective_by_shared_tensor": {
                    name: any(
                        gradient[name].abs().max().item() > GRADIENT_FLOOR
                        for gradient in class_gradients
                    )
                    for name in SHARED_SUPPORT_NAMES
                },
            }
        by_mechanism[mechanism_id] = {"classes": classes}
    m2_rows = [row for row in records if row["mechanism_id"] == "M2"]
    saturation = {}
    for key in (
        "evidence_saturation_fraction",
        "abnormal_probability_saturation_fraction",
        "normal_probability_saturation_fraction",
    ):
        values = [float(row["evidence"][key]) for row in m2_rows]
        saturation[key] = {
            "mean_fraction": sum(values) / len(values),
            "max_per_image_fraction": max(values),
        }
    return {
        "status": "SUCCESS",
        "schema_version": "cabg-v1.1-gate-d3-aggregate-1",
        "image_count": len(labels),
        "class_counts": dict(sorted(Counter(labels).items())),
        "record_count": len(records),
        "mechanisms": by_mechanism,
        "saturation": saturation,
        "m2_non_shared_leakage_count": sum(
            int(row["gradients"]["non_shared_leakage_count"]) for row in m2_rows
        ),
        "m2_exact_one_position_collapse_count": sum(
            bool(row["spatial"]["exact_one_position_collapse"]) for row in m2_rows
        ),
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    required_outputs = (
        args.records_output,
        args.aggregate_output,
        args.model_load_audit_output,
        args.state_before_output,
        args.state_after_output,
        args.file_open_audit_output,
        args.rng_audit_output,
        args.monitor_summary_output,
    )
    if any(path.exists() for path in required_outputs):
        raise FileExistsError("Refusing to overwrite Gate D3 runtime evidence.")
    preflight = _load_json(args.preflight)
    if preflight.get("decision") != "READY_FOR_GATE_D3_MODEL_LOAD":
        raise CABGContractError("Gate D3 preflight did not authorize model loading.")
    dataset_audit = _load_json(args.dataset_audit)
    manifest = _load_jsonl(Path(str(dataset_audit["locked_manifest"])))
    if len(manifest) != 12:
        raise CABGContractError("Gate D3 runtime requires exactly 12 locked rows.")
    labels = [normalize_label(row["label"]) for row in manifest]
    if Counter(labels) != Counter({NORMAL_LABEL: 6, ABNORMAL_LABEL: 6}):
        raise CABGContractError("Gate D3 runtime class balance changed.")
    os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = str(dataset_audit["annotation"])
    os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = str(dataset_audit["image_root"])
    # Import after the locked registry environment is set: qwenvl.data resolves
    # these paths at module-import time.
    from reproduction.stage2h.runtime import (
        load_stage2h_runtime,
        make_stage2h_dataset,
        move_batch_to_device,
    )

    monitor = NvidiaSmiMonitor(args.monitor_output)
    phases = PhaseMarkers(args.phase_markers_output)
    monitor.start()
    phases.mark("monitor_started")
    wrapper = None
    outer_rng = None
    rng_after_images = None
    records: list[dict[str, object]] = []
    mechanism_gradients: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
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
        scope = runtime_audit["trainable_scope"]
        trainable_names = list(scope["observed_names"])
        if (
            len(trainable_names) != EXPECTED_TRAINABLE_TENSORS
            or int(scope["trainable_parameter_elements"]) != EXPECTED_TRAINABLE_ELEMENTS
            or not set(SHARED_SUPPORT_NAMES).issubset(trainable_names)
        ):
            raise CABGContractError("D3 observed trainable/shared support changed.")
        parameters = dict(model.named_parameters())
        trainable_parameters = [parameters[name] for name in trainable_names]
        shared_parameters = [parameters[name] for name in SHARED_SUPPORT_NAMES]
        state_before = trainable_state_record(model)
        if state_before != runtime_audit["step0_state"]:
            raise CABGContractError("D3 model state changed between reset and state capture.")
        _exclusive_json(args.state_before_output, state_before)
        outer_rng = snapshot_rng_state()
        rng_outer_before = rng_state_fingerprint(outer_rng)
        model_load_audit = {
            "status": "SUCCESS",
            "schema_version": "cabg-v1.1-gate-d3-model-load-1",
            "runtime_audit": runtime_audit,
            "model_dtype": "bfloat16",
            "model_training": bool(model.training),
            "gradient_checkpointing": bool(runtime_audit["gradient_checkpointing"]),
            "use_cache": bool(model.config.use_cache),
            "anomaly_query_mode": model.visual.anomaly_query_mode,
            "num_pooling_size": int(model.visual.num_pooling_size),
            "output_token_count": 16,
            "fusion_gate_present": getattr(model.visual.anomaly_qformer, "fusion_gate", None) is not None,
            "optimizer_created": False,
            "scheduler_created": False,
            "parameter_update_performed": False,
        }
        if (
            not model_load_audit["model_training"]
            or not model_load_audit["gradient_checkpointing"]
            or model_load_audit["use_cache"]
            or model_load_audit["anomaly_query_mode"] != "single"
            or model_load_audit["num_pooling_size"] != 4
            or model_load_audit["fusion_gate_present"]
        ):
            raise CABGContractError(f"D3 runtime configuration changed: {model_load_audit}")
        _exclusive_json(args.model_load_audit_output, model_load_audit)

        _, dataset, collator = make_stage2h_dataset(
            args.base_model, "medic_ad_stage2h_calibration", shuffle=False
        )
        if len(dataset) != 12 or len(dataset.list_data_dict) != 12:
            raise CABGContractError("D3 generated dataset does not contain exactly 12 images.")
        expected_dataset_rows = [
            (f"cabg-d3-{row['sample_id']}", str(row["sample_id"]), str(row["relative_path"]), str(row["sha256"]))
            for row in manifest
        ]
        observed_dataset_rows = [
            (
                str(row.get("id")),
                str(row.get("cabg_sample_id")),
                str(row.get("image")),
                str(row.get("image_sha256")),
            )
            for row in dataset.list_data_dict
        ]
        if observed_dataset_rows != expected_dataset_rows:
            raise CABGContractError("D3 dataset order or identity changed.")
        file_audit.install()
        support_fingerprint = canonical_json_sha256(list(SHARED_SUPPORT_NAMES))
        source_commit = str(preflight["git"]["commit"])
        implementation_fingerprint = str(preflight["source"]["fingerprint"])
        dataset_manifest_sha256 = str(preflight["dataset_manifest_sha256"])

        for image_index, (row, label) in enumerate(zip(manifest, labels)):
            forward_id = f"forward-{image_index:03d}"
            phases.mark("image_started", image_index=image_index, sample_id=row["sample_id"])
            image_started = time.perf_counter()
            batch = move_batch_to_device(collator([dataset[image_index]]), wrapper.device)
            anomaly_labels = batch.pop("anomaly_labels")
            expected_binary = 1 if label == ABNORMAL_LABEL else 0
            if anomaly_labels.numel() != 1 or int(anomaly_labels.item()) != expected_binary:
                raise CABGContractError(f"D3 runtime label mismatch at image {image_index}.")
            batch["tune_mode"] = "default"
            batch["return_anomaly_evidence"] = True
            rng_before_image = rng_state_fingerprint(snapshot_rng_state())
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            forward_started = time.perf_counter()
            observer_module = model.visual.anomaly_qformer.anomaly_attention
            with AnomalyAttentionObserver(observer_module) as observer:
                outputs = model(**batch)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            if outputs.loss is None or outputs.anomaly_evidence is None:
                raise CABGContractError("D3 forward did not return LM loss and anomaly evidence.")
            evidence = outputs.anomaly_evidence
            observed = observer.finalize(evidence)
            evidence_fingerprint = tensor_sha256(evidence)
            loss_started = time.perf_counter()
            mechanisms = compute_mechanisms(evidence, [label])
            torch.cuda.synchronize()
            all_mechanism_loss_seconds = time.perf_counter() - loss_started
            lm_gradients = torch.autograd.grad(
                outputs.loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            lm_gradient_rows, lm_gradient_map = gradient_statistics(
                SHARED_SUPPORT_NAMES, lm_gradients
            )
            lm_norm = global_norm(lm_gradient_map)
            evidence_metrics = _tensor_metrics(evidence, saturation="evidence")
            abnormal_metrics = _tensor_metrics(
                observed["abnormal_probability"], saturation="probability"
            )
            normal_metrics = _tensor_metrics(
                observed["normal_probability"], saturation="probability"
            )
            image_records = []
            for mechanism_index, mechanism_id in enumerate(MECHANISM_IDS):
                gradient_started = time.perf_counter()
                auxiliary_gradients = torch.autograd.grad(
                    mechanisms[mechanism_id]["per_image_loss"][0],
                    trainable_parameters,
                    retain_graph=mechanism_index < len(MECHANISM_IDS) - 1,
                    allow_unused=True,
                )
                torch.cuda.synchronize()
                gradient_seconds = time.perf_counter() - gradient_started
                auxiliary_rows, auxiliary_map = gradient_statistics(
                    trainable_names, auxiliary_gradients
                )
                auxiliary_shared = {
                    name: auxiliary_map[name]
                    for name in SHARED_SUPPORT_NAMES
                    if name in auxiliary_map
                }
                non_shared = set(trainable_names) - set(SHARED_SUPPORT_NAMES)
                leakage = [
                    value
                    for value in auxiliary_rows
                    if value["name"] in non_shared
                    and (
                        not value["finite"]
                        or (value["present"] and float(value["max_abs"]) != 0.0)
                    )
                ]
                cosine = gradient_cosine(lm_gradient_map, auxiliary_shared, floor=GRADIENT_FLOOR)
                if set(auxiliary_shared) != set(SHARED_SUPPORT_NAMES):
                    raise CABGContractError(
                        f"D3 {mechanism_id} shared-support gradient is structurally absent."
                    )
                mechanism_gradients[mechanism_id].append(auxiliary_shared)
                spatial = None
                if mechanism_id == "M2":
                    nonzero = int(mechanisms[mechanism_id]["nonzero_pooling_weights"][0].item())
                    spatial = {
                        "entropy": float(mechanisms[mechanism_id]["spatial_entropy"][0].detach().cpu().item()),
                        "effective_support": float(mechanisms[mechanism_id]["effective_support"][0].detach().cpu().item()),
                        "top11_mass": float(mechanisms[mechanism_id]["top11_mass"][0].detach().cpu().item()),
                        "max_pooling_weight": float(mechanisms[mechanism_id]["max_pooling_weight"][0].detach().cpu().item()),
                        "min_nonzero_pooling_weight": float(mechanisms[mechanism_id]["min_nonzero_pooling_weight"][0].detach().cpu().item()),
                        "nonzero_pooling_weights": nonzero,
                        "exact_one_position_collapse": nonzero == 1,
                    }
                image_records.append(
                    {
                        "schema_version": "cabg-v1.1-gate-d3-record-1",
                        "status": "SUCCESS",
                        "sample_id": str(row["sample_id"]),
                        "image_sha256": str(row["sha256"]),
                        "relative_path": str(row["relative_path"]),
                        "label": label,
                        "mechanism_id": mechanism_id,
                        "mechanism": str(mechanisms[mechanism_id]["mechanism"]),
                        "forward_id": forward_id,
                        "evidence_fingerprint": evidence_fingerprint,
                        "losses": {
                            "lm": float(outputs.loss.detach().float().cpu().item()),
                            "auxiliary": float(
                                mechanisms[mechanism_id]["per_image_loss"][0].detach().cpu().item()
                            ),
                        },
                        "score": float(mechanisms[mechanism_id]["score"][0].detach().cpu().item()),
                        "gradients": {
                            "lm_shared_global_norm": lm_norm,
                            "auxiliary_shared_global_norm": global_norm(auxiliary_shared),
                            "cosine_defined": bool(cosine["defined"]),
                            "cosine": cosine["value"],
                            "lm_shared_per_tensor": lm_gradient_rows,
                            "auxiliary_all_trainable_per_tensor": auxiliary_rows,
                            "non_shared_leakage_count": len(leakage),
                        },
                        "evidence": {
                            "evidence_min": evidence_metrics["min"],
                            "evidence_max": evidence_metrics["max"],
                            "evidence_mean": evidence_metrics["mean"],
                            "evidence_saturation_fraction": evidence_metrics["saturation_fraction"],
                            "abnormal_probability_min": abnormal_metrics["min"],
                            "abnormal_probability_max": abnormal_metrics["max"],
                            "abnormal_probability_mean": abnormal_metrics["mean"],
                            "abnormal_probability_saturation_fraction": abnormal_metrics["saturation_fraction"],
                            "normal_probability_min": normal_metrics["min"],
                            "normal_probability_max": normal_metrics["max"],
                            "normal_probability_mean": normal_metrics["mean"],
                            "normal_probability_saturation_fraction": normal_metrics["saturation_fraction"],
                            "reconstruction_exact": bool(observed["reconstruction_exact"]),
                            "reconstruction_max_abs": float(observed["reconstruction_max_abs"]),
                        },
                        "spatial": spatial,
                        "memory": {
                            "image_peak_allocated_mib": 0.0,
                            "image_peak_reserved_mib": 0.0,
                            "run_external_peak_mib": 0.0,
                        },
                        "runtime": {
                            "forward_seconds": forward_seconds,
                            "mechanism_loss_seconds": all_mechanism_loss_seconds,
                            "mechanism_gradient_seconds": gradient_seconds,
                            "image_seconds": 0.0,
                        },
                        "rng": {"before_image": rng_before_image, "after_image": "0" * 64},
                        "provenance": {
                            "source_commit": source_commit,
                            "implementation_fingerprint": implementation_fingerprint,
                            "dataset_manifest_sha256": dataset_manifest_sha256,
                            "support_fingerprint": support_fingerprint,
                        },
                    }
                )
            del auxiliary_gradients, lm_gradients, outputs, mechanisms, evidence, observed, batch, anomaly_labels
            gc.collect()
            torch.cuda.synchronize()
            image_seconds = time.perf_counter() - image_started
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
            rng_after_image = rng_state_fingerprint(snapshot_rng_state())
            for record in image_records:
                record["memory"]["image_peak_allocated_mib"] = peak_allocated
                record["memory"]["image_peak_reserved_mib"] = peak_reserved
                record["runtime"]["image_seconds"] = image_seconds
                record["rng"]["after_image"] = rng_after_image
            records.extend(image_records)
            phases.mark(
                "image_completed",
                image_index=image_index,
                sample_id=row["sample_id"],
                peak_allocated_mib=peak_allocated,
                peak_reserved_mib=peak_reserved,
            )

        rng_after_images = snapshot_rng_state()
        state_after = trainable_state_record(model)
        state_grad_audit = _state_grad_audit(model, trainable_names)
        state_after["grad_audit"] = state_grad_audit
        state_after["matches_before"] = (
            state_after["trainable_state_fingerprint"] == state_before["trainable_state_fingerprint"]
        )
        if not state_after["matches_before"] or not state_grad_audit["all_grad_none"]:
            raise CABGContractError("D3 read-only parameter-state or .grad contract failed.")
        _exclusive_json(args.state_after_output, state_after)
        file_result = file_audit.result()
        if file_result["status"] != "SUCCESS":
            raise CABGContractError(f"D3 file-open audit failed: {file_result}")
        _exclusive_json(args.file_open_audit_output, file_result)
        restore_rng_state(outer_rng)
        rng_after_restore = rng_state_fingerprint(snapshot_rng_state())
        rng_result = {
            "status": "SUCCESS" if rng_after_restore == rng_outer_before else "FAILED",
            "schema_version": "cabg-v1.1-gate-d3-rng-1",
            "outer_before": rng_outer_before,
            "after_images_before_restore": rng_state_fingerprint(rng_after_images),
            "after_restore": rng_after_restore,
            "restored_exactly": rng_after_restore == rng_outer_before,
            "natural_sequential_stream": True,
            "per_mechanism_reseeded": False,
        }
        if rng_result["status"] != "SUCCESS":
            raise CABGContractError("D3 outer RNG restoration failed.")
        _exclusive_json(args.rng_audit_output, rng_result)
        aggregate = _class_aggregate(labels, mechanism_gradients, records)
        phases.mark("diagnostic_completed")
        del (
            observer,
            observer_module,
            parameters,
            trainable_parameters,
            shared_parameters,
            lm_gradient_map,
            auxiliary_map,
            auxiliary_shared,
            dataset,
            collator,
        )
        del model, wrapper
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
    external_peak = float(monitor_result["peak_memory_used_mib"])
    for record in records:
        record["memory"]["run_external_peak_mib"] = external_peak
    if len(records) != 36:
        raise CABGContractError(f"D3 diagnostic produced {len(records)} records instead of 36.")
    write_records_exclusive(args.records_output, records)
    aggregate["external_peak_mib"] = external_peak
    aggregate["external_peak_limit_mib"] = EXTERNAL_PEAK_LIMIT_MIB
    aggregate["external_peak_within_limit"] = external_peak <= EXTERNAL_PEAK_LIMIT_MIB
    aggregate["saturation_limit"] = SATURATION_LIMIT
    aggregate["gradient_floor"] = GRADIENT_FLOOR
    _exclusive_json(args.aggregate_output, aggregate)
    result = {
        "status": "SUCCESS",
        "stage": "D3",
        "records": len(records),
        "images": 12,
        "forwards": 12,
        "external_peak_mib": external_peak,
        "records_sha256": sha256_file(args.records_output),
        "aggregate_sha256": sha256_file(args.aggregate_output),
        "runner_decision": "NOT_AUTHORITATIVE_INDEPENDENT_VERIFICATION_REQUIRED",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--model-load-audit-output", type=Path, required=True)
    parser.add_argument("--state-before-output", type=Path, required=True)
    parser.add_argument("--state-after-output", type=Path, required=True)
    parser.add_argument("--file-open-audit-output", type=Path, required=True)
    parser.add_argument("--rng-audit-output", type=Path, required=True)
    parser.add_argument("--monitor-output", type=Path, required=True)
    parser.add_argument("--monitor-summary-output", type=Path, required=True)
    parser.add_argument("--phase-markers-output", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
