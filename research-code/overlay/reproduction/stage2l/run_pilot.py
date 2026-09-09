#!/usr/bin/env python3
"""Real-model, two-phase CABG-LCE-MIL v1.2 D4 feasibility pilot."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from reproduction.stage2h.state_audit import tensor_sha256, trainable_state_record
from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2i.rng_state import restore_rng_state, rng_state_fingerprint, snapshot_rng_state
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit, NvidiaSmiMonitor, PhaseMarkers
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2l import RUN_ID, SCHEMA_VERSION
from reproduction.stage2l.checkpoint import load_strict, save_atomic, tensor_inventory
from reproduction.stage2l.constants import (
    ADAM_BETAS, ADAM_EPSILON, LEARNING_RATE, PHASE1_BLOCKS, PILOT_BLOCKS, SEED,
    S9_FINGERPRINT, S9_NAMES, SOURCE_FILES, STRUCTURAL_ZERO_NAME, TRAINABLE_NAMES,
    TRAINABLE_SCHEMA, WEIGHT_DECAY,
)
from reproduction.stage2l.controller import DynamicController, compute_block_gradients, exact_s9_from_full


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CABGContractError(f"Expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _exclusive_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append(path: Path, record: dict[str, object], previous: str) -> str:
    payload = dict(record)
    payload["previous_record_sha256"] = previous
    digest = canonical_json_sha256(payload)
    payload["record_sha256"] = digest
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, raw.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu(member) for key, member in value.items()}
    if isinstance(value, list):
        return [_cpu(member) for member in value]
    if isinstance(value, tuple):
        return tuple(_cpu(member) for member in value)
    return value


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _cosine(step: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * min(step, PILOT_BLOCKS) / PILOT_BLOCKS))


def _parameter_rows(values: Mapping[str, torch.Tensor]) -> list[dict[str, object]]:
    return [{"name": name, "dtype": str(values[name].dtype).removeprefix("torch."), "shape": list(values[name].shape),
             "elements": values[name].numel(), "sha256": tensor_sha256(values[name])} for name in TRAINABLE_NAMES]


def _provenance(preflight: dict[str, Any], dataset: dict[str, Any]) -> dict[str, object]:
    return {
        "source_commit": preflight["repository"]["commit"],
        "implementation_fingerprint": preflight["source"]["fingerprint"],
        "base_checkpoint_fingerprint": canonical_json_sha256(preflight["base_checkpoint"]),
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "configuration_sha256": canonical_json_sha256({
            "seed": SEED, "blocks": PILOT_BLOCKS, "phase1_blocks": PHASE1_BLOCKS,
            "block_composition": dataset["block_composition"], "lr": LEARNING_RATE,
            "betas": ADAM_BETAS, "adam_epsilon": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY,
            "scheduler": "cosine_zero_warmup", "s9_fingerprint": S9_FINGERPRINT,
        }),
    }


def _checkpoint_payload(
    model, masters: Mapping[str, torch.nn.Parameter], optimizer, scheduler, controller,
    *, cursor: int, dataset: dict[str, Any], provenance: dict[str, object], last_hash: str,
) -> dict[str, object]:
    named = dict(model.named_parameters())
    return {
        "schema_version": "cabg-lce-mil-v1.2-d4-checkpoint-1",
        "adapter_state": {name: named[name].detach().cpu().to(torch.bfloat16).clone() for name in TRAINABLE_NAMES},
        "master_state": {name: masters[name].detach().cpu().float().clone() for name in TRAINABLE_NAMES},
        "optimizer_state": _cpu(optimizer.state_dict()),
        "scheduler_state": _cpu(scheduler.state_dict()),
        "controller_state": controller.state_dict(),
        "sampler_state": {"cursor": cursor, "manifest_sha256": dataset["manifest_sha256"],
                          "order_sha256": canonical_json_sha256(_rows(Path(dataset["manifest"])))},
        "rng_state": snapshot_rng_state(),
        "provenance": provenance,
        "trace_state": {"completed_blocks": cursor, "last_record_sha256": last_hash},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    phase = args.phase
    phase_start, phase_end = (0, PHASE1_BLOCKS) if phase == "phase1" else (PHASE1_BLOCKS, PILOT_BLOCKS)
    preflight, dataset_audit = _load(args.preflight), _load(args.dataset_audit)
    manifest = _rows(Path(dataset_audit["manifest"]))
    provenance = _provenance(preflight, dataset_audit)
    if len(manifest) != 12 or Counter(row["d4_label"] for row in manifest) != Counter({"abnormal": 8, "normal": 4}):
        raise CABGContractError("D4 runtime manifest identity changed.")
    expected_schema = [(name, tuple(shape), elements) for name, shape, elements in TRAINABLE_SCHEMA]
    phase_dir = args.run_root / phase
    if phase_dir.exists():
        raise FileExistsError(phase_dir)
    phase_dir.mkdir(parents=True)
    monitor = NvidiaSmiMonitor(phase_dir / "nvidia-smi.csv")
    markers = PhaseMarkers(phase_dir / "phase-markers.jsonl")
    allowed_rows = manifest[phase_start * 3:phase_end * 3]
    allowed_paths = [str((Path(dataset_audit["image_root"]) / row["relative_path"]).resolve()) for row in allowed_rows]
    file_audit = MedicalImageOpenAudit(Path(dataset_audit["image_root"]), allowed_paths)
    monitor.start()
    markers.mark("monitor_started", phase_name=phase)
    wrapper = None
    try:
        os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = str(dataset_audit["annotation"])
        os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = str(dataset_audit["image_root"])
        from reproduction.stage2h.runtime import load_stage2h_runtime, make_stage2h_dataset, move_batch_to_device
        markers.mark("model_load_started")
        wrapper, runtime_audit = load_stage2h_runtime(args.base_model, seed=SEED, training=True, gradient_checkpointing=True)
        model = wrapper.llm
        markers.mark("model_load_completed")
        scope = runtime_audit["trainable_scope"]
        observed_schema = [(row["name"], tuple(row["shape"]), int(row["elements"])) for row in scope["parameters"]]
        if list(scope["observed_names"]) != list(TRAINABLE_NAMES) or observed_schema != expected_schema:
            raise CABGContractError("D4 trainable T21 schema changed.")
        named = dict(model.named_parameters())
        trainable = [named[name] for name in TRAINABLE_NAMES]
        masters = {name: torch.nn.Parameter(named[name].detach().float().clone(), requires_grad=False) for name in TRAINABLE_NAMES}
        optimizer = torch.optim.AdamW(list(masters.values()), lr=LEARNING_RATE, betas=ADAM_BETAS,
                                      eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine)
        controller = DynamicController()
        _, dataset, collator = make_stage2h_dataset(args.base_model, "medic_ad_stage2h_calibration", shuffle=False)
        observed = [(row.get("cabg_sample_id"), row.get("image_sha256"), row.get("d4_block_id"), row.get("d4_block_position")) for row in dataset.list_data_dict]
        expected = [(row["sample_id"], row["sha256"], row["d4_block_id"], row["d4_block_position"]) for row in manifest]
        if observed != expected:
            raise CABGContractError("D4 dataset order/identity changed.")
        previous = "0" * 64
        resume_audit = {"phase": phase, "resumed": False}
        if phase == "phase2":
            payload, checkpoint_manifest = load_strict(args.resume_checkpoint, provenance)
            if payload["sampler_state"]["cursor"] != PHASE1_BLOCKS:
                raise CABGContractError("D4 resume cursor is not the registered phase boundary.")
            for name in TRAINABLE_NAMES:
                masters[name].data.copy_(payload["master_state"][name].to(wrapper.device))
                named[name].data.copy_(payload["adapter_state"][name].to(wrapper.device))
            optimizer.load_state_dict(payload["optimizer_state"])
            _move_optimizer_state(optimizer, wrapper.device)
            scheduler.load_state_dict(payload["scheduler_state"])
            controller.load_state_dict(payload["controller_state"])
            restore_rng_state(payload["rng_state"])
            previous = payload["trace_state"]["last_record_sha256"]
            resume_audit = {
                "phase": phase, "resumed": True, "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
                "cursor": payload["sampler_state"]["cursor"], "controller_state": controller.state_dict(),
                "rng_fingerprint": rng_state_fingerprint(snapshot_rng_state()),
                "checkpoint_rng_fingerprint": checkpoint_manifest["rng_fingerprint"],
                "adapter_exact": all(torch.equal(named[name].detach().cpu(), payload["adapter_state"][name]) for name in TRAINABLE_NAMES),
                "master_exact": all(torch.equal(masters[name].detach().cpu(), payload["master_state"][name]) for name in TRAINABLE_NAMES),
                "optimizer_tensor_inventory_sha256": canonical_json_sha256(tensor_inventory({"optimizer": _cpu(optimizer.state_dict())})),
                "checkpoint_optimizer_tensor_inventory_sha256": canonical_json_sha256(tensor_inventory({"optimizer": payload["optimizer_state"]})),
            }
            if not resume_audit["adapter_exact"] or not resume_audit["master_exact"] or resume_audit["rng_fingerprint"] != resume_audit["checkpoint_rng_fingerprint"] or resume_audit["optimizer_tensor_inventory_sha256"] != resume_audit["checkpoint_optimizer_tensor_inventory_sha256"]:
                raise CABGContractError("D4 fresh-process resume was not exact.")
        elif args.trace.exists():
            raise FileExistsError(args.trace)
        _exclusive_json(phase_dir / "resume-audit.json", resume_audit)
        file_audit.install()
        for block_id in range(phase_start, phase_end):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            block_rows = manifest[block_id * 3:(block_id + 1) * 3]
            labels = [row["d4_label"] for row in block_rows]
            if Counter(labels) != Counter({"normal": 1, "abnormal": 2}):
                raise CABGContractError("D4 logical block composition changed.")
            markers.mark("block_started", block_id=block_id)
            rng_before = rng_state_fingerprint(snapshot_rng_state())
            lm_images, lce_images, image_records = [], [], []
            for local_index, row in enumerate(block_rows):
                dataset_index = block_id * 3 + local_index
                batch = move_batch_to_device(collator([dataset[dataset_index]]), wrapper.device)
                anomaly_labels = batch.pop("anomaly_labels")
                if int(anomaly_labels.item()) != (1 if row["d4_label"] == "abnormal" else 0):
                    raise CABGContractError("D4 label/image alignment changed.")
                batch["tune_mode"] = "default"
                batch["return_anomaly_evidence"] = True
                forward_started = time.perf_counter()
                with RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
                    result = model(**batch)
                observed_maps = observer.finalize(result.anomaly_evidence)
                lce = compute_lce(observed_maps["abnormal_raw"], observed_maps["normal_raw"], [row["d4_label"]])
                lm_raw = torch.autograd.grad(result.loss, trainable, retain_graph=True, allow_unused=True)
                lce_raw = torch.autograd.grad(lce["per_image_loss"][0], trainable, retain_graph=False, allow_unused=True)
                lm_named = {}
                for name, value in zip(TRAINABLE_NAMES, lm_raw):
                    if value is None or not torch.isfinite(value).all():
                        raise CABGContractError(f"D4 LM gradient invalid: {name}")
                    lm_named[name] = value.detach().float().cpu()
                lce_full = dict(zip(TRAINABLE_NAMES, lce_raw))
                lce_named = exact_s9_from_full(lce_full, S9_NAMES)
                lm_images.append(lm_named)
                lce_images.append(lce_named)
                image_records.append({
                    "sample_id": row["sample_id"], "sha256": row["sha256"], "label": row["d4_label"],
                    "lm_loss": float(result.loss.detach()), "lce_loss": float(lce["per_image_loss"][0].detach()),
                    "lce_score": float(lce["score"][0].detach()),
                    "lce_support": {"exact_s9": True, "structural_zero_outside_s9": True,
                                    "structural_zero_name": STRUCTURAL_ZERO_NAME},
                    "effective_support": float(lce["effective_support"][0].detach()),
                    "top11_mass": float(lce["top11_mass"][0].detach()),
                    "forward_seconds": time.perf_counter() - forward_started,
                    "lm_s9_norm": math.sqrt(sum(float(lm_named[name].double().square().sum()) for name in S9_NAMES)),
                    "lce_s9_norm": math.sqrt(sum(float(lce_named[name].double().square().sum()) for name in S9_NAMES)),
                })
                del batch, anomaly_labels, result, lce, observed_maps, lm_raw, lce_raw, lce_full
                gc.collect()
                torch.cuda.synchronize()
            applied, details = compute_block_gradients(
                lm_images, lce_images, labels, trainable_names=TRAINABLE_NAMES,
                s9_names=S9_NAMES, controller=controller,
            )
            audit_payload = {
                "schema_version": SCHEMA_VERSION, "block_id": block_id, "labels": labels,
                "per_image_lm_s9": [{name: row[name] for name in S9_NAMES} for row in lm_images],
                "per_image_lce_s9": lce_images,
                "aggregate_lm": details["aggregate_lm"], "aggregate_lce": details["aggregate_lce"],
            }
            audit_path = phase_dir / f"block-{block_id:04d}-gradient-audit.pt"
            torch.save(audit_payload, audit_path)
            audit_sha = sha256_file(audit_path)
            before = {name: masters[name].detach().clone() for name in TRAINABLE_NAMES}
            optimizer.zero_grad(set_to_none=True)
            for name in TRAINABLE_NAMES:
                masters[name].grad = applied[name].to(wrapper.device)
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            changes, changed = [], 0
            for name in TRAINABLE_NAMES:
                if not torch.isfinite(masters[name]).all():
                    raise CABGContractError(f"D4 master parameter became non-finite: {name}")
                difference = masters[name].detach() - before[name]
                did_change = bool(torch.any(difference != 0))
                changed += int(did_change)
                named[name].data.copy_(masters[name].data.to(torch.bfloat16))
                changes.append({"name": name, "changed": did_change, "l2_change": float(difference.double().norm()),
                                "max_abs_change": float(difference.abs().max()), "master_sha256": tensor_sha256(masters[name]),
                                "adapter_sha256": tensor_sha256(named[name])})
            if changed == 0 or any(parameter.grad is not None for parameter in trainable):
                raise CABGContractError("D4 parameter update or model .grad isolation failed.")
            rng_after = rng_state_fingerprint(snapshot_rng_state())
            record = {
                "schema_version": SCHEMA_VERSION, "status": "SUCCESS", "run_id": RUN_ID,
                "claim_scope": "development_only_training_feasibility", "phase": phase, "block_id": block_id,
                "ordered_samples": [{"sample_id": row["sample_id"], "sha256": row["sha256"], "label": row["d4_label"]} for row in block_rows],
                "images": image_records,
                "controller": details["budget"],
                "class_rms": {"lm": details["lm_class_rms"], "lce": details["lce_class_rms"],
                              "lm_balanced": details["lm_rms"], "lce_balanced": details["lce_rms"]},
                "gradient": details["gradient"],
                "optimizer": {"type": "AdamW", "step": block_id + 1, "learning_rate_used": learning_rate_used,
                              "next_learning_rate": optimizer.param_groups[0]["lr"], "betas": list(ADAM_BETAS), "epsilon": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY},
                "scheduler": {"type": "cosine", "step": scheduler.last_epoch, "zero_warmup": True},
                "parameter_update": {"changed_tensor_count": changed, "tensors": changes},
                "applied_gradient": {"tensor_count": len(applied), "all_finite": all(torch.isfinite(value).all() for value in applied.values()),
                                     "tensor_sha256": {name: tensor_sha256(applied[name]) for name in TRAINABLE_NAMES}},
                "gradient_audit": {"path": str(audit_path.resolve()), "sha256": audit_sha},
                "memory": {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2},
                "runtime": {"block_seconds": time.perf_counter() - started,
                            "images_per_second": 3.0 / (time.perf_counter() - started)},
                "rng": {"before_block": rng_before, "after_block": rng_after},
                "provenance": provenance,
            }
            previous = _append(args.trace, record, previous)
            markers.mark("block_completed", block_id=block_id, record_sha256=previous)
            del before, applied, details, lm_images, lce_images, audit_payload
            gc.collect()
            torch.cuda.empty_cache()
        checkpoint_path = args.checkpoint_output
        checkpoint_payload = _checkpoint_payload(model, masters, optimizer, scheduler, controller,
                                                 cursor=phase_end, dataset=dataset_audit,
                                                 provenance=provenance, last_hash=previous)
        checkpoint_manifest = save_atomic(checkpoint_path, checkpoint_payload)
        file_result = file_audit.result()
        file_result.update({"phase": phase, "protected_internal_test_outputs_read": 0})
        if file_result["status"] != "SUCCESS":
            raise CABGContractError("D4 file-open allowlist was incomplete or crossed.")
        _exclusive_json(phase_dir / "file-open-audit.json", file_result)
        _exclusive_json(phase_dir / "state-final.json", trainable_state_record(model))
        _exclusive_json(phase_dir / "checkpoint-save.json", checkpoint_manifest)
        markers.mark("phase_completed", phase_name=phase, blocks=phase_end - phase_start)
        result = {"status": "SUCCESS", "decision": f"D4_{phase.upper()}_COMPLETE", "phase": phase,
                  "blocks_completed": phase_end - phase_start, "cursor": phase_end,
                  "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
                  "last_record_sha256": previous, "controller_state": controller.state_dict()}
        _exclusive_json(phase_dir / "result.json", result)
        return result
    finally:
        markers.mark("monitor_stopping")
        markers.close()
        monitor_result = monitor.stop()
        if not (phase_dir / "monitor-summary.json").exists():
            _exclusive_json(phase_dir / "monitor-summary.json", monitor_result)
        if wrapper is not None:
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "phase2"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "phase2" and args.resume_checkpoint is None:
        parser.error("phase2 requires --resume-checkpoint")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
