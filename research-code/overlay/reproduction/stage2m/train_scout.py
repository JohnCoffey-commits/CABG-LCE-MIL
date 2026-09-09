#!/usr/bin/env python3
"""Matched two-phase trainer for the development-only D4-Scout."""

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
from reproduction.stage2l.checkpoint import load_strict, save_atomic, tensor_inventory
from reproduction.stage2l.constants import CHECKPOINT_SCHEMA_VERSION
from reproduction.stage2l.controller import DynamicController, compute_block_gradients, exact_s9_from_full
from reproduction.stage2m import RUN_ID, SCHEMA_VERSION
from reproduction.stage2m.constants import (
    ADAM_BETAS,
    ADAM_EPSILON,
    EXTERNAL_PEAK_LIMIT_MIB,
    LEARNING_RATE,
    PHASE1_BLOCKS,
    RHO_MAX,
    SCOUT_BLOCKS,
    SEED,
    S9_FINGERPRINT,
    S9_NAMES,
    STRUCTURAL_ZERO_NAME,
    TRAINABLE_NAMES,
    TRAINABLE_SCHEMA,
    WEIGHT_DECAY,
)
from reproduction.stage2m.training_math import compute_lm_only_block
from reproduction.stage2n import (
    ADJUSTMENT_RUN_ID,
    ADJUSTMENT_SCHEMA_VERSION,
    MOMENT_RESET_ARM,
    REPAIR_RUN_ID,
    REPAIR_SCHEMA_VERSION,
    SUPPORTED_REPAIR_ARMS,
)
from reproduction.stage2n.repair import (
    apply_repair_multiplier,
    initial_repair_state,
    repair_decision,
    validate_repair_state,
)
from reproduction.stage2o import CAUSAL_ARMS, RUN_ID as CAUSAL_RUN_ID, SCHEMA_VERSION as CAUSAL_SCHEMA_VERSION
from reproduction.stage2o.causal import (
    apply_causal_gradient,
    apply_optimizer_intervention,
    load_locked_parent,
    validate_parent_audit,
)
from reproduction.stage2p import (
    ARMS as FIXED_ARMS, RUN_ID as FIXED_RUN_ID, SCHEMA_VERSION as FIXED_SCHEMA,
    CLAIM_SCOPE as FIXED_SCOPE, INITIAL_STATE_SHA256, INITIAL_RNG_SHA256,
)
from reproduction.stage2p.schedule import apply_fixed_cutoff, validate_origin


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


def reset_s9_directional_moments(
    optimizer: torch.optim.Optimizer,
    masters: Mapping[str, torch.nn.Parameter],
    names: tuple[str, ...] = S9_NAMES,
) -> dict[str, object]:
    reset_names: list[str] = []
    pre_reset_nonzero = 0
    for name in names:
        state = optimizer.state.get(masters[name])
        if not isinstance(state, dict) or "exp_avg" not in state:
            raise CABGContractError(f"Missing AdamW first moment for registered reset: {name}")
        first_moment = state["exp_avg"]
        if not isinstance(first_moment, torch.Tensor) or not torch.isfinite(first_moment).all():
            raise CABGContractError(f"Invalid AdamW first moment for registered reset: {name}")
        pre_reset_nonzero += int(bool(torch.any(first_moment != 0)))
        first_moment.zero_()
        if torch.any(first_moment != 0):
            raise CABGContractError(f"AdamW first moment reset failed: {name}")
        reset_names.append(name)
    return {
        "applied": True,
        "parameter_count": len(reset_names),
        "state_tensor_count": len(reset_names),
        "state_key": "exp_avg",
        "reset_names": reset_names,
        "pre_reset_nonzero_parameter_count": pre_reset_nonzero,
        "exp_avg_sq_preserved": True,
        "step_preserved": True,
        "post_reset_all_zero": True,
    }


def cosine_multiplier(step: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * min(step, SCOUT_BLOCKS) / SCOUT_BLOCKS))


def matched_configuration() -> dict[str, object]:
    return {
        "seed": SEED,
        "blocks": SCOUT_BLOCKS,
        "phase1_blocks": PHASE1_BLOCKS,
        "block_composition": {"normal": 1, "abnormal": 2},
        "trainable_names": list(TRAINABLE_NAMES),
        "learning_rate": LEARNING_RATE,
        "betas": list(ADAM_BETAS),
        "adam_epsilon": ADAM_EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "cosine_zero_warmup_horizon_24",
        "global_clip_norm": 1.0,
        "s9_fingerprint": S9_FINGERPRINT,
        "diagnostic_lce_gradient_both_arms": True,
    }


def provenance(preflight: Mapping[str, Any], split: Mapping[str, Any], arm: str) -> dict[str, object]:
    objectives = {
        "cabg_lce": "lm_plus_dynamic_cabg_lce",
        "lm_only": "lm_only",
        "cabg_cosine_taper": "lm_plus_dynamic_cabg_lce_fixed_cosine_taper",
        "cabg_spatial_guard": "lm_plus_dynamic_cabg_lce_persistent_spatial_guard",
        MOMENT_RESET_ARM: "lm_plus_dynamic_cabg_lce_persistent_spatial_guard_with_one_time_s9_exp_avg_reset",
        "cabg_continue_control": "shared_block12_continue_dynamic_cabg_lce",
        "lce_off_state_kept": "shared_block12_lm_only_applied_keep_all_optimizer_state",
        "lce_off_s9_exp_avg_reset": "shared_block12_lm_only_applied_reset_s9_exp_avg_once",
        "cabg_original_control": "from_initialization_original_dynamic_cabg_lce_24_blocks",
        "cabg_fixed_cutoff": "from_initialization_dynamic_cabg_lce_blocks1_12_lm_only_applied_blocks13_24_keep_state",
    }
    if arm not in objectives:
        raise CABGContractError(f"Unknown Scout arm: {arm}")
    return {
        "source_commit": preflight["repository"]["commit"],
        "implementation_fingerprint": preflight["source"]["fingerprint"],
        "base_checkpoint_fingerprint": canonical_json_sha256(preflight["base_checkpoint"]),
        "train_manifest_sha256": split["train_manifest_sha256"],
        "eval_manifest_sha256": split["eval_manifest_sha256"],
        "matched_configuration_sha256": canonical_json_sha256(matched_configuration()),
        "arm": arm,
        "applied_objective": objectives[arm],
    }


def _checkpoint_payload(
    model, masters: Mapping[str, torch.nn.Parameter], optimizer, scheduler, controller,
    *, cursor: int, split: Mapping[str, Any], run_provenance: Mapping[str, object], last_hash: str,
    repair_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    named = dict(model.named_parameters())
    sampler_state: dict[str, object] = {
        "cursor": cursor,
        "manifest_sha256": split["train_manifest_sha256"],
        "order_sha256": canonical_json_sha256(_rows(Path(split["train_manifest"]))),
    }
    if repair_state is not None:
        sampler_state["repair_state"] = dict(repair_state)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_state": {name: named[name].detach().cpu().to(torch.bfloat16).clone() for name in TRAINABLE_NAMES},
        "master_state": {name: masters[name].detach().cpu().float().clone() for name in TRAINABLE_NAMES},
        "optimizer_state": _cpu(optimizer.state_dict()),
        "scheduler_state": _cpu(scheduler.state_dict()),
        "controller_state": controller.state_dict(),
        "sampler_state": sampler_state,
        "rng_state": snapshot_rng_state(),
        "provenance": dict(run_provenance),
        "trace_state": {"completed_blocks": cursor, "last_record_sha256": last_hash},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    phase_start, phase_end = (0, PHASE1_BLOCKS) if args.phase == "phase1" else (PHASE1_BLOCKS, SCOUT_BLOCKS)
    preflight, split = _load(args.preflight), _load(args.split_audit)
    manifest = _rows(Path(split["train_manifest"]))
    run_provenance = provenance(preflight, split, args.arm)
    if len(manifest) != 72 or Counter(row["scout_label"] for row in manifest) != Counter({"abnormal": 48, "normal": 24}):
        raise CABGContractError("Scout runtime training manifest identity changed.")
    expected_schema = [(name, tuple(shape), elements) for name, shape, elements in TRAINABLE_SCHEMA]
    phase_dir = args.run_root / args.arm / args.phase
    if args.arm in FIXED_ARMS:
        validate_origin(args.arm, args.phase, args.run_root, args.resume_checkpoint, args.parent_audit)
    if phase_dir.exists():
        raise FileExistsError(phase_dir)
    phase_dir.mkdir(parents=True)
    monitor = NvidiaSmiMonitor(phase_dir / "nvidia-smi.csv")
    markers = PhaseMarkers(phase_dir / "phase-markers.jsonl")
    phase_rows = manifest[phase_start * 3:phase_end * 3]
    allowed_paths = [str((Path(split["image_root"]) / row["relative_path"]).resolve()) for row in phase_rows]
    file_audit = MedicalImageOpenAudit(Path(split["image_root"]), allowed_paths)
    monitor.start()
    monitor_active = True
    markers.mark("monitor_started", phase_name=args.phase, arm=args.arm)
    wrapper = None
    try:
        os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = str(split["train_annotation"])
        os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = str(split["image_root"])
        from reproduction.stage2h.runtime import load_stage2h_runtime, make_stage2h_dataset, move_batch_to_device

        markers.mark("model_load_started")
        wrapper, runtime_audit = load_stage2h_runtime(args.base_model, seed=SEED, training=True, gradient_checkpointing=True)
        model = wrapper.llm
        if args.arm in FIXED_ARMS:
            if canonical_json_sha256(runtime_audit["step0_state"]) != INITIAL_STATE_SHA256:
                raise CABGContractError("Fixed-cutoff original initialization mismatch.")
            _exclusive_json(phase_dir / "initial-state.json", runtime_audit["step0_state"])
        markers.mark("model_load_completed")
        scope = runtime_audit["trainable_scope"]
        observed_schema = [(row["name"], tuple(row["shape"]), int(row["elements"])) for row in scope["parameters"]]
        if list(scope["observed_names"]) != list(TRAINABLE_NAMES) or observed_schema != expected_schema:
            raise CABGContractError("Scout trainable T21 schema changed.")
        named = dict(model.named_parameters())
        trainable = [named[name] for name in TRAINABLE_NAMES]
        masters = {name: torch.nn.Parameter(named[name].detach().float().clone(), requires_grad=False) for name in TRAINABLE_NAMES}
        optimizer = torch.optim.AdamW(list(masters.values()), lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_multiplier)
        controller = DynamicController()
        repair_state = None if args.arm in (*CAUSAL_ARMS, *FIXED_ARMS) else initial_repair_state(args.arm)
        _, dataset, collator = make_stage2h_dataset(args.base_model, "medic_ad_stage2h_calibration", shuffle=False)
        observed = [
            (row.get("cabg_sample_id"), row.get("image_sha256"), row.get("scout_block_id"), row.get("scout_block_position"), row.get("scout_exposure_id"))
            for row in dataset.list_data_dict
        ]
        expected = [
            (row["sample_id"], row["sha256"], row["scout_block_id"], row["scout_block_position"], row["scout_exposure_id"])
            for row in manifest
        ]
        if observed != expected:
            raise CABGContractError("Scout dataset order/identity changed.")
        previous = "0" * 64
        resume_audit: dict[str, object] = {"phase": args.phase, "arm": args.arm, "resumed": False}
        if args.phase == "phase2":
            if args.arm in CAUSAL_ARMS:
                if args.parent_audit is None:
                    raise CABGContractError("Causal continuation requires the serialized parent audit.")
                parent_audit = _load(args.parent_audit)
                validate_parent_audit(parent_audit)
                payload, checkpoint_manifest = load_locked_parent(args.resume_checkpoint)
            else:
                parent_audit = None
                payload, checkpoint_manifest = load_strict(args.resume_checkpoint, run_provenance)
            if payload["sampler_state"]["cursor"] != PHASE1_BLOCKS:
                raise CABGContractError("Scout resume cursor is not block 12.")
            for name in TRAINABLE_NAMES:
                masters[name].data.copy_(payload["master_state"][name].to(wrapper.device))
                named[name].data.copy_(payload["adapter_state"][name].to(wrapper.device))
            optimizer.load_state_dict(payload["optimizer_state"])
            _move_optimizer_state(optimizer, wrapper.device)
            scheduler.load_state_dict(payload["scheduler_state"])
            controller.load_state_dict(payload["controller_state"])
            if args.arm in SUPPORTED_REPAIR_ARMS:
                repair_state = validate_repair_state(args.arm, payload["sampler_state"].get("repair_state"))
            elif "repair_state" in payload["sampler_state"]:
                raise CABGContractError("Non-repair Scout checkpoint unexpectedly contains repair state.")
            restore_rng_state(payload["rng_state"])
            previous = payload["trace_state"]["last_record_sha256"]
            pre_intervention_optimizer_inventory = canonical_json_sha256(tensor_inventory({"optimizer": _cpu(optimizer.state_dict())}))
            causal_intervention = None
            if args.arm in CAUSAL_ARMS:
                causal_intervention = apply_optimizer_intervention(args.arm, optimizer, masters)
            resume_audit = {
                "phase": args.phase,
                "arm": args.arm,
                "resumed": True,
                "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
                "cursor": payload["sampler_state"]["cursor"],
                "controller_state": controller.state_dict(),
                "rng_fingerprint": rng_state_fingerprint(snapshot_rng_state()),
                "checkpoint_rng_fingerprint": checkpoint_manifest["rng_fingerprint"],
                "adapter_exact": all(torch.equal(named[name].detach().cpu(), payload["adapter_state"][name]) for name in TRAINABLE_NAMES),
                "master_exact": all(torch.equal(masters[name].detach().cpu(), payload["master_state"][name]) for name in TRAINABLE_NAMES),
                "optimizer_tensor_inventory_sha256": pre_intervention_optimizer_inventory,
                "checkpoint_optimizer_tensor_inventory_sha256": canonical_json_sha256(tensor_inventory({"optimizer": payload["optimizer_state"]})),
                "repair_state": repair_state,
                "scheduler_exact": scheduler.state_dict() == payload["scheduler_state"],
                "controller_exact": controller.state_dict() == payload["controller_state"],
            }
            if args.arm in CAUSAL_ARMS:
                resume_audit.update({
                    "shared_parent": parent_audit,
                    "shared_parent_trace_anchor": payload["trace_state"]["last_record_sha256"],
                    "shared_parent_provenance": payload["provenance"],
                    "pre_intervention_optimizer_inventory_sha256": pre_intervention_optimizer_inventory,
                    "causal_optimizer_intervention": causal_intervention,
                    "post_intervention_rng_fingerprint": rng_state_fingerprint(snapshot_rng_state()),
                })
            if not resume_audit["adapter_exact"] or not resume_audit["master_exact"] or resume_audit["rng_fingerprint"] != resume_audit["checkpoint_rng_fingerprint"] or resume_audit["optimizer_tensor_inventory_sha256"] != resume_audit["checkpoint_optimizer_tensor_inventory_sha256"]:
                raise CABGContractError("Scout fresh-process resume was not exact.")
            if args.arm in CAUSAL_ARMS and resume_audit["post_intervention_rng_fingerprint"] != resume_audit["checkpoint_rng_fingerprint"]:
                raise CABGContractError("Causal optimizer intervention changed RNG state.")
            if args.arm in FIXED_ARMS and not (resume_audit["scheduler_exact"] and resume_audit["controller_exact"]):
                raise CABGContractError("Fixed-cutoff scheduler/controller restore mismatch.")
            if args.arm not in CAUSAL_ARMS and args.parent_audit is not None:
                raise CABGContractError("Non-causal Scout phase cannot accept a parent audit.")
        elif args.trace.exists():
            raise FileExistsError(args.trace)
        _exclusive_json(phase_dir / "resume-audit.json", resume_audit)
        file_audit.install()
        initial_state_sha256 = canonical_json_sha256(runtime_audit["step0_state"])
        for block_id in range(phase_start, phase_end):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            block_rows = manifest[block_id * 3:(block_id + 1) * 3]
            labels = [row["scout_label"] for row in block_rows]
            if Counter(labels) != Counter({"normal": 1, "abnormal": 2}):
                raise CABGContractError("Scout logical block composition changed.")
            markers.mark("block_started", block_id=block_id)
            rng_before = rng_state_fingerprint(snapshot_rng_state())
            if args.arm in FIXED_ARMS and block_id == 0 and rng_before != INITIAL_RNG_SHA256:
                raise CABGContractError("Fixed-cutoff original training RNG mismatch.")
            lm_images: list[dict[str, torch.Tensor]] = []
            lce_images: list[dict[str, torch.Tensor]] = []
            image_records: list[dict[str, object]] = []
            for local_index, row in enumerate(block_rows):
                dataset_index = block_id * 3 + local_index
                batch = move_batch_to_device(collator([dataset[dataset_index]]), wrapper.device)
                anomaly_labels = batch.pop("anomaly_labels")
                if int(anomaly_labels.item()) != (1 if row["scout_label"] == "abnormal" else 0):
                    raise CABGContractError("Scout label/image alignment changed.")
                batch["tune_mode"] = "default"
                batch["return_anomaly_evidence"] = True
                forward_started = time.perf_counter()
                with RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
                    result = model(**batch)
                observed_maps = observer.finalize(result.anomaly_evidence)
                lce = compute_lce(observed_maps["abnormal_raw"], observed_maps["normal_raw"], [row["scout_label"]])
                lm_raw = torch.autograd.grad(result.loss, trainable, retain_graph=True, allow_unused=True)
                lce_raw = torch.autograd.grad(lce["per_image_loss"][0], trainable, retain_graph=False, allow_unused=True)
                lm_named: dict[str, torch.Tensor] = {}
                for name, value in zip(TRAINABLE_NAMES, lm_raw):
                    if value is None or not torch.isfinite(value).all():
                        raise CABGContractError(f"Scout LM gradient invalid: {name}")
                    lm_named[name] = value.detach().float().cpu()
                lce_full = dict(zip(TRAINABLE_NAMES, lce_raw))
                lce_named = exact_s9_from_full(lce_full, S9_NAMES)
                lm_images.append(lm_named)
                lce_images.append(lce_named)
                image_records.append({
                    "sample_id": row["sample_id"],
                    "exposure_id": row["scout_exposure_id"],
                    "sha256": row["sha256"],
                    "label": row["scout_label"],
                    "lm_loss": float(result.loss.detach()),
                    "lce_loss": float(lce["per_image_loss"][0].detach()),
                    "lce_score": float(lce["score"][0].detach()),
                    "effective_support": float(lce["effective_support"][0].detach()),
                    "top11_mass": float(lce["top11_mass"][0].detach()),
                    "lm_s9_norm": math.sqrt(sum(float(lm_named[name].double().square().sum()) for name in S9_NAMES)),
                    "lce_s9_norm": math.sqrt(sum(float(lce_named[name].double().square().sum()) for name in S9_NAMES)),
                    "exact_s9": True,
                    "structural_zero_outside_s9": True,
                    "structural_zero_name": STRUCTURAL_ZERO_NAME,
                    "forward_seconds": time.perf_counter() - forward_started,
                })
                del batch, anomaly_labels, result, lce, observed_maps, lm_raw, lce_raw, lce_full
                gc.collect()
                torch.cuda.synchronize()
            repair_record = None
            causal_record = None
            fixed_record = None
            if args.arm != "lm_only":
                applied, details = compute_block_gradients(
                    lm_images, lce_images, labels, trainable_names=TRAINABLE_NAMES, s9_names=S9_NAMES, controller=controller
                )
                if args.arm in FIXED_ARMS:
                    applied, details, fixed_record = apply_fixed_cutoff(args.arm, block_id, applied, details)
                elif args.arm in CAUSAL_ARMS:
                    applied, details, multiplier = apply_causal_gradient(args.arm, details)
                    causal_record = {
                        "branch": args.arm,
                        "shared_parent_checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
                        "shared_parent_trace_anchor": payload["trace_state"]["last_record_sha256"],
                        "applied_lce_multiplier": multiplier,
                        "controller_diagnostics_computed": True,
                        "heldout_input_used": False,
                        "optimizer_intervention_applied_before_continuation": bool(causal_intervention["applied"]),
                    }
                elif args.arm in SUPPORTED_REPAIR_ARMS:
                    repair_record, repair_state = repair_decision(args.arm, block_id, image_records, repair_state)
                    applied, details = apply_repair_multiplier(details, float(repair_record["multiplier"]))
                    moment_reset = {"applied": False}
                    if (
                        args.arm == MOMENT_RESET_ARM
                        and repair_record["state_before"].get("latched") is False
                        and repair_record["state_after"].get("latched") is True
                    ):
                        moment_reset = reset_s9_directional_moments(optimizer, masters)
                    repair_record["optimizer_moment_reset"] = moment_reset
            else:
                applied, details = compute_lm_only_block(
                    lm_images, lce_images, labels, trainable_names=TRAINABLE_NAMES, s9_names=S9_NAMES
                )
            gradient = details["gradient"]
            budget = details["budget"]
            trust_ratio = float(gradient["scaled_auxiliary_shared_norm"]) / (float(gradient["lm_shared_norm"]) + 1e-12)
            if trust_ratio > RHO_MAX + 1e-5:
                raise CABGContractError("Scout trust cap failed.")
            before = {name: masters[name].detach().clone() for name in TRAINABLE_NAMES}
            optimizer.zero_grad(set_to_none=True)
            for name in TRAINABLE_NAMES:
                masters[name].grad = applied[name].to(wrapper.device)
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            changed = 0
            total_l2_squared = 0.0
            max_abs_change = 0.0
            for name in TRAINABLE_NAMES:
                if not torch.isfinite(masters[name]).all():
                    raise CABGContractError(f"Scout master parameter became non-finite: {name}")
                difference = masters[name].detach() - before[name]
                changed += int(bool(torch.any(difference != 0)))
                total_l2_squared += float(difference.double().square().sum())
                max_abs_change = max(max_abs_change, float(difference.abs().max()))
                named[name].data.copy_(masters[name].data.to(torch.bfloat16))
            if changed == 0 or any(parameter.grad is not None for parameter in trainable):
                raise CABGContractError("Scout parameter update or model .grad isolation failed.")
            applied_hashes = {name: tensor_sha256(applied[name]) for name in TRAINABLE_NAMES}
            rng_after = rng_state_fingerprint(snapshot_rng_state())
            cap_lambda = float(budget.get("lambda_controller", budget["lambda_final"]))
            cap_active = bool(cap_lambda + 1e-12 < float(budget["lambda_raw"])) if args.arm != "lm_only" else False
            run_id = CAUSAL_RUN_ID if args.arm in CAUSAL_ARMS else ADJUSTMENT_RUN_ID if args.arm == MOMENT_RESET_ARM else REPAIR_RUN_ID if args.arm in SUPPORTED_REPAIR_ARMS else RUN_ID
            schema_version = CAUSAL_SCHEMA_VERSION if args.arm in CAUSAL_ARMS else ADJUSTMENT_SCHEMA_VERSION if args.arm == MOMENT_RESET_ARM else REPAIR_SCHEMA_VERSION if args.arm in SUPPORTED_REPAIR_ARMS else SCHEMA_VERSION
            claim_scope = "development_only_shared_checkpoint_causal_mechanism" if args.arm in CAUSAL_ARMS else "development_only_stability_repair" if args.arm in SUPPORTED_REPAIR_ARMS else "development_only_exploratory_effect_scout"
            if args.arm in FIXED_ARMS:
                run_id, schema_version, claim_scope = FIXED_RUN_ID, FIXED_SCHEMA, FIXED_SCOPE
            record = {
                "schema_version": schema_version,
                "status": "SUCCESS",
                "run_id": run_id,
                "claim_scope": claim_scope,
                "arm": args.arm,
                "phase": args.phase,
                "block_id": block_id,
                "ordered_samples": [{"sample_id": row["sample_id"], "exposure_id": row["scout_exposure_id"], "sha256": row["sha256"], "label": row["scout_label"]} for row in block_rows],
                "images": image_records,
                "controller": {**budget, "cap_active": cap_active, "trust_ratio": trust_ratio},
                "class_rms": {"lm": details["lm_class_rms"], "lce": details["lce_class_rms"], "lm_balanced": details["lm_rms"], "lce_balanced": details["lce_rms"]},
                "gradient": gradient,
                "optimizer": {"type": "AdamW", "step": block_id + 1, "learning_rate_used": learning_rate_used, "next_learning_rate": optimizer.param_groups[0]["lr"], "betas": list(ADAM_BETAS), "epsilon": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY},
                "scheduler": {"type": "cosine", "step": scheduler.last_epoch, "horizon": SCOUT_BLOCKS, "zero_warmup": True},
                "parameter_update": {"changed_tensor_count": changed, "total_l2_change": math.sqrt(total_l2_squared), "max_abs_change": max_abs_change},
                "applied_gradient": {"tensor_count": len(applied), "all_finite": all(torch.isfinite(value).all() for value in applied.values()), "inventory_sha256": canonical_json_sha256(applied_hashes)},
                "memory": {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2},
                "runtime": {"block_seconds": time.perf_counter() - started, "images_per_second": 3.0 / (time.perf_counter() - started)},
                "watch": {"minimum_effective_support": min(float(row["effective_support"]) for row in image_records), "maximum_top11_mass": max(float(row["top11_mass"]) for row in image_records), "support_below_128": any(float(row["effective_support"]) < 128.0 for row in image_records), "top11_above_0_35": any(float(row["top11_mass"]) > 0.35 for row in image_records)},
                "rng": {"before_block": rng_before, "after_block": rng_after},
                "initial_state_sha256": initial_state_sha256,
                "provenance": run_provenance,
            }
            if repair_record is not None:
                record["repair"] = repair_record
            if causal_record is not None:
                record["causal"] = causal_record
            if fixed_record is not None:
                record["fixed_cutoff"] = fixed_record
            previous = _append(args.trace, record, previous)
            markers.mark("block_completed", block_id=block_id, record_sha256=previous)
            del before, applied, details, lm_images, lce_images
            gc.collect()
            torch.cuda.empty_cache()
        checkpoint_payload = _checkpoint_payload(
            model, masters, optimizer, scheduler, controller, cursor=phase_end, split=split,
            run_provenance=run_provenance, last_hash=previous, repair_state=repair_state,
        )
        checkpoint_manifest = save_atomic(args.checkpoint_output, checkpoint_payload)
        file_result = file_audit.result()
        file_result.update({"phase": args.phase, "arm": args.arm, "protected_internal_test_outputs_read": 0})
        if file_result["status"] != "SUCCESS":
            raise CABGContractError("Scout training file-open allowlist failed.")
        _exclusive_json(phase_dir / "file-open-audit.json", file_result)
        _exclusive_json(phase_dir / "state-final.json", trainable_state_record(model))
        _exclusive_json(phase_dir / "checkpoint-save.json", checkpoint_manifest)
        markers.mark("phase_completed", phase_name=args.phase, arm=args.arm, blocks=phase_end - phase_start)
        monitor_result = monitor.stop()
        monitor_active = False
        _exclusive_json(phase_dir / "monitor-summary.json", monitor_result)
        if monitor_result["peak_memory_used_mib"] > EXTERNAL_PEAK_LIMIT_MIB:
            raise CABGContractError("Scout external VRAM limit exceeded.")
        result = {
            "status": "SUCCESS",
            "decision": f"D4_SCOUT_{args.arm.upper()}_{args.phase.upper()}_COMPLETE",
            "arm": args.arm,
            "phase": args.phase,
            "blocks_completed": phase_end - phase_start,
            "cursor": phase_end,
            "checkpoint": str(args.checkpoint_output.resolve()),
            "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
            "last_record_sha256": previous,
            "controller_state": controller.state_dict(),
            "repair_state": repair_state,
            "external_peak_mib": monitor_result["peak_memory_used_mib"],
        }
        _exclusive_json(phase_dir / "result.json", result)
        return result
    finally:
        if monitor_active:
            try:
                monitor_result = monitor.stop()
                if not (phase_dir / "monitor-summary.json").exists():
                    _exclusive_json(phase_dir / "monitor-summary.json", monitor_result)
            except Exception:
                pass
        try:
            markers.mark("process_stopping", phase_name=args.phase, arm=args.arm)
            markers.close()
        except Exception:
            pass
        if wrapper is not None:
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("cabg_lce", "lm_only", *SUPPORTED_REPAIR_ARMS, *CAUSAL_ARMS, *FIXED_ARMS), required=True)
    parser.add_argument("--phase", choices=("phase1", "phase2"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--parent-audit", type=Path)
    args = parser.parse_args()
    if args.phase == "phase2" and args.resume_checkpoint is None:
        parser.error("phase2 requires --resume-checkpoint")
    if args.arm in CAUSAL_ARMS and args.phase != "phase2":
        parser.error("causal continuation arms can run only phase2 blocks 13-24")
    if args.arm in CAUSAL_ARMS and args.parent_audit is None:
        parser.error("causal continuation arms require --parent-audit")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
