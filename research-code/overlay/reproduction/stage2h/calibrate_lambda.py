#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from reproduction.stage2h.evidence import (
    LAMBDA_CANDIDATES,
    LambdaCalibrationAccumulator,
    calibration_parameter_items,
    compute_evidence_margin_loss,
)
from reproduction.stage2h.runtime import (
    load_stage2h_runtime,
    make_stage2h_dataset,
    move_batch_to_device,
)


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def rng_restore(snapshot):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if snapshot["torch_cuda"]:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def rng_fingerprint(snapshot):
    digest = hashlib.sha256()
    digest.update(repr(snapshot["python"]).encode("utf-8"))
    digest.update(repr(snapshot["numpy"][:2]).encode("utf-8"))
    digest.update(snapshot["numpy"][1].tobytes())
    digest.update(tensor_bytes(snapshot["torch_cpu"]))
    for state in snapshot["torch_cuda"]:
        digest.update(tensor_bytes(state))
    return digest.hexdigest()


def gradient_record(name, gradient, *, requires_grad):
    if gradient is None:
        return {
            "name": name,
            "requires_grad": bool(requires_grad),
            "gradient_present": False,
            "finite": True,
            "max_abs": 0.0,
            "l2_norm": 0.0,
        }
    detached = gradient.detach().float()
    return {
        "name": name,
        "requires_grad": bool(requires_grad),
        "gradient_present": True,
        "finite": bool(torch.isfinite(detached).all().item()),
        "max_abs": float(detached.abs().max().item()),
        "l2_norm": float(detached.norm().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=4)
    parser.add_argument("--candidates", type=float, nargs="+", default=list(LAMBDA_CANDIDATES))
    parser.add_argument("--protocol-version", default="2.1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.reset_peak_memory_stats()
    wrapper, runtime_audit = load_stage2h_runtime(
        args.base_model,
        seed=42,
        training=True,
        gradient_checkpointing=True,
    )
    model = wrapper.llm
    _, dataset, collator = make_stage2h_dataset(
        args.base_model,
        "medic_ad_stage2h_calibration",
        shuffle=False,
    )
    if len(dataset) != args.expected_images:
        raise RuntimeError(
            f"Stage 2H calibration requires exactly {args.expected_images} images, got {len(dataset)}."
        )
    accumulator = LambdaCalibrationAccumulator(model)
    calibration_rng = rng_snapshot()
    rng_before = rng_fingerprint(calibration_rng)
    observations = []
    isolation_records = None
    target_items = calibration_parameter_items(model)
    target_names = {name for name, _ in target_items}
    anomaly = model.visual.anomaly_qformer
    qformer_items = [
        (f"model.visual.anomaly_qformer.{name}", parameter)
        for name, parameter in anomaly.named_parameters()
    ]
    try:
        for index in range(len(dataset)):
            batch = move_batch_to_device(collator([dataset[index]]), wrapper.device)
            labels = batch.pop("anomaly_labels")
            batch["tune_mode"] = "default"
            batch["return_anomaly_evidence"] = True
            outputs = model(**batch)
            lm_loss = outputs.loss
            evidence_result = compute_evidence_margin_loss(outputs.anomaly_evidence, labels)
            evidence_loss = evidence_result["loss"]
            if index == 0:
                active_qformer_items = [
                    (name, parameter)
                    for name, parameter in qformer_items
                    if parameter.requires_grad
                ]
                isolation_gradients = torch.autograd.grad(
                    evidence_loss,
                    [parameter for _, parameter in active_qformer_items],
                    retain_graph=True,
                    allow_unused=True,
                )
                active_gradients = {
                    name: gradient
                    for (name, _), gradient in zip(active_qformer_items, isolation_gradients)
                }
                isolation_records = [
                    gradient_record(
                        name,
                        active_gradients.get(name),
                        requires_grad=parameter.requires_grad,
                    )
                    for name, parameter in qformer_items
                ]
            accumulator.add(lm_loss=lm_loss, evidence_loss=evidence_loss)
            evidence = outputs.anomaly_evidence.detach().float()
            observations.append(
                {
                    "microbatch_index": index,
                    "anomaly_label": int(labels.item()),
                    "lm_loss": float(lm_loss.detach().float().cpu().item()),
                    "evidence_loss": float(evidence_loss.detach().float().cpu().item()),
                    "evidence_shape": list(evidence.shape),
                    "evidence_min": float(evidence.min().cpu().item()),
                    "evidence_max": float(evidence.max().cpu().item()),
                    "evidence_mean": float(evidence.mean().cpu().item()),
                    "saturation_count": int(evidence_result["saturation_count"].cpu().item()),
                    "saturation_total": int(evidence_result["saturation_total"].cpu().item()),
                }
            )
            del outputs, lm_loss, evidence_loss, evidence_result, evidence, batch, labels
            torch.cuda.empty_cache()
        calibration = accumulator.finalize(candidates=args.candidates)
    finally:
        rng_restore(calibration_rng)
    rng_after = rng_fingerprint(rng_snapshot())
    if rng_after != rng_before:
        raise RuntimeError("Stage 2H calibration RNG restoration failed.")
    if isolation_records is None:
        raise RuntimeError("Stage 2H auxiliary gradient isolation was not evaluated.")
    by_name = {record["name"]: record for record in isolation_records}
    target_isolation = [by_name[name] for name in sorted(target_names)]
    for record in target_isolation:
        if not record["gradient_present"] or not record["finite"] or record["max_abs"] <= 1e-12:
            raise RuntimeError(f"Stage 2H evidence target gradient failed: {record}")
    forbidden_prefixes = (
        "model.visual.anomaly_qformer.q_former.",
        "model.visual.anomaly_qformer.post_ffn.",
        "model.visual.anomaly_qformer.diff_q_former.",
        "model.visual.anomaly_qformer.diff_post_ffn.",
    )
    forbidden_names = {"model.visual.anomaly_qformer.gate_scale"}
    forbidden = [
        record
        for record in isolation_records
        if record["name"] in forbidden_names or record["name"].startswith(forbidden_prefixes)
    ]
    leaked = [record for record in forbidden if (not record["finite"] or record["max_abs"] != 0.0)]
    if leaked:
        raise RuntimeError(f"Stage 2H auxiliary gradient isolation failed: {leaked}")
    for row in calibration["parameter_gradients"]:
        if row["evidence_mean_gradient_norm_fp32"] is None or row["evidence_mean_gradient_norm_fp32"] <= 1e-12:
            raise RuntimeError(f"Stage 2H calibrated evidence gradient is ineffective: {row['name']}")

    saturation_count = sum(row["saturation_count"] for row in observations)
    saturation_total = sum(row["saturation_total"] for row in observations)
    calibration_succeeded = calibration["status"] == "SUCCESS"
    result = {
        "status": "SUCCESS" if calibration_succeeded else "PAUSE",
        "stage": "2H-E",
        "protocol_version": args.protocol_version,
        "candidate_set": sorted(float(value) for value in args.candidates),
        "calibration_role": "step0_training_subset_only",
        "runtime_audit": runtime_audit,
        "rng_fingerprint_before": rng_before,
        "rng_fingerprint_after_restore": rng_after,
        "rng_isolation_passed": True,
        "gate_scale": float(anomaly.gate_scale.detach().float().cpu().item()),
        "microbatch_observations": observations,
        "evidence_saturation": {
            "numerator": saturation_count,
            "denominator": saturation_total,
            "proportion": saturation_count / saturation_total,
        },
        "auxiliary_gradient_table": isolation_records,
        "target_gradient_gate_passed": True,
        "gate_and_downstream_isolation_passed": True,
        "calibration": calibration,
        "pause_reason": (
            None
            if calibration_succeeded
            else "No fixed lambda candidate produced a gradient ratio inside [0.05,0.20]."
        ),
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_cuda_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
