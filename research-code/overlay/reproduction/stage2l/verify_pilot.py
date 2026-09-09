#!/usr/bin/env python3
"""Independent D4 evaluator; recomputes controller and applied gradients from artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, tensor_sha256
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2l.constants import (
    EMA_BETA, EPSILON, GLOBAL_CLIP_NORM, LEARNING_RATE, LM_REFERENCE_FLOOR,
    PILOT_BLOCKS, RHO, RHO_MAX, S9_NAMES, TRAINABLE_NAMES,
)


class VerificationError(RuntimeError):
    pass


def _norm(values: dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float32)
    for name in sorted(values):
        value = values[name].detach().float().cpu()
        if not torch.isfinite(value).all():
            raise VerificationError(f"Non-finite gradient tensor: {name}")
        total += value.square().sum()
    return float(total.sqrt())


def _mean(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack([row[name].float() for row in rows]).mean(0) for name in sorted(rows[0])}


def _class_mean(rows: list[dict[str, torch.Tensor]], labels: list[str]) -> dict[str, torch.Tensor]:
    result = {}
    for name in sorted(rows[0]):
        normal = torch.stack([row[name].float() for row, label in zip(rows, labels) if label == "normal"]).mean(0)
        abnormal = torch.stack([row[name].float() for row, label in zip(rows, labels) if label == "abnormal"]).mean(0)
        result[name] = 0.5 * (normal + abnormal)
    return result


def _rms(rows: list[dict[str, torch.Tensor]], labels: list[str]) -> tuple[float, dict[str, float]]:
    squared = []
    for row in rows:
        total = torch.zeros((), dtype=torch.float32)
        for name in sorted(row):
            total += row[name].float().square().sum().cpu()
        squared.append(total)
    stacked = torch.stack(squared)
    classes = {
        label: stacked[torch.tensor([member == label for member in labels], dtype=torch.bool)].mean()
        for label in ("normal", "abnormal")
    }
    if min(float(value) for value in classes.values()) <= 0 or not all(math.isfinite(float(value)) for value in classes.values()):
        raise VerificationError("Class RMS is zero or non-finite.")
    balanced = (0.5 * (classes["normal"] + classes["abnormal"]) + EPSILON**2).sqrt()
    return float(balanced), {
        key: float(value.sqrt()) for key, value in classes.items()
    }


def _close(observed: float, expected: float, context: str, atol: float = 1e-6, rtol: float = 2e-6) -> None:
    if not math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol):
        raise VerificationError(f"{context}: observed={observed}, expected={expected}")


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["checkpoint_sha256"] != sha256_file(path) or manifest["checkpoint_bytes"] != path.stat().st_size:
        raise VerificationError("Checkpoint byte identity failed.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if rng_state_fingerprint(payload["rng_state"]) != manifest["rng_fingerprint"]:
        raise VerificationError("Checkpoint RNG fingerprint failed.")
    return payload, manifest


def verify(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    dataset = json.loads(args.dataset_audit.read_text())
    selected = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    training = [json.loads(line) for line in args.training_development.read_text().splitlines() if line.strip()]
    d3r = [json.loads(line) for line in args.d3r_manifest.read_text().splitlines() if line.strip()]
    if dataset.get("manifest_sha256") != sha256_file(args.manifest) or dataset.get("d3r_exclusion_sha256") != sha256_file(args.d3r_manifest):
        raise VerificationError("D4 dataset audit byte identities failed.")
    d3r_ids = {str(row["sample_id"]) for row in d3r}
    d3r_hashes = {str(row["sha256"]) for row in d3r}
    training_ids = {str(row["sample_id"]) for row in training}
    if len(training) != 132 or len(selected) != 12:
        raise VerificationError("D4 source/selected manifest counts failed.")
    if any(str(row["sample_id"]) not in training_ids or str(row["sample_id"]) in d3r_ids or str(row["sha256"]) in d3r_hashes for row in selected):
        raise VerificationError("D4 independent training-boundary subtraction failed.")
    if len(records) != PILOT_BLOCKS or [row["block_id"] for row in records] != list(range(PILOT_BLOCKS)):
        raise VerificationError("D4 trace does not contain the exact four registered blocks.")
    previous = "0" * 64
    lm_ema = lce_ema = 0.0
    blocks = []
    all_samples = []
    for index, record in enumerate(records):
        supplied = record["record_sha256"]
        unhashed = dict(record)
        unhashed.pop("record_sha256")
        if record["previous_record_sha256"] != previous or canonical_json_sha256(unhashed) != supplied:
            raise VerificationError("D4 trace hash chain failed.")
        previous = supplied
        expected_provenance = {
            "source_commit": json.loads(args.preflight.read_text())["repository"]["commit"],
            "implementation_fingerprint": json.loads(args.preflight.read_text())["source"]["fingerprint"],
        }
        if any(record["provenance"][key] != value for key, value in expected_provenance.items()):
            raise VerificationError("D4 record source provenance failed.")
        labels = [sample["label"] for sample in record["ordered_samples"]]
        if Counter(labels) != Counter({"normal": 1, "abnormal": 2}):
            raise VerificationError("D4 block class composition failed.")
        all_samples.extend((sample["sample_id"], sample["sha256"]) for sample in record["ordered_samples"])
        audit_path = Path(record["gradient_audit"]["path"])
        if sha256_file(audit_path) != record["gradient_audit"]["sha256"]:
            raise VerificationError("D4 gradient audit file hash failed.")
        payload = torch.load(audit_path, map_location="cpu", weights_only=False)
        lm_images, lce_images = payload["per_image_lm_s9"], payload["per_image_lce_s9"]
        lm_rms, lm_class = _rms(lm_images, labels)
        lce_rms, lce_class = _rms(lce_images, labels)
        aggregate_lm_s9 = _mean(lm_images)
        aggregate_lce = _class_mean(lce_images, labels)
        for name in S9_NAMES:
            torch.testing.assert_close(payload["aggregate_lm"][name], aggregate_lm_s9[name], rtol=0, atol=0)
            torch.testing.assert_close(payload["aggregate_lce"][name], aggregate_lce[name], rtol=0, atol=0)
        lm_norm, lce_norm = _norm(aggregate_lm_s9), _norm(aggregate_lce)
        if lm_norm <= LM_REFERENCE_FLOOR:
            raise VerificationError("D4 aggregate LM S9 reference failed.")
        lm_ema = EMA_BETA * lm_ema + (1 - EMA_BETA) * lm_rms
        lce_ema = EMA_BETA * lce_ema + (1 - EMA_BETA) * lce_rms
        correction = 1 - EMA_BETA ** (index + 1)
        corrected_lm, corrected_lce = lm_ema / correction, lce_ema / correction
        lambda_raw = RHO * corrected_lm / (corrected_lce + EPSILON)
        lambda_cap = RHO_MAX * lm_norm / (lce_norm + EPSILON)
        lambda_final = min(lambda_raw, lambda_cap)
        budget = record["controller"]
        for key, expected in {
            "lm_ema": lm_ema, "lce_ema": lce_ema, "corrected_lm_rms": corrected_lm,
            "corrected_lce_rms": corrected_lce, "lm_s9_norm": lm_norm, "lce_s9_norm": lce_norm,
            "lambda_raw": lambda_raw, "lambda_cap": lambda_cap, "lambda_final": lambda_final,
        }.items():
            _close(float(budget[key]), expected, f"block {index} controller {key}")
        for label in ("normal", "abnormal"):
            _close(float(record["class_rms"]["lm"][label]), lm_class[label], f"block {index} LM class RMS")
            _close(float(record["class_rms"]["lce"][label]), lce_class[label], f"block {index} LCE class RMS")
        aggregate_lm = {name: payload["aggregate_lm"][name].float() for name in TRAINABLE_NAMES}
        combined = {name: value.clone() for name, value in aggregate_lm.items()}
        for name in S9_NAMES:
            combined[name].add_(aggregate_lce[name].float() * lambda_final)
        scaled_norm = _norm({name: aggregate_lce[name].float() * lambda_final for name in S9_NAMES})
        tolerance = max(1e-8, 1e-5 * lm_norm)
        if scaled_norm > RHO_MAX * lm_norm + tolerance:
            raise VerificationError("D4 independently recomputed trust cap failed.")
        preclip = _norm(combined)
        clip_scale = min(1.0, GLOBAL_CLIP_NORM / (preclip + 1e-12))
        applied = {name: value * clip_scale for name, value in combined.items()}
        for name in TRAINABLE_NAMES:
            if tensor_sha256(applied[name]) != record["applied_gradient"]["tensor_sha256"][name]:
                raise VerificationError(f"D4 applied gradient hash mismatch: block={index}, tensor={name}")
        _close(record["gradient"]["scaled_auxiliary_shared_norm"], scaled_norm, "scaled auxiliary norm")
        _close(record["gradient"]["preclip_full_norm"], preclip, "preclip norm")
        _close(record["gradient"]["clip_scale"], clip_scale, "clip scale")
        if record["optimizer"]["step"] != index + 1 or record["scheduler"]["step"] != index + 1:
            raise VerificationError("D4 optimizer/scheduler step count failed.")
        if record["parameter_update"]["changed_tensor_count"] <= 0 or not record["applied_gradient"]["all_finite"]:
            raise VerificationError("D4 optimizer update or applied-gradient finiteness failed.")
        if any(not image["lce_support"]["exact_s9"] or not image["lce_support"]["structural_zero_outside_s9"] for image in record["images"]):
            raise VerificationError("D4 exact S9/structural-zero support audit failed.")
        blocks.append({
            "block_id": index, "lm_rms": lm_rms, "lce_rms": lce_rms,
            "lambda_raw": lambda_raw, "lambda_cap": lambda_cap, "lambda_final": lambda_final,
            "cap_active": lambda_cap <= lambda_raw, "trust_ratio": scaled_norm / lm_norm,
            "preclip_norm": preclip, "clip_scale": clip_scale,
            "changed_tensors": record["parameter_update"]["changed_tensor_count"],
            "effective_support": [image["effective_support"] for image in record["images"]],
            "block_seconds": record["runtime"]["block_seconds"],
        })
    if len(set(all_samples)) != 12:
        raise VerificationError("D4 pilot reused a sample or content identity.")
    phase1_payload, phase1_manifest = _load_checkpoint(args.phase1_checkpoint)
    final_payload, final_manifest = _load_checkpoint(args.final_checkpoint)
    resume = json.loads((args.run_root / "phase2" / "resume-audit.json").read_text())
    if not (resume["resumed"] and resume["adapter_exact"] and resume["master_exact"]):
        raise VerificationError("D4 phase2 exact resume audit failed.")
    if resume["checkpoint_sha256"] != phase1_manifest["checkpoint_sha256"]:
        raise VerificationError("D4 phase2 resumed the wrong checkpoint.")
    if phase1_payload["trace_state"]["completed_blocks"] != 2 or final_payload["trace_state"]["completed_blocks"] != 4:
        raise VerificationError("D4 checkpoint cursors failed.")
    expected_controller = {"beta": EMA_BETA, "lm_ema": lm_ema, "lce_ema": lce_ema, "valid_blocks": 4}
    for key, value in expected_controller.items():
        if isinstance(value, float):
            _close(float(final_payload["controller_state"][key]), value, f"final controller {key}")
        elif final_payload["controller_state"][key] != value:
            raise VerificationError(f"Final controller {key} failed.")
    file_audits = [json.loads((args.run_root / phase / "file-open-audit.json").read_text()) for phase in ("phase1", "phase2")]
    if any(row["status"] != "SUCCESS" or row["non_allowlisted_paths"] or row["protected_internal_test_outputs_read"] != 0 for row in file_audits):
        raise VerificationError("D4 file isolation audit failed.")
    monitors = [json.loads((args.run_root / phase / "monitor-summary.json").read_text()) for phase in ("phase1", "phase2")]
    peak = max(row["peak_memory_used_mib"] for row in monitors)
    total_seconds = sum(block["block_seconds"] for block in blocks)
    source = json.loads(args.preflight.read_text())
    result = {
        "status": "SUCCESS", "decision": "PASS_D4_PILOT",
        "claim_scope": "development_only_training_feasibility_only",
        "blocks": 4, "images": 12, "optimizer_steps": 4, "fresh_process_resume": True,
        "all_s9_supported_every_image": True, "all_lm_lce_gradients_finite": True,
        "all_trust_caps_pass": True, "all_applied_gradient_hashes_pass": True,
        "parameter_updates_valid": True, "checkpoint_rng_state_exact": True,
        "file_isolation_pass": True, "d3r_images_used": 0,
        "protected_internal_test_image_files_opened": 0, "protected_internal_test_outputs_read": 0,
        "external_peak_mib": peak, "external_peak_limit_mib": 22500,
        "resource_gate_pass": peak <= 22500, "block_seconds_total": total_seconds,
        "images_per_second": 12.0 / total_seconds,
        "blocks_detail": blocks,
        "source_commit": source["repository"]["commit"],
        "implementation_fingerprint": source["source"]["fingerprint"],
        "phase1_checkpoint_sha256": phase1_manifest["checkpoint_sha256"],
        "final_checkpoint_sha256": final_manifest["checkpoint_sha256"],
        "last_record_sha256": previous,
        "scientific_claims_not_authorized": [
            "effectiveness", "performance_improvement", "superiority", "generalization",
            "patient_independence", "clinical_validity", "protected_internal_test_performance",
        ],
    }
    if not result["resource_gate_pass"]:
        result["status"] = "FAILED"
        result["decision"] = "BLOCKED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase1-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-development", type=Path, required=True)
    parser.add_argument("--d3r-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()
