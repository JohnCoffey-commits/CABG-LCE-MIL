#!/usr/bin/env python3
"""Independent evaluator for CABG-MIL v1.2 R0 artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.constants import ABNORMAL_LABEL, NORMAL_LABEL, SHARED_SUPPORT_NAMES
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2j import SCHEMA_VERSION
from reproduction.stage2j.candidate_math import CANDIDATE_ID
from reproduction.stage2j.run_redesign_r0 import (
    EXPECTED_D3_EVIDENCE_FINGERPRINT,
    EXPECTED_D3_VERIFICATION_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXTERNAL_PEAK_LIMIT_MIB,
)
from reproduction.stage2j.statistics import (
    distribution_record,
    named_global_norm,
    pearson,
    spearman,
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CABGContractError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CABGContractError(f"Expected JSONL objects: {path}")
    return rows


def _exclusive_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _close(left: object, right: object, *, tolerance: float = 2e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _reconstruct_bf16_forward_branches(
    payload: dict[str, object], shape: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Reconstruct producer tensors without changing BF16 arithmetic order.

    The JSON payload contains FP32-exact serializations of BF16 values.  The
    producer formed ``raw_gap`` and ``evidence`` while those operands were
    still BF16.  Subtracting the serialized operands in FP32 is therefore not
    an equivalent reconstruction (for example, 512 - 83 rounds to 428 in
    BF16 but is 429 in FP32).
    """

    if payload.get("dtype") != "float32_serialized_from_bfloat16_forward":
        raise CABGContractError("R0 raw payload dtype contract changed.")
    keys = (
        "abnormal_raw_flat",
        "normal_raw_flat",
        "abnormal_probability_flat",
        "normal_probability_flat",
    )
    expected = math.prod(shape)
    serialized: dict[str, torch.Tensor] = {}
    for key in keys:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != expected:
            raise CABGContractError(f"R0 raw payload is malformed: {key}")
        tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        if not torch.isfinite(tensor).all():
            raise CABGContractError(f"R0 raw payload is non-finite: {key}")
        if not torch.equal(tensor.to(torch.bfloat16).float(), tensor):
            raise CABGContractError(f"R0 raw payload is not an exact BF16 serialization: {key}")
        serialized[key] = tensor

    abnormal_raw_bf16 = serialized["abnormal_raw_flat"].to(torch.bfloat16)
    normal_raw_bf16 = serialized["normal_raw_flat"].to(torch.bfloat16)
    abnormal_probability_bf16 = serialized["abnormal_probability_flat"].to(torch.bfloat16)
    normal_probability_bf16 = serialized["normal_probability_flat"].to(torch.bfloat16)
    return {
        "abnormal_raw": abnormal_raw_bf16.float(),
        "normal_raw": normal_raw_bf16.float(),
        "raw_gap": (abnormal_raw_bf16 - normal_raw_bf16).float(),
        "abnormal_probability": abnormal_probability_bf16.float(),
        "normal_probability": normal_probability_bf16.float(),
        "evidence": (abnormal_probability_bf16 - normal_probability_bf16).float(),
    }


def _post_run_gpu_record(
    gpu_path: Path, compute_apps_path: Path
) -> dict[str, object]:
    """Parse the post-run GPU snapshot and explicit compute-process audit.

    ``utilization.gpu`` is a lagging sample-window measurement and can remain
    nonzero briefly after the process has released all memory.  Idleness is
    therefore defined by the locked memory ceiling plus an empty
    ``--query-compute-apps`` result; utilization is retained as an observation.
    """

    parts = [part.strip() for part in gpu_path.read_text(encoding="utf-8").strip().split(",")]
    if len(parts) != 4:
        raise CABGContractError("R0 post-run GPU record is malformed.")
    compute_processes = [
        line.strip()
        for line in compute_apps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record: dict[str, object] = {
        "name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "utilization_percent_observed": int(parts[3]),
        "active_compute_process_count": len(compute_processes),
        "active_compute_processes": compute_processes,
        "idle_criterion": "memory_used_mib<=32_and_active_compute_process_count==0",
    }
    record["idle_after_run"] = (
        int(record["memory_used_mib"]) <= 32
        and int(record["active_compute_process_count"]) == 0
    )
    return record


def _rows_by_name(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result = {str(row["name"]): row for row in rows}
    if len(result) != len(rows):
        raise CABGContractError("R0 gradient table contains duplicate parameter names.")
    return result


def _class_image_summary(values: list[float], labels: list[str]) -> dict[str, object]:
    classes = {}
    for label in (NORMAL_LABEL, ABNORMAL_LABEL):
        members = [value for value, observed in zip(values, labels) if observed == label]
        mean = sum(members) / len(members)
        variance = sum((value - mean) ** 2 for value in members) / len(members)
        ordered = sorted(members)
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
        classes[label] = {
            "count": len(members),
            "mean": mean,
            "median": median,
            "std_population": math.sqrt(variance),
            "min": min(members),
            "max": max(members),
        }
    pooled_variance = 0.5 * (classes[NORMAL_LABEL]["std_population"] ** 2 + classes[ABNORMAL_LABEL]["std_population"] ** 2)
    difference = classes[ABNORMAL_LABEL]["mean"] - classes[NORMAL_LABEL]["mean"]
    return {
        "classes": classes,
        "abnormal_minus_normal_mean": difference,
        "standardized_mean_difference_descriptive": difference / math.sqrt(pooled_variance) if pooled_variance > 0 else None,
    }


def _correlation_record(x: list[float], y: list[float]) -> dict[str, object]:
    return {"n": len(x), "pearson": pearson(x, y), "spearman": spearman(x, y), "inferential_test_performed": False}


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() or args.aggregate_output.exists():
        raise FileExistsError("Refusing to overwrite R0 evaluator outputs.")
    records = _load_jsonl(args.records)
    model_load = _load_json(args.model_load_audit)
    state_before = _load_json(args.state_before)
    state_after = _load_json(args.state_after)
    file_open = _load_json(args.file_open_audit)
    rng = _load_json(args.rng_audit)
    monitor = _load_json(args.monitor_summary)
    source = _load_json(args.source_inventory)
    dataset = _load_json(args.dataset_audit)
    manifest = _load_jsonl(Path(str(dataset["locked_manifest"])))
    failures: list[str] = []

    if source.get("fingerprint") != model_load.get("source_fingerprint"):
        failures.append("source_fingerprint_mismatch")
    if model_load.get("source_commit") is None or model_load.get("parent_gate_d3_commit") is None:
        failures.append("source_commit_provenance_missing")

    if len(records) != 12 or len({row.get("forward_id") for row in records}) != 12:
        raise CABGContractError("R0 evaluator requires exactly 12 records and 12 unique forwards.")
    labels = [str(row["label"]) for row in records]
    if Counter(labels) != Counter({NORMAL_LABEL: 6, ABNORMAL_LABEL: 6}):
        raise CABGContractError("R0 record class balance changed.")
    if sha256_file(Path(str(dataset["locked_manifest"]))) != EXPECTED_MANIFEST_SHA256 or len(manifest) != 12:
        raise CABGContractError("R0 source manifest changed.")
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if [str(row["sample_id"]) for row in records] != manifest_ids:
        raise CABGContractError("R0 record order/identity differs from the locked manifest.")

    parameters = model_load["runtime_audit"]["trainable_scope"]["parameters"]
    trainable_names = [str(row["name"]) for row in parameters]
    if len(trainable_names) != 21 or len(set(trainable_names)) != 21:
        raise CABGContractError("R0 model-load audit does not contain 21 unique trainable tensors.")
    scope_rows = {
        name: {
            "name": name,
            "module": name.rsplit(".", 1)[0],
            "shape": next(row["shape"] for row in parameters if row["name"] == name),
            "elements": int(next(row["elements"] for row in parameters if row["name"] == name)),
            "requires_grad": True,
            "intended_optimizer_scope_v1_1": True,
            "legacy_declared_shared_support": name in SHARED_SUPPORT_NAMES,
            "lm_effective_images": {NORMAL_LABEL: 0, ABNORMAL_LABEL: 0},
            "auxiliary_effective_images": {
                mechanism: {NORMAL_LABEL: 0, ABNORMAL_LABEL: 0}
                for mechanism in ("M0", "M1", "M2", CANDIDATE_ID)
            },
        }
        for name in trainable_names
    }

    raw_by_class = {
        label: {name: [] for name in ("abnormal_raw", "normal_raw", "raw_gap", "abnormal_probability", "normal_probability", "evidence", "lce_softsign_contrast")}
        for label in (NORMAL_LABEL, ABNORMAL_LABEL)
    }
    patch_layer_mean_by_class = {
        label: {name: [] for name in ("abnormal_raw", "normal_raw", "raw_gap", "abnormal_probability", "normal_probability", "evidence", "lce_softsign_contrast")}
        for label in (NORMAL_LABEL, ABNORMAL_LABEL)
    }
    image_means = defaultdict(list)
    mechanism_values = {
        mechanism: defaultdict(list) for mechanism in ("M0", "M1", "M2", CANDIDATE_ID)
    }
    per_image = []

    for record in records:
        if record.get("schema_version") != SCHEMA_VERSION or record.get("status") != "SUCCESS":
            raise CABGContractError("R0 record schema/status mismatch.")
        if record.get("claim_scope") != "exploratory_read_only_mechanism_diagnostic_only":
            failures.append("record_claim_scope")
        provenance = record["provenance"]
        if provenance.get("source_d3_evidence_fingerprint") != EXPECTED_D3_EVIDENCE_FINGERPRINT or provenance.get("dataset_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            raise CABGContractError("R0 record provenance mismatch.")
        label = str(record["label"])
        lm_rows = record["lm_gradients_all_trainable"]
        lm = _rows_by_name(lm_rows)
        if set(lm) != set(trainable_names):
            raise CABGContractError("R0 LM gradient table differs from the trainable scope.")
        for name, row in lm.items():
            if row["finite"] is not True:
                failures.append(f"non_finite_lm_gradient:{record['sample_id']}:{name}")
            if row["effective"] is True:
                scope_rows[name]["lm_effective_images"][label] += 1

        shape = tuple(int(value) for value in record["branches"]["tensor_shape"])
        if len(shape) != 3 or shape[0] != 1:
            raise CABGContractError("R0 raw payload shape is invalid.")
        payload = record["branches"]["raw_payload"]
        branches = _reconstruct_bf16_forward_branches(payload, shape)
        abnormal_raw = branches["abnormal_raw"]
        normal_raw = branches["normal_raw"]
        abnormal_probability = branches["abnormal_probability"]
        normal_probability = branches["normal_probability"]
        if not torch.allclose(
            torch.sigmoid(abnormal_raw.to(torch.bfloat16)).float(),
            abnormal_probability,
            rtol=0.0,
            atol=2 ** -8,
        ):
            raise CABGContractError("R0 abnormal probability does not match BF16 sigmoid(raw score).")
        if not torch.allclose(
            torch.sigmoid(normal_raw.to(torch.bfloat16)).float(),
            normal_probability,
            rtol=0.0,
            atol=2 ** -8,
        ):
            raise CABGContractError("R0 normal probability does not match BF16 sigmoid(raw score).")
        kinds = {"abnormal_raw": "raw_logit", "normal_raw": "raw_logit", "raw_gap": "raw_gap", "abnormal_probability": "probability", "normal_probability": "probability", "evidence": "evidence"}
        for branch, tensor in branches.items():
            recomputed = distribution_record(tensor, kind=kinds[branch])
            stored = record["branches"]["all_layer_patch_values"][branch]
            for key in ("count", "min", "max", "mean", "median", "std_population"):
                if key == "count":
                    if int(recomputed[key]) != int(stored[key]):
                        raise CABGContractError(f"R0 stored {branch} count mismatch.")
                elif not _close(recomputed[key], stored[key]):
                    raise CABGContractError(f"R0 stored {branch} statistic mismatch: {key}")
            raw_by_class[label][branch].append(tensor.reshape(-1))
            patch_layer_mean_by_class[label][branch].append(tensor.mean(dim=1).reshape(-1))
            image_means[branch].append(float(tensor.mean()))
        # LCE is intentionally defined on an FP32 working gap.  This is
        # distinct from the BF16 producer ``raw_gap`` branch reconstructed
        # above and must not be used to validate that stored branch.
        lce_working_gap = abnormal_raw.float() - normal_raw.float()
        lce_contrast = lce_working_gap / (1.0 + lce_working_gap.abs())
        raw_by_class[label]["lce_softsign_contrast"].append(lce_contrast.reshape(-1))
        patch_layer_mean_by_class[label]["lce_softsign_contrast"].append(
            lce_contrast.mean(dim=1).reshape(-1)
        )
        image_means["lce_softsign_contrast"].append(float(lce_contrast.mean()))

        mechanisms = record["mechanisms"]
        if set(mechanisms) != {"M0", "M1", "M2", CANDIDATE_ID}:
            raise CABGContractError("R0 record mechanism set changed.")
        m2_intersection = mechanisms["M2"]["support"]["actual_intersection"]
        lm_intersection_norm = named_global_norm(lm_rows, m2_intersection)
        m2_aux_norm = float(mechanisms["M2"]["auxiliary_global_norm_actual_intersection"])
        ratio = m2_aux_norm / lm_intersection_norm if lm_intersection_norm > 0 else None
        for mechanism, value in mechanisms.items():
            auxiliary = _rows_by_name(value["auxiliary_gradients_all_trainable"])
            if set(auxiliary) != set(trainable_names):
                raise CABGContractError("R0 auxiliary gradient table differs from trainable scope.")
            for name, row in auxiliary.items():
                if row["finite"] is not True:
                    failures.append(f"non_finite_aux_gradient:{record['sample_id']}:{mechanism}:{name}")
                if row["effective"] is True:
                    scope_rows[name]["auxiliary_effective_images"][mechanism][label] += 1
            expected_intersection = sorted(name for name in trainable_names if lm[name]["effective"] and auxiliary[name]["effective"])
            if value["support"]["actual_intersection"] != expected_intersection:
                raise CABGContractError("R0 stored support intersection is not the actual gradient intersection.")
            mechanism_values[mechanism]["loss"].append(float(value["loss"]))
            mechanism_values[mechanism]["score"].append(float(value["score"]))
            mechanism_values[mechanism]["aux_norm_actual"].append(float(value["auxiliary_global_norm_actual_intersection"]))
            mechanism_values[mechanism]["active_hinge"].append(bool(value["active_hinge"]))
        per_image.append(
            {
                "sample_id": record["sample_id"],
                "label": label,
                "a_saturation": float(record["branches"]["all_layer_patch_values"]["abnormal_probability"]["saturation_fraction"]),
                "n_saturation": float(record["branches"]["all_layer_patch_values"]["normal_probability"]["saturation_fraction"]),
                "e_saturation": float(record["branches"]["all_layer_patch_values"]["evidence"]["saturation_fraction"]),
                "raw_gap_mean": image_means["raw_gap"][-1],
                "m2_effective_spatial_support": float(mechanisms["M2"]["spatial"]["effective_support"]),
                "lm_norm_actual_intersection": lm_intersection_norm,
                "m2_aux_norm_actual_intersection": m2_aux_norm,
                "m2_gradient_ratio_actual_intersection": ratio,
            }
        )

    raw_aggregate = {}
    patch_layer_mean_aggregate = {}
    for label in (NORMAL_LABEL, ABNORMAL_LABEL):
        raw_aggregate[label] = {
            branch: distribution_record(torch.cat(values), kind={"abnormal_raw": "raw_logit", "normal_raw": "raw_logit", "raw_gap": "raw_gap", "abnormal_probability": "probability", "normal_probability": "probability", "evidence": "evidence", "lce_softsign_contrast": "evidence"}[branch])
            for branch, values in raw_by_class[label].items()
        }
        patch_layer_mean_aggregate[label] = {
            branch: distribution_record(torch.cat(values), kind={"abnormal_raw": "raw_logit", "normal_raw": "raw_logit", "raw_gap": "raw_gap", "abnormal_probability": "probability", "normal_probability": "probability", "evidence": "evidence", "lce_softsign_contrast": "evidence"}[branch])
            for branch, values in patch_layer_mean_by_class[label].items()
        }
    separation = {branch: _class_image_summary(values, labels) for branch, values in image_means.items()}
    mechanism_aggregate = {}
    for mechanism, values in mechanism_values.items():
        mechanism_aggregate[mechanism] = {
            "classes": {
                label: {
                    "count": labels.count(label),
                    "mean_loss": sum(value for value, observed in zip(values["loss"], labels) if observed == label) / labels.count(label),
                    "mean_score": sum(value for value, observed in zip(values["score"], labels) if observed == label) / labels.count(label),
                    "zero_aux_gradient_images": sum(value <= 1e-12 for value, observed in zip(values["aux_norm_actual"], labels) if observed == label),
                    "active_hinge_images": sum(active for active, observed in zip(values["active_hinge"], labels) if observed == label),
                }
                for label in (NORMAL_LABEL, ABNORMAL_LABEL)
            }
        }

    valid_ratios = [row for row in per_image if row["m2_gradient_ratio_actual_intersection"] is not None]
    correlations = {
        "a_saturation_vs_m2_aux_norm": _correlation_record([row["a_saturation"] for row in per_image], [row["m2_aux_norm_actual_intersection"] for row in per_image]),
        "e_saturation_vs_m2_aux_norm": _correlation_record([row["e_saturation"] for row in per_image], [row["m2_aux_norm_actual_intersection"] for row in per_image]),
        "spatial_support_vs_m2_aux_norm": _correlation_record([row["m2_effective_spatial_support"] for row in per_image], [row["m2_aux_norm_actual_intersection"] for row in per_image]),
        "lm_norm_vs_m2_aux_norm": _correlation_record([row["lm_norm_actual_intersection"] for row in per_image], [row["m2_aux_norm_actual_intersection"] for row in per_image]),
        "class_label_vs_m2_gradient_ratio": _correlation_record([1.0 if row["label"] == ABNORMAL_LABEL else 0.0 for row in valid_ratios], [row["m2_gradient_ratio_actual_intersection"] for row in valid_ratios]),
        "raw_gap_mean_vs_m2_gradient_ratio": _correlation_record([row["raw_gap_mean"] for row in valid_ratios], [row["m2_gradient_ratio_actual_intersection"] for row in valid_ratios]),
    }

    integrity = {
        "parameter_state_unchanged": state_after.get("matches_before") is True and state_before.get("parameters") == state_after.get("parameters"),
        "all_parameter_grads_none": state_after.get("grad_audit", {}).get("all_grad_none") is True,
        "rng_restored_exactly": rng.get("restored_exactly") is True and rng.get("outer_before") == rng.get("after_restore"),
        "file_open_isolation_passed": file_open.get("status") == "SUCCESS" and file_open.get("non_allowlisted_paths") == [] and file_open.get("unopened_allowlisted_paths") == [] and file_open.get("internal_test_outputs_read") == 0,
        "external_peak_mib": int(monitor.get("peak_memory_used_mib", -1)),
        "external_peak_within_limit": int(monitor.get("peak_memory_used_mib", -1)) <= EXTERNAL_PEAK_LIMIT_MIB,
        "source_d3_verification_sha256": model_load.get("source_d3_verification_sha256"),
        "source_d3_evidence_fingerprint": model_load.get("source_d3_evidence_fingerprint"),
        "source_inventory_sha256": sha256_file(args.source_inventory),
    }
    integrity["post_run_gpu"] = _post_run_gpu_record(
        args.post_run_gpu, args.post_run_compute_apps
    )
    integrity["gpu_idle_after_run"] = integrity["post_run_gpu"]["idle_after_run"]
    if not all(value for key, value in integrity.items() if key not in {"external_peak_mib", "source_d3_verification_sha256", "source_d3_evidence_fingerprint", "source_inventory_sha256"}):
        failures.append("engineering_integrity")
    if integrity["source_d3_verification_sha256"] != EXPECTED_D3_VERIFICATION_SHA256 or integrity["source_d3_evidence_fingerprint"] != EXPECTED_D3_EVIDENCE_FINGERPRINT:
        failures.append("d3_provenance")

    actual_m2_intersection_all_images = sorted(
        name for name, row in scope_rows.items()
        if row["lm_effective_images"][NORMAL_LABEL] == 6
        and row["lm_effective_images"][ABNORMAL_LABEL] == 6
        and row["auxiliary_effective_images"]["M2"][NORMAL_LABEL] == 6
        and row["auxiliary_effective_images"]["M2"][ABNORMAL_LABEL] == 6
    )
    legacy_missing_actual = sorted(set(actual_m2_intersection_all_images) - set(SHARED_SUPPORT_NAMES))
    aggregate = {
        "status": "SUCCESS" if not failures else "FAILED",
        "schema_version": "cabg-mil-v1.2-redesign-r0-aggregate-1",
        "claim_scope": "exploratory_read_only_mechanism_diagnostic_only",
        "image_count": 12,
        "forward_count": 12,
        "class_counts": dict(sorted(Counter(labels).items())),
        "actual_trainable_scope": list(scope_rows.values()),
        "support_summary": {
            "legacy_declared_shared_support": list(SHARED_SUPPORT_NAMES),
            "legacy_declared_shared_support_fingerprint": canonical_json_sha256(list(SHARED_SUPPORT_NAMES)),
            "actual_m2_intersection_effective_on_all_12": actual_m2_intersection_all_images,
            "actual_intersection_missing_from_legacy_six": legacy_missing_actual,
            "legacy_six_is_complete_actual_intersection": not legacy_missing_actual and set(actual_m2_intersection_all_images) == set(SHARED_SUPPORT_NAMES),
        },
        "raw_distributions_by_class": raw_aggregate,
        "patch_after_layer_mean_distributions_by_class": patch_layer_mean_aggregate,
        "image_level_class_separation": separation,
        "mechanism_decomposition": mechanism_aggregate,
        "per_image_descriptive_values": per_image,
        "descriptive_correlations": correlations,
        "integrity": integrity,
        "failures": sorted(set(failures)),
        "inferential_statistics_performed": False,
        "effectiveness_claim_authorized": False,
        "confirmatory_use_authorized": False,
    }
    if aggregate["status"] != "SUCCESS":
        raise CABGContractError(f"R0 independent evaluator failed: {aggregate['failures']}")
    _exclusive_json(args.aggregate_output, aggregate)
    verification = {
        "status": "SUCCESS",
        "decision": "R0_DIAGNOSTIC_VERIFIED_NO_EFFECTIVENESS_CLAIM",
        "schema_version": "cabg-mil-v1.2-redesign-r0-verification-1",
        "records_sha256": sha256_file(args.records),
        "aggregate_sha256": sha256_file(args.aggregate_output),
        "model_load_audit_sha256": sha256_file(args.model_load_audit),
        "state_before_sha256": sha256_file(args.state_before),
        "state_after_sha256": sha256_file(args.state_after),
        "file_open_audit_sha256": sha256_file(args.file_open_audit),
        "rng_audit_sha256": sha256_file(args.rng_audit),
        "monitor_summary_sha256": sha256_file(args.monitor_summary),
        "post_run_gpu_sha256": sha256_file(args.post_run_gpu),
        "post_run_compute_apps_sha256": sha256_file(args.post_run_compute_apps),
        "source_inventory_sha256": sha256_file(args.source_inventory),
        "evidence_fingerprint": canonical_json_sha256({"records": sha256_file(args.records), "aggregate": sha256_file(args.aggregate_output), "source": sha256_file(args.source_inventory)}),
        "integrity": integrity,
        "claim_scope": aggregate["claim_scope"],
        "effectiveness_claim_authorized": False,
        "d3r_authorized": False,
        "training_authorized": False,
    }
    _exclusive_json(args.output, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-load-audit", type=Path, required=True)
    parser.add_argument("--state-before", type=Path, required=True)
    parser.add_argument("--state-after", type=Path, required=True)
    parser.add_argument("--file-open-audit", type=Path, required=True)
    parser.add_argument("--rng-audit", type=Path, required=True)
    parser.add_argument("--monitor-summary", type=Path, required=True)
    parser.add_argument("--post-run-gpu", type=Path, required=True)
    parser.add_argument("--post-run-compute-apps", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
