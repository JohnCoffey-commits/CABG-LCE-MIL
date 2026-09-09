#!/usr/bin/env python3
"""Independent artifact and decision verifier for CABG-MIL v1.1 Gate D3."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import (
    ABNORMAL_LABEL,
    EXPECTED_TRAINABLE_ELEMENTS,
    EXPECTED_TRAINABLE_TENSORS,
    NORMAL_LABEL,
    SHARED_SUPPORT_NAMES,
)
from reproduction.stage2i.d3_artifact_schema import validate_mechanism_record
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file


MANIFEST_SHA256 = "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c"
GRADIENT_FLOOR = 1e-12
SATURATION_LIMIT = 0.10
EXTERNAL_PEAK_LIMIT_MIB = 22500
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


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


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=tolerance, abs_tol=tolerance
    )


def _gradient_rows(record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = record["gradients"]["auxiliary_all_trainable_per_tensor"]
    return {str(row["name"]): row for row in rows}


def _verify_inventory(
    inventory_path: Path,
    check_log_path: Path,
    required_paths: list[Path],
    excluded_paths: list[Path],
) -> dict[str, object]:
    if not inventory_path.is_file() or not check_log_path.is_file():
        raise CABGContractError("D3 final verification requires inventory and check log.")
    rows = []
    seen = set()
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        match = SHA_LINE.fullmatch(line)
        if match is None:
            raise CABGContractError(f"Malformed D3 inventory line: {line}")
        expected, path_text = match.groups()
        path = Path(path_text).resolve()
        if path in seen:
            raise CABGContractError(f"Duplicate D3 inventory path: {path}")
        seen.add(path)
        if not path.is_file() or sha256_file(path) != expected:
            raise CABGContractError(f"D3 inventory hash mismatch: {path}")
        rows.append({"path": str(path), "sha256": expected})
    if not rows:
        raise CABGContractError("D3 inventory is empty.")
    missing = sorted(str(path.resolve()) for path in required_paths if path.resolve() not in seen)
    unexpected = sorted(str(path.resolve()) for path in excluded_paths if path.resolve() in seen)
    if missing or unexpected:
        raise CABGContractError(
            f"D3 inventory coverage mismatch: missing={missing}, unexpected_exclusions={unexpected}"
        )
    check_text = check_log_path.read_text(encoding="utf-8")
    if "FAILED" in check_text or not check_text.strip():
        raise CABGContractError("D3 sha256sum check log is absent or failed.")
    return {
        "status": "SUCCESS",
        "inventory_sha256": sha256_file(inventory_path),
        "check_log_sha256": sha256_file(check_log_path),
        "file_count": len(rows),
        "required_paths_present": len(required_paths),
        "explicit_exclusions": sorted(str(path.resolve()) for path in excluded_paths),
    }


def verify(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    preflight = _load_json(args.preflight)
    dataset = _load_json(args.dataset_audit)
    model_load = _load_json(args.model_load_audit)
    state_before = _load_json(args.state_before)
    state_after = _load_json(args.state_after)
    file_open = _load_json(args.file_open_audit)
    rng = _load_json(args.rng_audit)
    monitor = _load_json(args.monitor_summary)
    aggregate = _load_json(args.aggregate)
    records = _load_jsonl(args.records)
    manifest = _load_jsonl(Path(str(dataset["locked_manifest"])))

    if preflight.get("status") != "SUCCESS" or preflight.get("decision") != "READY_FOR_GATE_D3_MODEL_LOAD":
        raise CABGContractError("D3 preflight evidence is invalid.")
    if dataset.get("source_manifest_sha256") != MANIFEST_SHA256 or len(manifest) != 12:
        raise CABGContractError("D3 dataset/manifest provenance changed.")
    if len(records) != 36:
        raise CABGContractError(f"D3 mechanism record count is {len(records)}, not 36.")
    hard_failures = []
    expected_provenance = {
        "source_commit": str(preflight["git"]["commit"]),
        "implementation_fingerprint": str(preflight["source"]["fingerprint"]),
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "support_fingerprint": canonical_json_sha256(list(SHARED_SUPPORT_NAMES)),
    }
    for record in records:
        validate_mechanism_record(record)
        numeric_values = (
            record["losses"]["lm"],
            record["losses"]["auxiliary"],
            record["score"],
            record["gradients"]["lm_shared_global_norm"],
            record["gradients"]["auxiliary_shared_global_norm"],
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise CABGContractError("D3 record contains a non-finite loss, score or norm.")
        all_gradient_rows = (
            list(record["gradients"]["lm_shared_per_tensor"])
            + list(record["gradients"]["auxiliary_all_trainable_per_tensor"])
        )
        if any(row["finite"] is not True for row in all_gradient_rows):
            hard_failures.append(f"non_finite_gradient:{record['sample_id']}:{record['mechanism_id']}")
        if record["provenance"] != expected_provenance:
            raise CABGContractError("D3 record provenance differs from preflight evidence.")

    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    if len(manifest_by_id) != 12:
        raise CABGContractError("D3 manifest sample identities are not unique.")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    if set(grouped) != set(manifest_by_id):
        raise CABGContractError("D3 record and manifest sample identities differ.")

    m2_rows = []
    for sample_id, rows in grouped.items():
        manifest_row = manifest_by_id[sample_id]
        if len(rows) != 3 or {row["mechanism_id"] for row in rows} != {"M0", "M1", "M2"}:
            raise CABGContractError(f"D3 sample lacks an exact M0/M1/M2 triplet: {sample_id}")
        common_fields = ("forward_id", "evidence_fingerprint", "image_sha256", "relative_path", "label")
        for field in common_fields:
            if len({row[field] for row in rows}) != 1:
                raise CABGContractError(f"D3 same-forward field differs within {sample_id}: {field}")
        expected = {
            "image_sha256": str(manifest_row["sha256"]),
            "relative_path": str(manifest_row["relative_path"]),
            "label": normalize_label(manifest_row["label"]),
        }
        if any(rows[0][key] != value for key, value in expected.items()):
            raise CABGContractError(f"D3 record/manifest identity mismatch: {sample_id}")
        m0 = next(row for row in rows if row["mechanism_id"] == "M0")
        m1 = next(row for row in rows if row["mechanism_id"] == "M1")
        if not _close(float(m0["score"]), float(m1["score"])):
            raise CABGContractError(f"D3 M0/M1 hard score differs: {sample_id}")
        m2_rows.append(next(row for row in rows if row["mechanism_id"] == "M2"))
    if len({row["forward_id"] for row in records}) != 12:
        raise CABGContractError("D3 does not contain exactly one forward ID per image.")

    class_counts = Counter(str(row["label"]) for row in m2_rows)
    if class_counts != Counter({NORMAL_LABEL: 6, ABNORMAL_LABEL: 6}):
        raise CABGContractError(f"D3 M2 class counts changed: {class_counts}")
    shared = set(SHARED_SUPPORT_NAMES)
    class_tensor_effective = {
        label: {name: False for name in SHARED_SUPPORT_NAMES}
        for label in (NORMAL_LABEL, ABNORMAL_LABEL)
    }
    class_squared_norms = defaultdict(list)
    saturation_values = defaultdict(list)
    for row in m2_rows:
        label = str(row["label"])
        gradients = _gradient_rows(row)
        if set(gradients) != set(model_load["runtime_audit"]["trainable_scope"]["observed_names"]):
            raise CABGContractError("D3 auxiliary gradient table differs from loaded trainable scope.")
        if len(gradients) != EXPECTED_TRAINABLE_TENSORS:
            raise CABGContractError("D3 auxiliary gradient table does not contain 21 tensors.")
        effective_image = False
        for name in SHARED_SUPPORT_NAMES:
            value = gradients[name]
            effective = bool(value["present"]) and bool(value["finite"]) and float(value["max_abs"]) > GRADIENT_FLOOR
            effective_image = effective_image or effective
            class_tensor_effective[label][name] = class_tensor_effective[label][name] or effective
        if not effective_image:
            hard_failures.append(f"m2_no_effective_shared_gradient:{row['sample_id']}")
        leakage = [
            value
            for name, value in gradients.items()
            if name not in shared
            and (not value["finite"] or (value["present"] and float(value["max_abs"]) != 0.0))
        ]
        if leakage or int(row["gradients"]["non_shared_leakage_count"]) != len(leakage):
            hard_failures.append(f"m2_non_shared_leakage:{row['sample_id']}")
        class_squared_norms[label].append(float(row["gradients"]["auxiliary_shared_global_norm"]) ** 2)
        for key in (
            "evidence_saturation_fraction",
            "abnormal_probability_saturation_fraction",
            "normal_probability_saturation_fraction",
        ):
            value = float(row["evidence"][key])
            saturation_values[key].append(value)
            if value >= SATURATION_LIMIT:
                hard_failures.append(f"per_image_{key}:{row['sample_id']}")
        if row["evidence"]["reconstruction_exact"] is not True or float(row["evidence"]["reconstruction_max_abs"]) != 0.0:
            hard_failures.append(f"an_reconstruction:{row['sample_id']}")
        if bool(row["spatial"]["exact_one_position_collapse"]) or int(row["spatial"]["nonzero_pooling_weights"]) <= 1:
            hard_failures.append(f"m2_spatial_collapse:{row['sample_id']}")

    class_rms = {}
    for label in (NORMAL_LABEL, ABNORMAL_LABEL):
        for name, effective in class_tensor_effective[label].items():
            if not effective:
                hard_failures.append(f"m2_class_tensor_ineffective:{label}:{name}")
        values = class_squared_norms[label]
        class_rms[label] = math.sqrt(sum(values) / len(values))
        if not math.isfinite(class_rms[label]) or class_rms[label] <= GRADIENT_FLOOR:
            hard_failures.append(f"m2_class_rms:{label}")
    saturation_aggregate = {}
    for key, values in saturation_values.items():
        saturation_aggregate[key] = sum(values) / len(values)
        if saturation_aggregate[key] >= SATURATION_LIMIT:
            hard_failures.append(f"aggregate_{key}")

    if model_load.get("status") != "SUCCESS" or any(
        (
            model_load.get("model_dtype") != "bfloat16",
            model_load.get("model_training") is not True,
            model_load.get("gradient_checkpointing") is not True,
            model_load.get("use_cache") is not False,
            model_load.get("anomaly_query_mode") != "single",
            model_load.get("num_pooling_size") != 4,
            model_load.get("output_token_count") != 16,
            model_load.get("fusion_gate_present") is not False,
            model_load.get("optimizer_created") is not False,
            model_load.get("scheduler_created") is not False,
            model_load.get("parameter_update_performed") is not False,
        )
    ):
        raise CABGContractError(f"D3 model-load configuration is invalid: {model_load}")
    scope = model_load["runtime_audit"]["trainable_scope"]
    if (
        int(scope["trainable_parameter_tensors"]) != EXPECTED_TRAINABLE_TENSORS
        or int(scope["trainable_parameter_elements"]) != EXPECTED_TRAINABLE_ELEMENTS
    ):
        raise CABGContractError("D3 loaded trainable scope changed.")

    if state_before.get("parameters") != state_after.get("parameters"):
        hard_failures.append("trainable_parameter_hash_changed")
    if state_after.get("matches_before") is not True or state_after.get("grad_audit", {}).get("all_grad_none") is not True:
        hard_failures.append("parameter_state_or_grad_audit")
    if rng.get("status") != "SUCCESS" or rng.get("restored_exactly") is not True or rng.get("outer_before") != rng.get("after_restore"):
        hard_failures.append("rng_not_restored")
    if (
        file_open.get("status") != "SUCCESS"
        or file_open.get("non_allowlisted_paths") != []
        or file_open.get("unopened_allowlisted_paths") != []
        or int(file_open.get("internal_test_outputs_read", -1)) != 0
        or set(file_open.get("opened_paths", [])) != set(dataset.get("allowed_image_paths", []))
    ):
        hard_failures.append("file_open_isolation")

    post_gpu_parts = [part.strip() for part in args.post_run_gpu.read_text(encoding="utf-8").strip().split(",")]
    if len(post_gpu_parts) != 4:
        raise CABGContractError("D3 post-run GPU record is malformed.")
    post_gpu = {
        "name": post_gpu_parts[0],
        "memory_total_mib": int(post_gpu_parts[1]),
        "memory_used_mib": int(post_gpu_parts[2]),
        "utilization_percent": int(post_gpu_parts[3]),
    }
    external_peak = int(monitor.get("peak_memory_used_mib", -1))
    if (
        monitor.get("status") != "SUCCESS"
        or int(monitor.get("sample_count", 0)) <= 0
        or int(monitor.get("memory_total_mib", -1)) != 23034
        or not Path(str(monitor.get("csv_path", ""))).is_file()
        or sha256_file(Path(str(monitor["csv_path"]))) != monitor.get("csv_sha256")
    ):
        raise CABGContractError("D3 external monitor evidence is invalid.")
    if external_peak > EXTERNAL_PEAK_LIMIT_MIB:
        hard_failures.append("external_peak_limit")
    if post_gpu["memory_used_mib"] > 32 or post_gpu["utilization_percent"] != 0:
        hard_failures.append("gpu_not_idle_after_process")
    if any(float(row["memory"]["run_external_peak_mib"]) != external_peak for row in records):
        raise CABGContractError("D3 record external Peak values differ from monitor evidence.")
    if float(aggregate.get("external_peak_mib", -1)) != external_peak:
        raise CABGContractError("D3 aggregate external Peak differs from monitor evidence.")
    if aggregate.get("class_counts") != {"abnormal": 6, "normal": 6} or aggregate.get("record_count") != 36:
        raise CABGContractError("D3 class aggregate count schema changed.")
    for label in (NORMAL_LABEL, ABNORMAL_LABEL):
        observed_rms = float(aggregate["mechanisms"]["M2"]["classes"][label]["shared_support_rms"])
        if not _close(observed_rms, class_rms[label]):
            raise CABGContractError(f"D3 aggregate M2 RMS mismatch: {label}")
    for key, value in saturation_aggregate.items():
        observed_value = float(aggregate["saturation"][key]["mean_fraction"])
        if not _close(observed_value, value):
            raise CABGContractError(f"D3 aggregate saturation mismatch: {key}")

    source_inventory = args.preflight.parent / "source-inventory.json"
    environment = args.preflight.parent / "environment.json"
    unit_tests = args.post_run_gpu.parent / "unit-tests.json"
    monitor_csv = args.post_run_gpu.parent / "nvidia-smi.csv"
    phase_markers = args.post_run_gpu.parent / "phase-markers.jsonl"
    annotation = Path(str(dataset["annotation"]))
    locked_manifest = Path(str(dataset["locked_manifest"]))
    inventory = _verify_inventory(
        args.inventory,
        args.inventory_check,
        [
            args.preflight,
            args.dataset_audit,
            args.records,
            args.aggregate,
            args.model_load_audit,
            args.state_before,
            args.state_after,
            args.file_open_audit,
            args.rng_audit,
            args.monitor_summary,
            args.post_run_gpu,
            source_inventory,
            environment,
            unit_tests,
            monitor_csv,
            phase_markers,
            annotation,
            locked_manifest,
        ],
        [args.inventory, args.inventory_check, args.output, args.run_status],
    )
    unique_failures = sorted(set(hard_failures))
    decision = "PASS_GATE_D3" if not unique_failures else "PAUSE"
    result = {
        "status": "SUCCESS",
        "schema_version": "cabg-v1.1-gate-d3-verification-1",
        "decision": decision,
        "failed_gates": unique_failures,
        "image_count": 12,
        "forward_count": 12,
        "record_count": 36,
        "class_counts": dict(sorted(class_counts.items())),
        "m2_class_shared_support_rms": class_rms,
        "m2_class_tensor_effective": class_tensor_effective,
        "saturation_aggregate": saturation_aggregate,
        "external_peak_mib": external_peak,
        "external_peak_limit_mib": EXTERNAL_PEAK_LIMIT_MIB,
        "post_run_gpu": post_gpu,
        "parameter_state_unchanged": "trainable_parameter_hash_changed" not in unique_failures,
        "all_parameter_grads_none": state_after.get("grad_audit", {}).get("all_grad_none") is True,
        "rng_restored_exactly": rng.get("restored_exactly") is True,
        "file_open_isolation_passed": "file_open_isolation" not in unique_failures,
        "inventory": inventory,
        "artifacts": {
            "preflight_sha256": sha256_file(args.preflight),
            "dataset_audit_sha256": sha256_file(args.dataset_audit),
            "records_sha256": sha256_file(args.records),
            "aggregate_sha256": sha256_file(args.aggregate),
            "model_load_audit_sha256": sha256_file(args.model_load_audit),
            "state_before_sha256": sha256_file(args.state_before),
            "state_after_sha256": sha256_file(args.state_after),
            "file_open_audit_sha256": sha256_file(args.file_open_audit),
            "rng_audit_sha256": sha256_file(args.rng_audit),
            "monitor_summary_sha256": sha256_file(args.monitor_summary),
            "post_run_gpu_sha256": sha256_file(args.post_run_gpu),
        },
        "evidence_bundle_fingerprint": canonical_json_sha256(
            {
                "records": sha256_file(args.records),
                "aggregate": sha256_file(args.aggregate),
                "monitor": sha256_file(args.monitor_summary),
                "inventory": sha256_file(args.inventory),
            }
        ),
        "next_gate": "D4_ELIGIBLE_BUT_NOT_AUTHORIZED" if decision == "PASS_GATE_D3" else None,
        "gate_d4_authorized": False,
        "claim_scope": "read_only_mechanism_diagnostic_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--model-load-audit", type=Path, required=True)
    parser.add_argument("--state-before", type=Path, required=True)
    parser.add_argument("--state-after", type=Path, required=True)
    parser.add_argument("--file-open-audit", type=Path, required=True)
    parser.add_argument("--rng-audit", type=Path, required=True)
    parser.add_argument("--monitor-summary", type=Path, required=True)
    parser.add_argument("--post-run-gpu", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-check", type=Path, required=True)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
