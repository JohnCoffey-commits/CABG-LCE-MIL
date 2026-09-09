#!/usr/bin/env python3
"""Independent D3R evaluator; intentionally imports no stage2k producer code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


RUN_ID = "cabg-lce-mil-v1.2-d3r-v1"
RECORD_SCHEMA = "cabg-lce-mil-v1.2-d3r-record-1"
MANIFEST_SHA256 = "7e581453de5f4abba2eafa87b982eccb751264f66e86ac10f3b92a160ae20db1"
SELECTION_SALT = "cabg-lce-mil-v1.2-d3r-v1"
SOURCE_MANIFEST_HASHES = {
    "training_development": "c27ea1381d81949c1565412b81df6472be6f08061d11aa140191bbd28dd2810f",
    "threshold_validation": "28ea92d7fe0f957ab4c34d5038bd1cc79db5805bd3ad923c0d1383c5c2e942d5",
    "old_d3": "7c2b04d63ba788e13e1e9a4827976aa21d9c7ee1754974f2bc2a2626cedb265c",
    "stage2h_excluded": "98943d6f1f6e333696ad0f38fd3af6d20e82c85239f389a285f9d222de5aeaee",
}
PARENT_COMMIT = "ce5d62d421f070770a388b53af151ef251462148"
FORMAL_R0_COMMIT = "5e49c66150c487f67f4bca1bb5e4cfdcb6615612"
APPROVED_R0_TREE = "fbe20a2e81304c54423d2ee9e16c35848c3a5b14"
POSITIONS = 1024
TAU = 1.0 / math.log(POSITIONS)
FLOOR = 1e-12
ATOL = 2e-6
RTOL = 2e-6
S9 = (
    "model.visual.anomaly_qformer.abnormal_prompt",
    "model.visual.anomaly_qformer.normal_prompt",
    "model.visual.anomaly_qformer.anomaly_attention.query_proj.weight",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.weight",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.bias",
    "model.visual.deep_prompt_embeddings.0",
    "model.visual.deep_prompt_embeddings.1",
    "model.visual.deep_prompt_embeddings.2",
    "model.visual.deep_prompt_embeddings.3",
)
STRUCTURAL_ZERO = "model.visual.anomaly_qformer.anomaly_attention.query_proj.bias"
SCHEMA = (
    ("model.visual.anomaly_qformer.abnormal_prompt", (1, 1, 1280), 1280),
    ("model.visual.anomaly_qformer.anomaly_attention.key_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.anomaly_attention.key_proj.weight", (512, 1280), 655360),
    ("model.visual.anomaly_qformer.anomaly_attention.query_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.anomaly_attention.query_proj.weight", (512, 1280), 655360),
    ("model.visual.anomaly_qformer.gate_scale", (), 1),
    ("model.visual.anomaly_qformer.normal_prompt", (1, 1, 1280), 1280),
    ("model.visual.anomaly_qformer.post_ffn.0.bias", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.0.weight", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.1.bias", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.1.weight", (3584, 3584), 12845056),
    ("model.visual.anomaly_qformer.post_ffn.3.bias", (3584,), 3584),
    ("model.visual.anomaly_qformer.post_ffn.3.weight", (3584, 3584), 12845056),
    ("model.visual.anomaly_qformer.q_former.key_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.q_former.key_proj.weight", (512, 3584), 1835008),
    ("model.visual.anomaly_qformer.q_former.query_proj.bias", (512,), 512),
    ("model.visual.anomaly_qformer.q_former.query_proj.weight", (512, 1280), 655360),
    ("model.visual.deep_prompt_embeddings.0", (10, 1280), 12800),
    ("model.visual.deep_prompt_embeddings.1", (10, 1280), 12800),
    ("model.visual.deep_prompt_embeddings.2", (10, 1280), 12800),
    ("model.visual.deep_prompt_embeddings.3", (10, 1280), 12800),
)
NAMES = tuple(row[0] for row in SCHEMA)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def checkpoint_reaudit(model_dir: Path) -> dict[str, object]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    if not shard_names:
        raise ValueError("Checkpoint contains no shards.")
    shards = []
    for name in shard_names:
        path = model_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append({"file": str(name), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = "".join(f"{row['file']}:{row['sha256']}\n" for row in shards).encode()
    return {
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "shard_count": len(shards),
        "physical_shard_bytes": sum(int(row["bytes"]) for row in shards),
        "checkpoint_shard_list_fingerprint": hashlib.sha256(payload).hexdigest(),
        "shards": shards,
    }


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def load_jsonl(path: Path) -> list[dict[str, object]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(path)
    return values


def write_exclusive(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalized_label(value: object) -> str:
    lowered = str(value).lower()
    if lowered in {"normal", "good", "no", "0"}:
        return "normal"
    if lowered in {"abnormal", "ungood", "yes", "1"}:
        return "abnormal"
    raise ValueError(value)


def selection_key(row: dict[str, object]) -> str:
    raw = "\0".join(
        (SELECTION_SALT, normalized_label(row["label"]), str(row["sha256"]), str(row["relative_path"]))
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def identity_set(rows: list[dict[str, object]], field: str) -> set[str]:
    return {str(row[field]) for row in rows}


def compare(name: str, actual: torch.Tensor, payload: list[object]) -> dict[str, object]:
    expected = torch.tensor(payload, dtype=torch.float32).reshape(actual.shape)
    difference = (actual.float() - expected).abs()
    scale = expected.abs().clamp_min(torch.finfo(torch.float32).tiny)
    maximum_absolute = float(difference.max())
    maximum_relative = float((difference / scale).max())
    return {
        "name": name,
        "shape": list(actual.shape),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "passed": bool(torch.allclose(actual.float(), expected, atol=ATOL, rtol=RTOL)),
    }


def independent_formula(record: dict[str, object]) -> dict[str, object]:
    payload = record["formula_payload"]
    shape = tuple(int(value) for value in payload["tensor_shape"])
    if shape[0] != 1 or shape[-1] != POSITIONS:
        raise ValueError(f"Invalid raw shape: {shape}")
    raw = payload["raw_logits"]
    abnormal = torch.tensor(raw["abnormal_flat"], dtype=torch.float32).reshape(shape)
    normal = torch.tensor(raw["normal_flat"], dtype=torch.float32).reshape(shape)
    gap = abnormal.float() - normal.float()
    contrast = gap / (1.0 + gap.abs())
    derivative = 1.0 / (1.0 + gap.abs()).square()
    layer_mean = contrast.mean(dim=1)
    score = TAU * (torch.logsumexp(layer_mean / TAU, dim=-1) - math.log(POSITIONS))
    target = torch.tensor([1.0 if record["label"] == "abnormal" else -1.0])
    loss = F.softplus(-target * score)
    weights = torch.softmax(layer_mean / TAU, dim=-1)
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
    producer = payload["producer_formula"]
    comparisons = [
        compare("gap", gap, producer["gap_flat"]),
        compare("contrast", contrast, producer["contrast_flat"]),
        compare("derivative", derivative, producer["derivative_flat"]),
        compare("layer_mean", layer_mean, producer["layer_mean_flat"]),
        compare("weights", weights, producer["weights_flat"]),
    ]
    score_match = math.isclose(float(score[0]), float(record["lce"]["score"]), abs_tol=ATOL, rel_tol=RTOL)
    loss_match = math.isclose(float(loss[0]), float(record["lce"]["loss"]), abs_tol=ATOL, rel_tol=RTOL)
    spatial = record["lce"]["spatial"]
    recomputed_spatial = {
        "effective_support": float(entropy.exp()[0]),
        "top11_mass": float(torch.topk(weights, k=11, dim=-1).values.sum(-1)[0]),
        "max_pooling_weight": float(weights.max(-1).values[0]),
        "min_nonzero_pooling_weight": float(torch.where(weights > 0, weights, torch.full_like(weights, float("inf"))).min(-1).values[0]),
        "nonzero_pooling_weights": int((weights > 0).sum(-1)[0]),
    }
    spatial_matches = {
        key: (
            recomputed_spatial[key] == spatial[key]
            if key == "nonzero_pooling_weights"
            else math.isclose(float(recomputed_spatial[key]), float(spatial[key]), abs_tol=ATOL, rel_tol=RTOL)
        )
        for key in recomputed_spatial
    }
    finite = all(torch.isfinite(value).all() for value in (abnormal, normal, gap, contrast, derivative, layer_mean, score, loss, weights, entropy))
    return {
        "formula_comparisons": comparisons,
        "score_match": score_match,
        "loss_match": loss_match,
        "spatial_matches": spatial_matches,
        "finite": bool(finite),
        "formula_passed": all(row["passed"] for row in comparisons) and score_match and loss_match and all(spatial_matches.values()),
        "spatial": recomputed_spatial,
    }


def gradient_gate(rows: list[dict[str, object]], expected_names: tuple[str, ...]) -> dict[str, object]:
    names = tuple(str(row.get("name")) for row in rows)
    passed_rows = [
        row
        for row in rows
        if row.get("present") is True
        and row.get("finite") is True
        and row.get("effective") is True
        and finite_number(row.get("l2_norm"))
        and float(row["l2_norm"]) > FLOOR
    ]
    return {"schema_exact": names == expected_names, "effective_count": len(passed_rows), "passed": names == expected_names and len(passed_rows) == len(expected_names)}


def lce_gradient_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    by_name = {str(row["name"]): row for row in rows}
    schema_exact = tuple(str(row["name"]) for row in rows) == NAMES
    s9_pass = all(
        by_name[name].get("present") is True
        and by_name[name].get("finite") is True
        and finite_number(by_name[name].get("l2_norm"))
        and float(by_name[name]["l2_norm"]) > FLOOR
        for name in S9
    )
    outside = [name for name in NAMES if name not in S9]
    leakage_pass = all(
        by_name[name].get("present") is False
        or (by_name[name].get("finite") is True and finite_number(by_name[name].get("l2_norm")) and float(by_name[name]["l2_norm"]) <= FLOOR)
        for name in outside
    )
    zero = by_name[STRUCTURAL_ZERO]
    structural_zero = zero.get("present") is False or (
        zero.get("finite") is True and finite_number(zero.get("l2_norm")) and float(zero["l2_norm"]) <= FLOOR
    )
    global_norm = math.sqrt(sum(float(by_name[name]["l2_norm"]) ** 2 for name in S9)) if s9_pass else 0.0
    return {
        "schema_exact": schema_exact,
        "s9_effective_count": sum(bool(by_name[name].get("effective")) for name in S9),
        "s9_tensor_pass": s9_pass,
        "s9_global_norm": global_norm,
        "s9_global_pass": math.isfinite(global_norm) and global_norm > FLOOR,
        "leakage_pass": leakage_pass,
        "structural_zero_pass": structural_zero,
        "passed": schema_exact and s9_pass and global_norm > FLOOR and leakage_pass and structural_zero,
    }


def gpu_idle(gpu_path: Path, compute_path: Path) -> dict[str, object]:
    rows = [row.strip() for row in gpu_path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        return {"passed": False, "reason": "gpu row count"}
    pieces = [piece.strip() for piece in rows[0].split(",")]
    memory_used = int(pieces[2])
    utilization = int(pieces[3])
    processes = [row.strip() for row in compute_path.read_text(encoding="utf-8").splitlines() if row.strip()]
    return {
        "gpu_name": pieces[0],
        "memory_total_mib": int(pieces[1]),
        "memory_used_mib": memory_used,
        "utilization_percent_observed": utilization,
        "compute_processes": processes,
        "passed": memory_used == 0 and not processes,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    records = load_jsonl(args.records)
    manifest = load_jsonl(args.manifest)
    model_load = load_json(args.model_load_audit)
    state_before = load_json(args.state_before)
    state_after = load_json(args.state_after)
    rng = load_json(args.rng_audit)
    file_open = load_json(args.file_open_audit)
    monitor = load_json(args.monitor_summary)
    source = load_json(args.source_inventory)
    dataset = load_json(args.dataset_audit)
    exclusion = load_json(args.exclusion_audit)

    source_manifest_paths = {
        "training_development": args.training_development,
        "threshold_validation": args.threshold_validation,
        "old_d3": args.old_d3,
        "stage2h_excluded": args.stage2h_excluded,
    }
    source_manifests = {name: load_jsonl(path) for name, path in source_manifest_paths.items()}

    integrity_failures: list[str] = []
    mechanism_failures: list[str] = []
    if sha256_file(args.manifest) != MANIFEST_SHA256 or len(manifest) != 24:
        integrity_failures.append("manifest_identity")
    source_hash_pass = {
        name: sha256_file(path) == SOURCE_MANIFEST_HASHES[name]
        for name, path in source_manifest_paths.items()
    }
    if not all(source_hash_pass.values()):
        integrity_failures.append("source_manifest_hashes")
    expected_selection = []
    for current_label in ("abnormal", "normal"):
        candidates = [
            row
            for row in source_manifests["training_development"]
            if normalized_label(row["label"]) == current_label
        ]
        candidates.sort(
            key=lambda row: (
                selection_key(row),
                str(row["sha256"]),
                str(row["relative_path"]),
                str(row["sample_id"]),
            )
        )
        expected_selection.extend(candidates[:12])
    expected_manifest_identity = [
        (str(row["sample_id"]), str(row["sha256"]), str(row["relative_path"]))
        for row in expected_selection
    ]
    observed_manifest_identity = [
        (str(row["sample_id"]), str(row["sha256"]), str(row["relative_path"]))
        for row in manifest
    ]
    deterministic_selection_pass = observed_manifest_identity == expected_manifest_identity
    if not deterministic_selection_pass:
        integrity_failures.append("deterministic_manifest_recomputation")
    disjointness = {}
    for name in ("threshold_validation", "old_d3", "stage2h_excluded"):
        disjointness[name] = {
            field: len(identity_set(manifest, field) & identity_set(source_manifests[name], field))
            for field in ("sample_id", "sha256", "relative_path")
        }
    if any(value for group in disjointness.values() for value in group.values()):
        integrity_failures.append("independent_dataset_disjointness")
    old_d3_containment = {
        field: identity_set(source_manifests["old_d3"], field).issubset(
            identity_set(source_manifests["threshold_validation"], field)
        )
        for field in ("sample_id", "sha256", "relative_path")
    }
    if not all(old_d3_containment.values()):
        integrity_failures.append("old_d3_threshold_containment")
    patient_boundary_pass = all(
        row.get("patient_id_available") is False
        and row.get("patient_independence_unverified") is True
        and row.get("reserved_from_future_d4_training") is True
        for row in manifest
    )
    if not patient_boundary_pass:
        integrity_failures.append("patient_or_future_d4_boundary")
    if len(records) != 24 or Counter(row.get("label") for row in records) != Counter({"normal": 12, "abnormal": 12}):
        integrity_failures.append("record_count_or_balance")
    expected_identities = [(row["sample_id"], row["sha256"], row["relative_path"], row["d3r_selection_key"]) for row in manifest]
    observed_identities = [(row.get("sample_id"), row.get("image_sha256"), row.get("relative_path"), row.get("selection_key")) for row in records]
    if expected_identities != observed_identities:
        integrity_failures.append("record_manifest_alignment")
    runtime_schema = model_load.get("runtime_audit", {}).get("trainable_scope", {})
    observed_schema = [(row["name"], tuple(row["shape"]), int(row["elements"])) for row in runtime_schema.get("parameters", [])]
    if observed_schema != list(SCHEMA) or tuple(runtime_schema.get("observed_names", [])) != NAMES:
        integrity_failures.append("trainable_schema")
    source_commit = model_load.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or PARENT_COMMIT not in model_load.get("source_ancestry", [])
    ):
        integrity_failures.append("source_commit")
    equivalence = model_load.get("approved_r0_equivalence", {})
    if equivalence != {
        "local_commit": PARENT_COMMIT,
        "formal_remote_commit": FORMAL_R0_COMMIT,
        "approved_tree": APPROVED_R0_TREE,
        "local_tree": APPROVED_R0_TREE,
        "formal_remote_tree": APPROVED_R0_TREE,
        "trees_equal": True,
    }:
        integrity_failures.append("approved_r0_tree_equivalence")
    if model_load.get("optimizer_created") is not False or model_load.get("scheduler_created") is not False or model_load.get("parameter_update_performed") is not False or model_load.get("checkpoint_saved") is not False:
        integrity_failures.append("read_only_flags")
    if model_load.get("quantization") is not False or model_load.get("offloading") is not False:
        integrity_failures.append("model_mutation_flags")
    if source.get("repository", {}).get("commit") != model_load.get("source_commit") or source.get("repository", {}).get("dirty") is not False:
        integrity_failures.append("source_inventory")
    source_rows = source.get("files", [])
    observed_source_rows = []
    for row in source_rows:
        path = args.repo_root / str(row["path"])
        if not path.is_file():
            observed_source_rows.append({"path": row["path"], "missing": True})
        else:
            observed_source_rows.append(
                {"path": str(row["path"]), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    source_files_pass = observed_source_rows == source_rows and canonical_json_sha256(source_rows) == source.get("fingerprint")
    if not source_files_pass:
        integrity_failures.append("independent_source_file_reaudit")
    checkpoint_observed = checkpoint_reaudit(args.base_model)
    checkpoint_pass = checkpoint_observed == model_load.get("checkpoint_audit") and canonical_json_sha256(checkpoint_observed) == model_load.get("checkpoint_fingerprint")
    if not checkpoint_pass:
        integrity_failures.append("independent_checkpoint_reaudit")

    image_hash_results = []
    image_root = Path(str(dataset["image_root"]))
    for row in manifest:
        image_path = image_root / str(row["relative_path"])
        observed = sha256_file(image_path) if image_path.is_file() else None
        passed = observed == row["sha256"] and image_path.stat().st_size == int(row["bytes"]) if image_path.is_file() else False
        image_hash_results.append({"sample_id": row["sample_id"], "sha256": observed, "passed": passed})
    if not all(row["passed"] for row in image_hash_results):
        integrity_failures.append("selected_image_bytes")
    allowed = set(str(value) for value in dataset.get("allowed_image_paths", []))
    if set(file_open.get("opened_paths", [])) != allowed or file_open.get("non_allowlisted_paths") != [] or file_open.get("unopened_allowlisted_paths") != []:
        integrity_failures.append("file_open_isolation")
    if file_open.get("protected_internal_test_outputs_read") != 0 or exclusion.get("protected_internal_test_outputs_read") != 0:
        integrity_failures.append("protected_internal_test_boundary")

    per_image = []
    tensor_counts = defaultdict(lambda: {"lm_normal": 0, "lm_abnormal": 0, "lce_normal": 0, "lce_abnormal": 0})
    for index, record in enumerate(records):
        row_failures = []
        if record.get("schema_version") != RECORD_SCHEMA or record.get("run_id") != RUN_ID or record.get("manifest_index") != index:
            integrity_failures.append(f"record_schema:{index}")
        provenance = record.get("provenance", {})
        if (
            provenance.get("manifest_sha256") != MANIFEST_SHA256
            or provenance.get("source_commit") != model_load.get("source_commit")
            or provenance.get("source_fingerprint") != source.get("fingerprint")
            or provenance.get("source_fingerprint") != model_load.get("source_fingerprint")
            or provenance.get("checkpoint_fingerprint") != model_load.get("checkpoint_fingerprint")
        ):
            integrity_failures.append(f"record_provenance:{index}")
        formula = independent_formula(record)
        lm = gradient_gate(record["lm_gradients"], NAMES)
        lce = lce_gradient_gate(record["lce"]["gradients"])
        label = str(record["label"])
        for gradient in record["lm_gradients"]:
            if gradient.get("effective"):
                tensor_counts[str(gradient["name"])][f"lm_{label}"] += 1
        for gradient in record["lce"]["gradients"]:
            if gradient.get("effective"):
                tensor_counts[str(gradient["name"])][f"lce_{label}"] += 1
        spatial = formula["spatial"]
        spatial_pass = (
            spatial["effective_support"] >= 128.0
            and spatial["top11_mass"] <= 0.25
            and spatial["max_pooling_weight"] <= 0.05
            and spatial["nonzero_pooling_weights"] == 1024
        )
        numerics = formula["finite"] and finite_number(record.get("lm_loss")) and finite_number(record["lce"].get("score")) and finite_number(record["lce"].get("loss"))
        if not lm["passed"]:
            row_failures.append("lm_path")
        if not lce["passed"]:
            row_failures.append("lce_support_or_leakage")
        if not formula["formula_passed"]:
            row_failures.append("formula")
        if not numerics:
            row_failures.append("numerics")
        if not spatial_pass:
            row_failures.append("spatial")
        if row_failures:
            mechanism_failures.extend(f"image:{index}:{name}" for name in row_failures)
        per_image.append(
            {
                "manifest_index": index,
                "sample_id": record["sample_id"],
                "label": label,
                "lm_gate": lm,
                "lce_gate": lce,
                "formula_gate": formula,
                "spatial_pass": spatial_pass,
                "numerics_pass": numerics,
                "runtime": record["runtime"],
                "memory": record["memory"],
                "failures": row_failures,
            }
        )

    image_seconds = [float(row["runtime"]["image_seconds"]) for row in records]
    forward_seconds = [float(row["runtime"]["forward_seconds"]) for row in records]
    post_load_seconds_values = {float(row["runtime"]["post_load_diagnostic_seconds"]) for row in records}
    external_peak = float(monitor.get("peak_memory_used_mib", float("inf")))
    resource = {
        "external_peak_mib": external_peak,
        "external_peak_limit_mib": 22500,
        "external_peak_pass": external_peak <= 22500,
        "image_seconds_mean": sum(image_seconds) / len(image_seconds),
        "image_seconds_max": max(image_seconds),
        "forward_seconds_mean": sum(forward_seconds) / len(forward_seconds),
        "forward_seconds_max": max(forward_seconds),
        "post_load_diagnostic_seconds": next(iter(post_load_seconds_values)) if len(post_load_seconds_values) == 1 else None,
    }
    resource["runtime_pass"] = resource["image_seconds_mean"] <= 5.0 and resource["image_seconds_max"] <= 10.0 and resource["post_load_diagnostic_seconds"] is not None and resource["post_load_diagnostic_seconds"] <= 900.0
    if not resource["external_peak_pass"]:
        mechanism_failures.append("resource:external_peak")
    if not resource["runtime_pass"]:
        mechanism_failures.append("resource:runtime")

    state_pass = state_after.get("matches_before") is True and state_before.get("parameters") == state_after.get("parameters") and state_after.get("grad_audit", {}).get("all_grad_none") is True
    rng_pass = rng.get("restored_exactly") is True and rng.get("outer_before") == rng.get("after_restore")
    idle = gpu_idle(args.post_run_gpu, args.post_run_compute_apps)
    if not state_pass:
        integrity_failures.append("parameter_state_or_grad")
    if not rng_pass:
        integrity_failures.append("rng_restore")
    if not idle["passed"]:
        integrity_failures.append("post_run_gpu_idle")
    if exclusion.get("status") != "SUCCESS" or any(value for group in exclusion.get("selected_overlap_with_exclusions", {}).values() for value in group.values()):
        integrity_failures.append("dataset_exclusion")

    tensor_table = [
        {
            "name": name,
            "shape": list(shape),
            "elements": elements,
            "s9": name in S9,
            "structural_zero": name == STRUCTURAL_ZERO,
            **tensor_counts[name],
        }
        for name, shape, elements in SCHEMA
    ]
    if integrity_failures:
        decision = "INVALID_QUARANTINED"
        status = "FAILED"
    elif mechanism_failures:
        decision = "PAUSE"
        status = "SUCCESS"
    else:
        decision = "PASS_D3R"
        status = "SUCCESS"
    aggregate = {
        "status": status,
        "decision": decision,
        "schema_version": "cabg-lce-mil-v1.2-d3r-aggregate-1",
        "run_id": RUN_ID,
        "claim_scope": "confirmatory_read_only_mechanism_diagnostic_only",
        "image_count": len(records),
        "forward_count": len({row.get("forward_id") for row in records}),
        "class_counts": dict(sorted(Counter(row.get("label") for row in records).items())),
        "per_image": per_image,
        "per_tensor_support": tensor_table,
        "resource": resource,
        "state_pass": state_pass,
        "rng_pass": rng_pass,
        "file_open_pass": "file_open_isolation" not in integrity_failures,
        "post_run_gpu": idle,
        "selected_image_hashes": image_hash_results,
        "dataset_recomputation": {
            "source_hash_pass": source_hash_pass,
            "deterministic_selection_pass": deterministic_selection_pass,
            "disjointness": disjointness,
            "old_d3_containment_in_threshold_validation": old_d3_containment,
            "patient_boundary_pass": patient_boundary_pass,
            "protected_internal_test_images_opened_by_evaluator": 0,
            "protected_internal_test_outputs_read_by_evaluator": 0,
        },
        "source_reaudit": {
            "files_pass": source_files_pass,
            "file_count": len(source_rows),
            "observed_files": observed_source_rows,
        },
        "checkpoint_reaudit": {
            "passed": checkpoint_pass,
            "observed": checkpoint_observed,
        },
        "integrity_failures": sorted(set(integrity_failures)),
        "mechanism_failures": sorted(set(mechanism_failures)),
        "inferential_statistics_performed": False,
        "effectiveness_claim_authorized": False,
        "training_authorized": False,
        "d4_authorized": False,
    }
    write_exclusive(args.aggregate_output, aggregate)
    verification = {
        "status": status,
        "decision": decision,
        "schema_version": "cabg-lce-mil-v1.2-d3r-verification-1",
        "run_id": RUN_ID,
        "records_sha256": sha256_file(args.records),
        "aggregate_sha256": sha256_file(args.aggregate_output),
        "manifest_sha256": sha256_file(args.manifest),
        "model_load_audit_sha256": sha256_file(args.model_load_audit),
        "state_before_sha256": sha256_file(args.state_before),
        "state_after_sha256": sha256_file(args.state_after),
        "rng_audit_sha256": sha256_file(args.rng_audit),
        "file_open_audit_sha256": sha256_file(args.file_open_audit),
        "monitor_summary_sha256": sha256_file(args.monitor_summary),
        "source_inventory_sha256": sha256_file(args.source_inventory),
        "post_run_gpu_sha256": sha256_file(args.post_run_gpu),
        "post_run_compute_apps_sha256": sha256_file(args.post_run_compute_apps),
        "producer_module_imported": False,
        "producer_math_imported": False,
        "formula_implementation": "independent_inline_fp32",
        "formula_atol": ATOL,
        "formula_rtol": RTOL,
        "integrity_failures": aggregate["integrity_failures"],
        "mechanism_failures": aggregate["mechanism_failures"],
        "claim_scope": aggregate["claim_scope"],
        "effectiveness_claim_authorized": False,
        "patient_independence_verified": False,
        "training_authorized": False,
        "d4_authorized": False,
    }
    write_exclusive(args.output, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    if integrity_failures:
        raise SystemExit(2)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--exclusion-audit", type=Path, required=True)
    parser.add_argument("--training-development", type=Path, required=True)
    parser.add_argument("--threshold-validation", type=Path, required=True)
    parser.add_argument("--old-d3", type=Path, required=True)
    parser.add_argument("--stage2h-excluded", type=Path, required=True)
    parser.add_argument("--model-load-audit", type=Path, required=True)
    parser.add_argument("--state-before", type=Path, required=True)
    parser.add_argument("--state-after", type=Path, required=True)
    parser.add_argument("--file-open-audit", type=Path, required=True)
    parser.add_argument("--rng-audit", type=Path, required=True)
    parser.add_argument("--monitor-summary", type=Path, required=True)
    parser.add_argument("--post-run-gpu", type=Path, required=True)
    parser.add_argument("--post-run-compute-apps", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
