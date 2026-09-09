#!/usr/bin/env python3
"""Held-out development evaluation using the locked LCE image score."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.cabg_math import CABGContractError
from reproduction.stage2i.fingerprint import canonical_json_sha256, git_identity, implementation_source_record, sha256_file
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit, NvidiaSmiMonitor
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2l.checkpoint import load_strict
from reproduction.stage2m import RUN_ID, SCHEMA_VERSION
from reproduction.stage2m.constants import EXTERNAL_PEAK_LIMIT_MIB, SEED, SOURCE_FILES, TRAINABLE_NAMES
from reproduction.stage2m.metrics import summarize_metrics
from reproduction.stage2m.train_scout import provenance
from reproduction.stage2n import (
    ADJUSTMENT_RUN_ID,
    ADJUSTMENT_SCHEMA_VERSION,
    MOMENT_RESET_ARM,
    REPAIR_RUN_ID,
    REPAIR_SCHEMA_VERSION,
    SUPPORTED_REPAIR_ARMS,
)
from reproduction.stage2o import CAUSAL_ARMS, RUN_ID as CAUSAL_RUN_ID, SCHEMA_VERSION as CAUSAL_SCHEMA_VERSION
from reproduction.stage2p import ARMS as FIXED_ARMS, RUN_ID as FIXED_RUN_ID, SCHEMA_VERSION as FIXED_SCHEMA, CLAIM_SCOPE as FIXED_SCOPE


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _exclusive_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _exclusive_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for row in rows:
            os.write(descriptor, (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.arm in FIXED_ARMS and args.expected_cursor != 24:
        raise CABGContractError("Fixed-cutoff held-out endpoint is block24 only.")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    preflight, split = _load(args.preflight), _load(args.split_audit)
    manifest = _rows(Path(split["eval_manifest"]))
    if len(manifest) != 32 or Counter(row["scout_label"] for row in manifest) != Counter({"abnormal": 20, "normal": 12}):
        raise CABGContractError("Scout held-out manifest identity changed.")
    if args.arm == "initial" and args.checkpoint is not None:
        raise CABGContractError("Initial Scout reference cannot use a trained checkpoint.")
    if args.arm != "initial" and args.checkpoint is None:
        raise CABGContractError("Trained Scout evaluation requires a checkpoint.")
    monitor = NvidiaSmiMonitor(args.output_dir / "nvidia-smi.csv")
    allowed = [str((Path(split["image_root"]) / row["relative_path"]).resolve()) for row in manifest]
    file_audit = MedicalImageOpenAudit(Path(split["image_root"]), allowed)
    monitor.start()
    monitor_active = True
    wrapper = None
    started = time.perf_counter()
    try:
        os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = str(split["eval_annotation"])
        os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = str(split["image_root"])
        from reproduction.stage2h.runtime import load_stage2h_runtime, make_stage2h_dataset, move_batch_to_device

        wrapper, runtime_audit = load_stage2h_runtime(args.base_model, seed=SEED, training=False, gradient_checkpointing=False)
        model = wrapper.llm
        named = dict(model.named_parameters())
        checkpoint_sha256 = None
        checkpoint_manifest_sha256 = None
        if args.arm != "initial":
            run_provenance = provenance(preflight, split, args.arm)
            payload, checkpoint_manifest = load_strict(args.checkpoint, run_provenance)
            if args.expected_cursor not in (12, 24) or payload["sampler_state"]["cursor"] != args.expected_cursor:
                raise CABGContractError("Scout evaluation checkpoint cursor differs from the requested registered boundary.")
            for name in TRAINABLE_NAMES:
                named[name].data.copy_(payload["adapter_state"][name].to(wrapper.device))
            checkpoint_sha256 = checkpoint_manifest["checkpoint_sha256"]
            checkpoint_manifest_sha256 = sha256_file(args.checkpoint.with_suffix(args.checkpoint.suffix + ".manifest.json"))
        model.eval()
        _, dataset, collator = make_stage2h_dataset(args.base_model, "medic_ad_stage2h_calibration", shuffle=False)
        observed = [(row.get("cabg_sample_id"), row.get("image_sha256"), row.get("scout_eval_rank")) for row in dataset.list_data_dict]
        expected = [(row["sample_id"], row["sha256"], row["scout_eval_rank"]) for row in manifest]
        if observed != expected:
            raise CABGContractError("Scout held-out dataset order/identity changed.")
        file_audit.install()
        predictions: list[dict[str, object]] = []
        is_repair = args.arm in SUPPORTED_REPAIR_ARMS
        is_causal = args.arm in CAUSAL_ARMS
        for index, row in enumerate(manifest):
            image_started = time.perf_counter()
            batch = move_batch_to_device(collator([dataset[index]]), wrapper.device)
            anomaly_labels = batch.pop("anomaly_labels")
            expected_label = 1 if row["scout_label"] == "abnormal" else 0
            if int(anomaly_labels.item()) != expected_label:
                raise CABGContractError("Scout evaluation label/image alignment changed.")
            batch["tune_mode"] = "default"
            batch["return_anomaly_evidence"] = True
            batch["use_cache"] = False
            with torch.inference_mode(), RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
                result = model(**batch)
            observed_maps = observer.finalize(result.anomaly_evidence)
            lce = compute_lce(observed_maps["abnormal_raw"], observed_maps["normal_raw"], [row["scout_label"]])
            values = [lce["score"], lce["per_image_loss"], lce["effective_support"], lce["top11_mass"]]
            if any(not torch.isfinite(value).all() for value in values):
                raise CABGContractError("Scout evaluation produced a non-finite quantity.")
            predictions.append({
                "schema_version": CAUSAL_SCHEMA_VERSION if is_causal else ADJUSTMENT_SCHEMA_VERSION if args.arm == MOMENT_RESET_ARM else REPAIR_SCHEMA_VERSION if is_repair else SCHEMA_VERSION,
                "run_id": CAUSAL_RUN_ID if is_causal else ADJUSTMENT_RUN_ID if args.arm == MOMENT_RESET_ARM else REPAIR_RUN_ID if is_repair else RUN_ID,
                "claim_scope": "development_only_shared_checkpoint_causal_mechanism" if is_causal else "development_only_stability_repair" if is_repair else "development_only_exploratory_effect_scout",
                "arm": args.arm,
                "sample_id": row["sample_id"],
                "sha256": row["sha256"],
                "relative_path": row["relative_path"],
                "label": row["scout_label"],
                "score": float(lce["score"][0]),
                "lce_loss": float(lce["per_image_loss"][0]),
                "effective_support": float(lce["effective_support"][0]),
                "top11_mass": float(lce["top11_mass"][0]),
                "image_seconds": time.perf_counter() - image_started,
            })
            del batch, anomaly_labels, result, observed_maps, lce
            gc.collect()
            torch.cuda.empty_cache()
        if args.arm in FIXED_ARMS:
            for prediction in predictions:
                prediction.update(schema_version=FIXED_SCHEMA, run_id=FIXED_RUN_ID, claim_scope=FIXED_SCOPE)
        _exclusive_jsonl(args.output_dir / "predictions.jsonl", predictions)
        file_result = file_audit.result()
        file_result.update({"arm": args.arm, "protected_internal_test_outputs_read": 0})
        if file_result["status"] != "SUCCESS":
            raise CABGContractError("Scout evaluation file-open allowlist failed.")
        _exclusive_json(args.output_dir / "file-open-audit.json", file_result)
        monitor_result = monitor.stop()
        monitor_active = False
        _exclusive_json(args.output_dir / "monitor-summary.json", monitor_result)
        if monitor_result["peak_memory_used_mib"] > EXTERNAL_PEAK_LIMIT_MIB:
            raise CABGContractError("Scout evaluation VRAM limit exceeded.")
        metrics = summarize_metrics(predictions)
        evaluator_identity = None
        if args.repo_root is not None:
            evaluator_identity = {
                "repository": git_identity(args.repo_root),
                "source": implementation_source_record(args.repo_root, SOURCE_FILES),
            }
        result = {
            "status": "SUCCESS",
            "decision": "D4_SCOUT_HELDOUT_EVALUATION_COMPLETE",
            "arm": args.arm,
            "claim_scope": "development_only_shared_checkpoint_causal_mechanism" if is_causal else "development_only_stability_repair" if is_repair else "development_only_exploratory_effect_scout",
            "metrics": metrics,
            "eval_manifest_sha256": split["eval_manifest_sha256"],
            "predictions_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
            "source_commit": preflight["repository"]["commit"],
            "implementation_fingerprint": preflight["source"]["fingerprint"],
            "initial_state_sha256": canonical_json_sha256(runtime_audit["step0_state"]),
            "evaluated_state_sha256": canonical_json_sha256(trainable_state_record(model)),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_cursor": None if args.arm == "initial" else args.expected_cursor,
            "evaluator_identity": evaluator_identity,
            "external_peak_mib": monitor_result["peak_memory_used_mib"],
            "wall_seconds": time.perf_counter() - started,
            "protected_internal_test_image_files_opened": 0,
            "protected_internal_test_outputs_read": 0,
            "threshold_selected": False,
            "vqa_exact_match_used_as_effectiveness_metric": False,
        }
        if args.arm in FIXED_ARMS:
            result["claim_scope"] = FIXED_SCOPE
        _exclusive_json(args.output_dir / "metrics.json", result)
        return result
    finally:
        if monitor_active:
            try:
                monitor_result = monitor.stop()
                if not (args.output_dir / "monitor-summary.json").exists():
                    _exclusive_json(args.output_dir / "monitor-summary.json", monitor_result)
            except Exception:
                pass
        if wrapper is not None:
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("initial", "cabg_lce", "lm_only", *SUPPORTED_REPAIR_ARMS, *CAUSAL_ARMS, *FIXED_ARMS), required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-cursor", type=int, default=24)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
