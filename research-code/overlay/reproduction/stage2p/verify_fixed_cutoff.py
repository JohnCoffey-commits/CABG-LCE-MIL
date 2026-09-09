"""Independent scalar/state verifier. Does not import the producer schedule/controller.

Recomputes controller quantities from per-image norm observations and checks serialized
state. Aggregate vector norms are logged observations, not independently reconstructed
raw autograd tensors. Existing synthetic gradient regression tests cover that layer.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import torch

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, git_identity, implementation_source_record
from reproduction.stage2i.rng_state import rng_state_fingerprint
from reproduction.stage2l.checkpoint import load_strict, tensor_inventory
from reproduction.stage2m.constants import SOURCE_FILES, TRAINABLE_NAMES
from reproduction.stage2m.summarize_scout import _check_trace, _dynamics, _verify_resume
from reproduction.stage2n.summarize_repair import _prediction_summary, _verify_file_audits
from reproduction.stage2p import ARMS, INITIAL_STATE_SHA256, INITIAL_RNG_SHA256, TRAIN_SHA256, EVAL_SHA256, ORDER_SHA256, CONFIG_SHA256, BASE_SHA256


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b, context):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=2e-6, abs_tol=1e-10), f"Scalar mismatch: {context}: {a} vs {b}")


def verify_data(split_path, preflight_path):
    split, preflight = load(split_path), load(preflight_path)
    require(preflight["status"] == "SUCCESS" and not preflight["repository"]["dirty"], "Source/preflight")
    require(canonical_json_sha256(preflight["base_checkpoint"]) == BASE_SHA256, "Base identity")
    require(preflight["split_audit_sha256"] == sha256_file(split_path), "Split audit changed")
    for key, expected in (("train_manifest", TRAIN_SHA256), ("eval_manifest", EVAL_SHA256)):
        require(split[key + "_sha256"] == expected == sha256_file(Path(split[key])), key)
    for key in ("train_annotation", "eval_annotation"):
        require(sha256_file(Path(split[key])) == split[key + "_sha256"], key)
    for key in ("old_d3_used", "previous_calibration_used", "threshold_or_hyperparameter_data_used"):
        require(split[key] is False, key)
    for key in ("protected_internal_test_image_files_opened", "protected_internal_test_outputs_read"):
        require(split[key] == 0, key)
    train, evaluation = rows(split["train_manifest"]), rows(split["eval_manifest"])
    require(len(train) == 72 and len(evaluation) == 32, "Data counts")
    require(Counter(r["scout_label"] for r in evaluation) == {"abnormal": 20, "normal": 12}, "Eval classes")
    exclusions = []
    source_path = Path("/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2/training-development-manifest.jsonl")
    require(sha256_file(source_path) == split["source_hashes"]["training_development"], "Development source identity")
    source_members = {(r["sample_id"], r["sha256"], r["relative_path"]) for r in rows(source_path)}
    require(all((r["sample_id"], r["sha256"], r["relative_path"]) in source_members for r in train + evaluation), "Non-development image")
    for name, path in (("d3r", "/home/data/medic-ad/cabg-lce-mil-v1.2-d3r-v1/manifest.jsonl"), ("d4_pilot", "/home/data/medic-ad/cabg-lce-mil-v1.2-d4-pilot-v2/manifest.jsonl")):
        require(sha256_file(Path(path)) == split["source_hashes"][name], "Exclusion manifest identity")
        exclusions.extend(rows(path))  # Metadata only; never open excluded images.
    for key in ("sample_id", "sha256", "relative_path"):
        a, b, excluded = ({r[key] for r in rs} for rs in (train, evaluation, exclusions))
        require(not a & b and not (a | b) & excluded, f"Overlap: {key}")
    # Rehash only the 96 allowed development images, never excluded/protected images.
    unique = {r["relative_path"]: r for r in train + evaluation}
    root = Path(split["image_root"]).resolve()
    for relative, r in unique.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Image path escape")
        require(sha256_file(path) == r["sha256"], "Image bytes changed")
    return split, train, evaluation, {"status": "SUCCESS", "allowed_development_images_rehashed": len(unique), "train_eval_and_exclusion_overlaps": 0, "excluded_image_files_opened": 0, "protected_internal_test_image_files_opened": 0, "protected_internal_test_outputs_read": 0}


def failed_block(row):
    return median(r["effective_support"] for r in row["images"]) < 128 or median(r["top11_mass"] for r in row["images"]) > .35


def verify_scalars(trace, arm):
    lm_ema = lce_ema = 0.0
    for i, row in enumerate(trace):
        images, c, g = row["images"], row["controller"], row["gradient"]
        require(Counter(r["label"] for r in images) == {"normal": 1, "abnormal": 2}, "Block composition")
        balanced = {}
        for objective in ("lm", "lce"):
            class_rms = {label: math.sqrt(sum(r[objective + "_s9_norm"] ** 2 for r in images if r["label"] == label) / sum(r["label"] == label for r in images)) for label in ("normal", "abnormal")}
            for label, value in class_rms.items():
                close(row["class_rms"][objective][label], value, objective + label)
            balanced[objective] = math.sqrt(.5 * sum(v ** 2 for v in class_rms.values()) + 1e-16)
            close(row["class_rms"][objective + "_balanced"], balanced[objective], "balanced RMS")
        lm_ema = .9 * lm_ema + .1 * balanced["lm"]
        lce_ema = .9 * lce_ema + .1 * balanced["lce"]
        close(c["lm_ema"], lm_ema, "LM EMA")
        close(c["lce_ema"], lce_ema, "LCE EMA")
        close(c["corrected_lm_rms"], lm_ema / (1 - .9 ** (i + 1)), "corrected LM")
        close(c["corrected_lce_rms"], lce_ema / (1 - .9 ** (i + 1)), "corrected LCE")
        raw = .1 * (lm_ema / (1 - .9 ** (i + 1))) / (lce_ema / (1 - .9 ** (i + 1)) + 1e-8)
        cap = .2 * c["lm_s9_norm"] / (c["lce_s9_norm"] + 1e-8)
        diagnostic = min(raw, cap)
        multiplier = 0.0 if arm == "cabg_fixed_cutoff" and i >= 12 else 1.0
        close(c["lambda_raw"], raw, "lambda raw")
        close(c["lambda_cap"], cap, "lambda cap")
        close(c["lambda_controller"], diagnostic, "lambda diagnostic")
        close(c["lambda_final"], diagnostic * multiplier, "lambda applied")
        require(c["valid_blocks"] == i + 1, "Controller reset/cursor")
        require(c["cap_active"] == (c["lambda_controller"] + 1e-12 < c["lambda_raw"]), "Cap activation flag")
        require(row["fixed_cutoff"] == {"block_number": i + 1, "applied_lce_multiplier": multiplier, "controller_diagnostics_computed": True, "heldout_input_used": False, "optimizer_or_controller_reset": False}, "Fixed cutoff contract")
        close(g["lm_shared_norm"], c["lm_s9_norm"], "LM norm")
        close(g["scaled_auxiliary_shared_norm"], c["lce_s9_norm"] * diagnostic * multiplier, "Applied LCE norm")
        close(c["trust_ratio"], g["scaled_auxiliary_shared_norm"] / (g["lm_shared_norm"] + 1e-12), "trust ratio")
        require(0 <= c["trust_ratio"] <= .20001 and g["cap_checked_before_clip"] is True, "Trust cap")
        close(g["clip_scale"], min(1.0, 1.0 / (g["preclip_full_norm"] + 1e-12)), "clip scale")
        close(g["postclip_full_norm"], g["preclip_full_norm"] * g["clip_scale"], "postclip norm")
        require(row["scheduler"]["step"] == row["optimizer"]["step"] == i + 1, "Optimizer/scheduler reset")
        close(row["optimizer"]["learning_rate_used"], 1e-4 * .5 * (1 + math.cos(math.pi * i / 24)), "LR schedule")
        require(row["parameter_update"]["changed_tensor_count"] <= 21, "Update support")
        if i:
            require(row["rng"]["before_block"] == trace[i-1]["rng"]["after_block"], "RNG discontinuity")


def prefix_distance(left, right):
    require(len(left) == len(right) and len(left) > 0, "Prefix length")
    require(all(math.isfinite(v) for v in [*left, *right]), "Nonfinite prefix")
    return math.sqrt(sum((a-b)**2 for a, b in zip(left, right))) / max(math.sqrt(sum(a*a for a in left)), math.sqrt(sum(b*b for b in right)), 1e-12)


def terminal(candidate_pass, reproduced, material_prefix):
    if material_prefix:
        return "INCONCLUSIVE"
    if not candidate_pass:
        return "NOT_VALIDATED"
    return "PASS_FIXED_CUTOFF" if reproduced else "INCONCLUSIVE"


def endpoint(metric, last_four):
    s = metric["spatial"]
    spatial = s["effective_support_median"] >= 256 and s["support_below_128_count"] <= 8 and s["top11_above_0_35_count"] <= 8 and last_four <= 1
    benefit = metric["auroc"] >= .8966666666666667 and metric["average_precision"] >= .9352228131900764
    return {"spatial_pass": spatial, "benefit_pass": benefit, "self_pass": spatial and benefit}


def verify(root, split_path, original, repo):
    preflight = load(root / "preflight.json")
    split, train, evaluation, isolation = verify_data(split_path, root / "preflight.json")
    identity = git_identity(repo)
    source = implementation_source_record(repo, SOURCE_FILES)
    require(not identity["dirty"], "Evaluator source dirty")
    traces, mids, metrics, dynamics, state_checks = {}, {}, {}, {}, {}
    for arm in ARMS:
        trace = rows(root / arm / "trace.jsonl")
        traces[arm] = trace
        _check_trace(trace, arm)
        verify_scalars(trace, arm)
        _verify_file_audits(root, arm)
        resume = _verify_resume(root, arm)
        require(resume.get("scheduler_exact") and resume.get("controller_exact") and resume.get("repair_state") is None, "No-reset resume")
        require(load(root / arm / "phase1/resume-audit.json")["resumed"] is False, "Not a fresh start")
        require(trace[0]["rng"]["before_block"] == INITIAL_RNG_SHA256, "Initial RNG")
        expected_provenance = dict(trace[0]["provenance"])
        for row in trace:
            require(row["initial_state_sha256"] == INITIAL_STATE_SHA256, "Initialization")
            require(row["provenance"] == expected_provenance, "Provenance drift")
            require(expected_provenance["source_commit"] == preflight["repository"]["commit"] and expected_provenance["implementation_fingerprint"] == preflight["source"]["fingerprint"], "Producer source")
            for key, value in (("base_checkpoint_fingerprint", BASE_SHA256), ("train_manifest_sha256", TRAIN_SHA256), ("eval_manifest_sha256", EVAL_SHA256), ("matched_configuration_sha256", CONFIG_SHA256)):
                require(expected_provenance[key] == value, key)
            expected = [{"sample_id": r["sample_id"], "exposure_id": r["scout_exposure_id"], "sha256": r["sha256"], "label": r["scout_label"]} for r in train[row["block_id"]*3:(row["block_id"]+1)*3]]
            require(row["ordered_samples"] == expected, "Training exposure/order")
            require([{k: r[k] for k in expected[0]} for r in row["images"]] == expected, "Image trace alignment")
        for phase in ("phase1", "phase2"):
            require(canonical_json_sha256(load(root / arm / phase / "initial-state.json")) == INITIAL_STATE_SHA256, "Initial metadata hash")
        checks = {}
        for cursor, name, phase in ((12, "mid-block12.pt", "phase1"), (24, "final-block24.pt", "phase2")):
            payload, manifest = load_strict(root / arm / "checkpoints" / name, expected_provenance)
            r = trace[cursor-1]
            require(payload["sampler_state"] == {"cursor": cursor, "manifest_sha256": TRAIN_SHA256, "order_sha256": ORDER_SHA256}, "Checkpoint sampler/order")
            require(payload["trace_state"] == {"completed_blocks": cursor, "last_record_sha256": r["record_sha256"]}, "Checkpoint trace anchor")
            require(payload["controller_state"] == {"beta": .9, "lm_ema": r["controller"]["lm_ema"], "lce_ema": r["controller"]["lce_ema"], "valid_blocks": cursor}, "Checkpoint EMA")
            require(rng_state_fingerprint(payload["rng_state"]) == r["rng"]["after_block"], "Checkpoint RNG")
            require(payload["scheduler_state"]["last_epoch"] == cursor, "Checkpoint scheduler")
            require(all(int(s["step"].item()) == cursor for s in payload["optimizer_state"]["state"].values()), "Adam step count")
            require(all(torch.equal(payload["adapter_state"][n], payload["master_state"][n].to(torch.bfloat16)) for n in TRAINABLE_NAMES), "Adapter/master casting")
            require(load(root / arm / phase / "checkpoint-save.json") == manifest, "Saved checkpoint record")
            checks[str(cursor)] = {"checkpoint_sha256": manifest["checkpoint_sha256"], "state_integrity": True, "rng_and_trace_anchor": True}
            if cursor == 12:
                require(resume["checkpoint_sha256"] == manifest["checkpoint_sha256"] and resume["controller_state"] == payload["controller_state"], "Own midpoint resume")
                require(resume["optimizer_tensor_inventory_sha256"] == canonical_json_sha256(tensor_inventory({"optimizer": payload["optimizer_state"]})), "Optimizer exact restore")
                mids[arm] = payload
            else:
                report = load(root / "evaluation" / arm / "metrics.json")
                require(report["checkpoint_sha256"] == manifest["checkpoint_sha256"] and report["checkpoint_cursor"] == 24, "Eval final checkpoint")
        state_checks[arm] = checks
        predictions = rows(root / "evaluation" / arm / "predictions.jsonl")
        require([(r["sample_id"], r["sha256"], r["label"], r["relative_path"]) for r in predictions] == [(r["sample_id"], r["sha256"], r["scout_label"], r["relative_path"]) for r in evaluation], "Eval identities")
        require(all(math.isfinite(float(r[k])) for r in predictions for k in ("score", "lce_loss", "effective_support", "top11_mass")), "Nonfinite evaluation")
        metrics[arm] = _prediction_summary(predictions)
        for key in ("auroc", "average_precision"):
            close(metrics[arm][key], report["metrics"][key], "Metric recomputation")
        require(report["predictions_sha256"] == sha256_file(root / "evaluation" / arm / "predictions.jsonl"), "Prediction digest")
        require(report["external_peak_mib"] <= 22500, "Eval resource bound")
        dynamics[arm] = _dynamics(trace, root, arm)
        require(dynamics[arm]["external_peak_mib"] <= 22500, "Training resource bound")
        dynamics[arm]["last_four_failed_blocks"] = sum(failed_block(r) for r in trace[-4:])
        dynamics[arm]["milestones"] = [{"block": i+1, "support_median": median(r["effective_support"] for r in trace[i]["images"]), "top11_mass_median": median(r["top11_mass"] for r in trace[i]["images"]), "lambda_applied": trace[i]["controller"]["lambda_final"], "lambda_controller": trace[i]["controller"]["lambda_controller"]} for i in (3,7,11,15,19,23)]
        dynamics[arm]["second_half"] = {"clip_count": sum(r["gradient"]["clip_scale"] < 1-1e-12 for r in trace[12:]), "diagnostic_cap_count": sum(r["controller"]["cap_active"] for r in trace[12:]), "applied_trust_max": max(r["controller"]["trust_ratio"] for r in trace[12:])}
    a, b = ARMS
    distances = {}
    for field in ("lm_loss", "lce_loss", "effective_support", "top11_mass"):
        distances[field] = prefix_distance(*[[im[field] for r in traces[arm][:12] for im in r["images"]] for arm in ARMS])
    distances["lambda_controller"] = prefix_distance(*[[r["controller"]["lambda_controller"] for r in traces[arm][:12]] for arm in ARMS])
    master_diff = math.sqrt(sum(float((mids[a]["master_state"][n].double()-mids[b]["master_state"][n].double()).square().sum()) for n in TRAINABLE_NAMES))
    update_path = max(sum(r["parameter_update"]["total_l2_change"] for r in traces[arm][:12]) for arm in ARMS)
    distances["block12_master_relative_update_path"] = master_diff / max(update_path, 1e-12)
    flags_equal = [failed_block(r) for r in traces[a][:12]] == [failed_block(r) for r in traces[b][:12]]
    prefix = {"normalized_l2_distances": distances, "threshold": .10, "spatial_failure_flags_equal": flags_equal, "material_divergence": max(distances.values()) > .10 or not flags_equal, "block12_master_l2_difference": master_diff, "rng_sequence_equal": [r["rng"] for r in traces[a][:12]] == [r["rng"] for r in traces[b][:12]], "applied_gradient_hashes_equal": [r["applied_gradient"]["inventory_sha256"] for r in traces[a][:12]] == [r["applied_gradient"]["inventory_sha256"] for r in traces[b][:12]], "block12_state_equal": {key: canonical_json_sha256(tensor_inventory({key: mids[a][key]})) == canonical_json_sha256(tensor_inventory({key: mids[b][key]})) for key in ("adapter_state", "master_state", "optimizer_state")}, "block12_controller_equal": mids[a]["controller_state"] == mids[b]["controller_state"], "block12_scheduler_equal": mids[a]["scheduler_state"] == mids[b]["scheduler_state"]}
    for arm in ("initial", "lm_only", "cabg_lce"):
        prediction_path = original / "evaluation" / arm / "predictions.jsonl"
        historic = rows(prediction_path)
        require([(r["sample_id"], r["sha256"], r["label"]) for r in historic] == [(r["sample_id"], r["sha256"], r["scout_label"]) for r in evaluation], "Historical eval matching")
        metrics["historical_" + arm] = _prediction_summary(historic)
    for key, floor in (("auroc", .8966666666666667), ("average_precision", .9352228131900764)):
        baseline, cabg = metrics["historical_lm_only"][key], metrics["historical_cabg_lce"][key]
        close(baseline + .9 * (cabg - baseline), floor, "Locked benefit floor")
    candidate = endpoint(metrics[b], dynamics[b]["last_four_failed_blocks"])
    s = metrics[a]["spatial"]
    reproduced = s["effective_support_median"] < 128 and s["support_below_128_count"] >= 16 and s["top11_above_0_35_count"] >= 16 and dynamics[a]["last_four_failed_blocks"] >= 2
    return {"status": "SUCCESS", "decision": terminal(candidate["self_pass"], reproduced, prefix["material_divergence"]), "candidate": candidate, "control_reproduced": reproduced, "prefix_comparability": prefix, "metrics": metrics, "training_dynamics": dynamics, "candidate_minus_concurrent_control": {k: metrics[b][k]-metrics[a][k] for k in ("auroc", "average_precision")}, "historical_gain_retention": {k: (metrics[b][k]-metrics["historical_lm_only"][k])/(metrics["historical_cabg_lce"][k]-metrics["historical_lm_only"][k]) for k in ("auroc", "average_precision")}, "state_checks": state_checks, "data_isolation": isolation, "fairness": {"same_initial_state": True, "same_initial_rng": True, "same_72_exposures": True, "same_t21_optimizer_schedule": True, "own_midpoint_exact_resume": True, "no_historical_parent": True, "no_online_heldout_input": True}, "producer_identity": preflight["repository"], "evaluator_identity": identity, "evaluator_fingerprint": source["fingerprint"], "verification_limits": "Recomputes scalar/controller/metrics/state evidence; raw per-image gradient vectors are not retained. Synthetic autograd tests cover gradient composition.", "claim_scope": "development_only_fixed_cutoff_end_to_end_not_independent_effectiveness_validation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--split-audit", type=Path, required=True)
    p.add_argument("--original-root", type=Path)
    p.add_argument("--repo-root", type=Path)
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Refuse overwrite evaluator output")
    result = verify_data(args.split_audit, args.run_root / "preflight.json")[3] if args.data_only else verify(args.run_root, args.split_audit, args.original_root, args.repo_root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
