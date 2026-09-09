#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


SEEDS = (42, 123, 2026)
METRIC_KEYS = (
    "overall_exact_match_accuracy",
    "closed_accuracy",
    "open_exact_match",
    "open_token_f1",
    "invalid_closed_responses",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label}: observed={observed}, expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.evidence_root
    logs = root / "logs"
    data = root / "data"
    outputs = root / "outputs"
    adapters = root / "adapters"
    required_statuses = (
        "prepare.status",
        "preflight.status",
        "reference-seed-42-generation.status",
        "reference-seed-123-generation.status",
        "reference-seed-2026-generation.status",
        "seed-42.status",
        "seed-123.status",
        "seed-2026.status",
        "multiseed-aggregate.status",
    )
    for filename in required_statuses:
        if (logs / filename).read_text(encoding="utf-8").strip() != "SUCCESS":
            raise RuntimeError(f"Unsuccessful status: {filename}")

    manifest = load_json(data / "manifest.json")
    if manifest.get("status") != "SUCCESS" or manifest.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Dataset manifest is invalid or includes official test data.")
    if manifest.get("train_records") != 32 or manifest.get("validation_records") != 16:
        raise RuntimeError("Dataset split sizes are incorrect.")
    if manifest.get("train_validation_image_overlap") != []:
        raise RuntimeError("Dataset manifest reports train/validation image overlap.")
    for split in ("train", "validation"):
        if sha256_file(data / f"{split}.json") != manifest[f"{split}_annotation_sha256"]:
            raise RuntimeError(f"Annotation SHA mismatch: {split}")
    for record in manifest["records"]:
        image_path = data / "images" / f"{record['id']}.jpg"
        if sha256_file(image_path) != record["local_image_sha256"]:
            raise RuntimeError(f"Image SHA mismatch: {record['id']}")

    references = []
    prediction_ids = None
    for seed in SEEDS:
        reference = load_json(logs / f"reference-seed-{seed}-generation.json")
        if (
            reference.get("status") != "SUCCESS"
            or reference.get("mode") != "untrained_reference"
            or reference.get("seed") != seed
        ):
            raise RuntimeError(f"Reference generation result is invalid: seed-{seed}")
        if reference.get("metrics", {}).get("total") != 16:
            raise RuntimeError(f"Reference generation sample count is incorrect: seed-{seed}")
        current_ids = [record["id"] for record in reference["predictions"]]
        if prediction_ids is None:
            prediction_ids = current_ids
        elif current_ids != prediction_ids:
            raise RuntimeError("Reference generation sample order mismatch.")
        references.append(reference)

    run_results = []
    adapter_summaries = []
    training_peaks = []
    generation_peaks = []
    for seed in SEEDS:
        run_id = f"seed-{seed}"
        adapter_path = adapters / f"{run_id}.safetensors"
        adapter_manifest = load_json(adapters / f"{run_id}.manifest.json")
        if adapter_path.stat().st_size != adapter_manifest.get("checkpoint_bytes"):
            raise RuntimeError(f"Adapter byte-size mismatch: {run_id}")
        if sha256_file(adapter_path) != adapter_manifest.get("checkpoint_sha256"):
            raise RuntimeError(f"Adapter SHA mismatch: {run_id}")
        if adapter_manifest.get("trainable_parameter_tensors") != 21:
            raise RuntimeError(f"Adapter tensor-count mismatch: {run_id}")
        if adapter_manifest.get("trainable_parameter_elements") != 29_561_345:
            raise RuntimeError(f"Adapter element-count mismatch: {run_id}")
        metadata = adapter_manifest.get("metadata", {})
        if metadata.get("seed") != seed or metadata.get("global_step") != 32:
            raise RuntimeError(f"Adapter metadata mismatch: {run_id}")

        trainer_state = load_json(outputs / run_id / "trainer_state.json")
        if trainer_state.get("global_step") != 32:
            raise RuntimeError(f"Trainer global step mismatch: {run_id}")
        train_steps = [
            record.get("step")
            for record in trainer_state.get("log_history", [])
            if "loss" in record and "grad_norm" in record
        ]
        eval_steps = [
            record.get("step")
            for record in trainer_state.get("log_history", [])
            if "eval_loss" in record
        ]
        if train_steps != list(range(1, 33)) or eval_steps != [16, 32]:
            raise RuntimeError(f"Trainer log-history mismatch: {run_id}")
        if list((outputs / run_id).glob("*.safetensors")):
            raise RuntimeError(f"Unexpected full-model checkpoint: {run_id}")

        result = load_json(logs / f"{run_id}-generation.json")
        if result.get("status") != "SUCCESS" or result.get("mode") != "trained_adapter":
            raise RuntimeError(f"Generation result is invalid: {run_id}")
        if result.get("seed") != seed or result.get("adapter_sha256") != adapter_manifest["checkpoint_sha256"]:
            raise RuntimeError(f"Generation adapter identity mismatch: {run_id}")
        if [record["id"] for record in result["predictions"]] != prediction_ids:
            raise RuntimeError(f"Generation sample order mismatch: {run_id}")
        if result.get("official_test_downloaded_or_used") is not False:
            raise RuntimeError(f"Official test usage detected: {run_id}")
        run_results.append(result)

        training_peak = int((logs / f"{run_id}-training-peak-vram-used-mib.txt").read_text().strip())
        generation_peak = int((logs / f"{run_id}-generation-peak-vram-used-mib.txt").read_text().strip())
        if not 0 < training_peak <= 23_034 or not 0 < generation_peak <= 23_034:
            raise RuntimeError(f"Invalid external VRAM peak: {run_id}")
        training_peaks.append(training_peak)
        generation_peaks.append(generation_peak)
        adapter_summaries.append(
            {
                "seed": seed,
                "bytes": adapter_path.stat().st_size,
                "sha256": adapter_manifest["checkpoint_sha256"],
            }
        )

    aggregate = load_json(logs / "multiseed-aggregate.json")
    if aggregate.get("status") != "SUCCESS" or aggregate.get("seeds") != list(SEEDS):
        raise RuntimeError("Aggregate result metadata is invalid.")
    if aggregate.get("reference_seeds") != list(SEEDS):
        raise RuntimeError("Aggregate reference seeds are invalid.")
    expected_reference_metrics = {str(result["seed"]): result["metrics"] for result in references}
    if aggregate.get("reference_metrics_by_seed") != expected_reference_metrics:
        raise RuntimeError("Aggregate reference metrics do not match the reference results.")
    for key in METRIC_KEYS:
        trained_values = [float(result["metrics"][key]) for result in run_results]
        reference_values = [float(result["metrics"][key]) for result in references]
        delta_values = [trained - untrained for trained, untrained in zip(trained_values, reference_values)]
        for label, values in (
            ("trained_aggregate", trained_values),
            ("reference_aggregate", reference_values),
            ("paired_delta_aggregate", delta_values),
        ):
            record = aggregate[label][key]
            if record["values"] != values:
                raise RuntimeError(f"{label} raw values mismatch: {key}")
            assert_close(record["mean"], statistics.fmean(values), f"{label} {key} mean")
            assert_close(record["population_std"], statistics.pstdev(values), f"{label} {key} std")

    summary = {
        "status": "SUCCESS",
        "evidence_root": str(root.resolve()),
        "official_test_downloaded_or_used": False,
        "dataset_records": {"train": 32, "validation": 16},
        "seeds": list(SEEDS),
        "adapter_summaries": adapter_summaries,
        "external_training_peak_vram_used_mib": training_peaks,
        "external_generation_peak_vram_used_mib": generation_peaks,
        "reference_metrics_by_seed": expected_reference_metrics,
        "reference_aggregate": aggregate["reference_aggregate"],
        "trained_aggregate": aggregate["trained_aggregate"],
        "paired_delta_aggregate": aggregate["paired_delta_aggregate"],
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite evidence verification: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
