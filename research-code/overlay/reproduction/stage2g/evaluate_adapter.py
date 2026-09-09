#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

import torch

from evaluation_core import QUESTION, run_image_evaluation, sha256_file
from reproduction.stage2f.evaluate_vqarad_exploratory import initialize_model
from reproduction.stage2f.source_fingerprint import implementation_source_fingerprint


EXPECTED_SOURCE_FINGERPRINT = "fb1fa1efb3f5c63fb0ae56dbe6674656b2286997e3f073f06839120837745d64"


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path.cwd()
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--expected-evaluation-manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter-train-run-id", required=True)
    parser.add_argument("--method", choices=("B0", "A3"), required=True)
    parser.add_argument("--mode", choices=("single", "multiscale"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()

    observed_commit = git_head()
    if observed_commit != args.expected_source_commit:
        raise RuntimeError(
            f"Adapter source commit mismatch: {observed_commit} != {args.expected_source_commit}"
        )
    observed_fingerprint = implementation_source_fingerprint(Path.cwd())
    if observed_fingerprint != EXPECTED_SOURCE_FINGERPRINT:
        raise RuntimeError(
            f"Adapter source fingerprint mismatch: {observed_fingerprint}"
        )
    if sha256_file(args.adapter) != args.expected_adapter_sha256:
        raise RuntimeError("Adapter checkpoint SHA256 mismatch before model load.")
    manifest_document = json.loads(args.adapter_manifest.read_text(encoding="utf-8"))
    if manifest_document.get("checkpoint_sha256") != args.expected_adapter_sha256:
        raise RuntimeError("Adapter manifest/checkpoint SHA256 mismatch.")
    expected_mode = "single" if args.method == "B0" else "multiscale"
    if args.mode != expected_mode:
        raise RuntimeError("Method/query-mode registry mismatch.")

    wrapper, loaded_manifest = initialize_model(
        args.base_model,
        args.adapter,
        args.seed,
        args.mode,
        50176,
        32,
        args.adapter_train_run_id,
        args.repeat,
    )
    if loaded_manifest.get("checkpoint_sha256") != args.expected_adapter_sha256:
        raise RuntimeError("Strict loader returned an unexpected adapter manifest.")
    torch.cuda.reset_peak_memory_stats()

    def generate(model_wrapper, image):
        with torch.inference_mode():
            response, _ = model_wrapper.generate_output(
                {"prompt": QUESTION, "image": image},
                tune_mode="default",
                diff_mode=False,
            )
        return response

    result = run_image_evaluation(
        wrapper=wrapper,
        data_root=args.data_root,
        manifest_path=args.evaluation_manifest,
        predictions_path=args.predictions,
        result_path=args.output,
        expected_manifest_sha256=args.expected_evaluation_manifest_sha256,
        generation=generate,
        run_metadata={
            "run_id": args.run_id,
            "run_role": "paired_method",
            "method": args.method,
            "anomaly_query_mode": args.mode,
            "seed": args.seed,
            "repeat": args.repeat,
            "adapter_train_run_id": args.adapter_train_run_id,
            "adapter": str(args.adapter),
            "adapter_sha256": args.expected_adapter_sha256,
            "adapter_manifest": str(args.adapter_manifest),
            "adapter_manifest_sha256": sha256_file(args.adapter_manifest),
            "adapter_schema_version": loaded_manifest.get("schema_version"),
            "base_model": str(args.base_model),
            "base_model_revision": loaded_manifest["metadata"]["base_model_revision"],
            "base_checkpoint_fingerprint": loaded_manifest["metadata"][
                "base_checkpoint_fingerprint"
            ],
            "adapter_implementation_source_fingerprint": loaded_manifest["metadata"][
                "implementation_source_fingerprint"
            ],
            "evaluation_source_commit": observed_commit,
            "evaluation_source_fingerprint": observed_fingerprint,
        },
        final_metadata=lambda: {
            "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2)
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
