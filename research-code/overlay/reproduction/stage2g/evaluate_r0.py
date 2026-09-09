#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation_core import QUESTION, run_image_evaluation, sha256_file
from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path.cwd()
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--expected-evaluation-manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    args = parser.parse_args()

    observed_commit = git_head()
    if observed_commit != args.expected_source_commit:
        raise RuntimeError(
            f"R0 source commit mismatch: {observed_commit} != {args.expected_source_commit}"
        )
    checkpoint_audit = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    if checkpoint_audit.get("status") != "SUCCESS":
        raise RuntimeError("R0 checkpoint audit is not successful.")
    if checkpoint_audit.get("model_revision") != args.expected_model_revision:
        raise RuntimeError("R0 checkpoint revision mismatch.")

    wrapper_args = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        output_attentions=False,
    )
    wrapper = Qwen2_5_VL(str(args.model), wrapper_args)
    wrapper.llm.eval()
    processor = wrapper.processor.image_processor
    processor.max_pixels = 50176
    processor.min_pixels = 784
    processor.size["longest_edge"] = 50176
    processor.size["shortest_edge"] = 784
    torch.cuda.reset_peak_memory_stats()

    def generate(model_wrapper, image):
        with torch.inference_mode():
            response, _ = model_wrapper.generate_output(
                {"prompt": QUESTION, "image": image},
                tune_mode="default",
                diff_mode=False,
            )
        return response

    vision_config = wrapper.llm.config.vision_config
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
            "run_role": "published_reference",
            "method": "R0",
            "anomaly_query_mode": getattr(vision_config, "anomaly_query_mode", "single"),
            "seed": None,
            "repeat": None,
            "model": str(args.model),
            "model_revision": args.expected_model_revision,
            "checkpoint_audit": str(args.checkpoint_audit),
            "checkpoint_audit_sha256": sha256_file(args.checkpoint_audit),
            "checkpoint_shard_list_fingerprint": checkpoint_audit[
                "checkpoint_shard_list_fingerprint"
            ],
            "evaluation_source_commit": observed_commit,
            "num_pooling_size": getattr(vision_config, "num_pooling_size", None),
            "vpt_tokens_number": getattr(vision_config, "vpt_tokens_number", None),
            "excluded_from_fb_maq_decision": True,
        },
        final_metadata=lambda: {
            "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2)
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
