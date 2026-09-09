#!/usr/bin/env python3

import json
import os
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration


def main() -> None:
    checkpoint_path = Path(
        os.environ.get(
            "MEDIC_AD_TRAIN_OUTPUT",
            "/home/outputs/medic-ad/training-smoke/stage1-one-step-v1",
        )
    )
    if not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(checkpoint_path / "config.json")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        output_loading_info=True,
    )

    failures = {
        "missing_keys": loading_info.get("missing_keys", []),
        "unexpected_keys": loading_info.get("unexpected_keys", []),
        "mismatched_keys": loading_info.get("mismatched_keys", []),
        "error_msgs": loading_info.get("error_msgs", []),
    }
    if any(failures.values()):
        raise RuntimeError(json.dumps(failures, indent=2, default=str))

    visual = model.visual
    if int(visual.vpt_tokens_number) != 10:
        raise ValueError(f"Expected 10 VPT tokens, got {visual.vpt_tokens_number}")
    if int(visual.num_pooling_size) != 4:
        raise ValueError(f"Expected pooling size 4, got {visual.num_pooling_size}")

    target_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "deep_prompt_embeddings" in name
        or (
            "anomaly_qformer" in name
            and ".diff_q_former." not in name
            and ".diff_post_ffn." not in name
        )
    )
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})

    result = {
        "status": "SUCCESS",
        "checkpoint": str(checkpoint_path),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stage1_target_parameters": target_parameter_count,
        "vpt_tokens_number": int(visual.vpt_tokens_number),
        "num_pooling_size": int(visual.num_pooling_size),
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
        "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "loading_info": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
