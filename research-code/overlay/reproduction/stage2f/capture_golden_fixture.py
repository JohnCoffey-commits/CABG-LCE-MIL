#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import AnomalyQformer


SOURCE_COMMIT = "580abb4"
MODEL_SOURCE = Path("transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py")
CONSTRUCTOR_SEED = 73021
INPUT_SEED = 99117


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    tensor_path = output_dir / "golden_tensors.safetensors"
    manifest_path = output_dir / "golden_manifest.json"
    if tensor_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite golden fixture in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = repo_root / MODEL_SOURCE
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    torch.manual_seed(CONSTRUCTOR_SEED)
    rng_before_constructor = torch.get_rng_state().clone()
    module = AnomalyQformer(
        hidden_size=8,
        num_pooling_size=4,
        embed_dim=4,
        spatial_merge_size=2,
        diff_only_mode=False,
    ).cpu().float().eval()
    rng_after_constructor = torch.get_rng_state().clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(INPUT_SEED)
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    visual_features = torch.randn(2, 32, 8, generator=generator)
    merged_visual_features = torch.randn(8, 3584, generator=generator)
    scaled_visual = torch.randn(2, 8, 32, 32, generator=generator)

    rng_before_forward = torch.get_rng_state().clone()
    with torch.inference_mode():
        pooled_query = module.global_pooling(scaled_visual)
        anomaly_token, diff_token = module(
            visual_features,
            merged_visual_features,
            grid_thw,
            diff_mode=True,
        )
    rng_after_forward = torch.get_rng_state().clone()
    if diff_token is None:
        raise RuntimeError("Golden fixture requires a non-empty Diff output.")

    tensors = {
        "rng_before_constructor": rng_before_constructor,
        "rng_after_constructor": rng_after_constructor,
        "rng_before_forward": rng_before_forward,
        "rng_after_forward": rng_after_forward,
        "grid_thw": grid_thw,
        "visual_features": visual_features,
        "merged_visual_features": merged_visual_features,
        "scaled_visual": scaled_visual,
        "pooled_query": pooled_query.contiguous(),
        "anomaly_token": anomaly_token.contiguous(),
        "diff_token": diff_token.contiguous(),
    }
    save_file(tensors, str(tensor_path))

    parameter_records = []
    for name, parameter in module.named_parameters():
        parameter_records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "elements": parameter.numel(),
            }
        )
    manifest = {
        "status": "SUCCESS",
        "fixture_type": "pre_change_b0_golden",
        "source_commit": SOURCE_COMMIT,
        "model_source": str(MODEL_SOURCE),
        "model_source_sha256": sha256_file(source_path),
        "constructor_seed": CONSTRUCTOR_SEED,
        "input_seed": INPUT_SEED,
        "torch_version": torch.__version__,
        "module": {
            "hidden_size": 8,
            "num_pooling_size": 4,
            "embed_dim": 4,
            "spatial_merge_size": 2,
            "diff_only_mode": False,
            "state_dict_keys": list(module.state_dict().keys()),
            "parameters": parameter_records,
            "parameter_tensors": len(parameter_records),
            "parameter_elements": sum(record["elements"] for record in parameter_records),
        },
        "outputs": {
            "pooled_query_shape": list(pooled_query.shape),
            "anomaly_token_shape": list(anomaly_token.shape),
            "diff_token_shape": list(diff_token.shape),
        },
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
