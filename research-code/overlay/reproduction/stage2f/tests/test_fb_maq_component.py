#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import AnomalyQformer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(name: str, observed: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(observed, expected):
        difference = (observed.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{name} is not bitwise equal; max_abs_diff={difference}")


def new_qformer(mode: str) -> AnomalyQformer:
    return AnomalyQformer(
        hidden_size=8,
        num_pooling_size=4,
        embed_dim=4,
        spatial_merge_size=2,
        diff_only_mode=False,
        anomaly_query_mode=mode,
    ).cpu().float().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite component-test evidence: {args.output}")

    manifest_path = args.fixture_dir / "golden_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_path = args.fixture_dir / manifest["tensor_file"]
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise RuntimeError("Golden tensor fixture SHA256 mismatch.")
    golden = load_file(str(tensor_path), device="cpu")

    constructor_seed = int(manifest["constructor_seed"])
    torch.manual_seed(constructor_seed)
    require_equal("single rng before constructor", torch.get_rng_state(), golden["rng_before_constructor"])
    single = new_qformer("single")
    require_equal("single rng after constructor", torch.get_rng_state(), golden["rng_after_constructor"])
    if list(single.state_dict()) != manifest["module"]["state_dict_keys"]:
        raise AssertionError("Single mode changed the pre-change state-dict key order or contents.")
    if single.fusion_gate is not None:
        raise AssertionError("Single mode must not instantiate a fusion gate.")

    with torch.inference_mode():
        pooled = single._build_anomaly_query(golden["scaled_visual"])
        anomaly_token, diff_token = single(
            golden["visual_features"],
            golden["merged_visual_features"],
            golden["grid_thw"],
            diff_mode=True,
        )
    require_equal("single pooled query", pooled, golden["pooled_query"])
    require_equal("single anomaly token", anomaly_token, golden["anomaly_token"])
    require_equal("single diff token", diff_token, golden["diff_token"])
    require_equal("single rng after forward", torch.get_rng_state(), golden["rng_after_forward"])

    torch.manual_seed(constructor_seed)
    multiscale = new_qformer("multiscale")
    require_equal("multiscale rng after constructor", torch.get_rng_state(), golden["rng_after_constructor"])
    single_state = single.state_dict()
    multiscale_state = multiscale.state_dict()
    added_keys = sorted(set(multiscale_state) - set(single_state))
    if added_keys != ["fusion_gate.bias", "fusion_gate.weight"]:
        raise AssertionError(f"Unexpected multiscale state keys: {added_keys}")
    for name, expected in single_state.items():
        require_equal(f"common parameter {name}", multiscale_state[name], expected)
    if multiscale.fusion_gate.weight.numel() + multiscale.fusion_gate.bias.numel() != 17:
        raise AssertionError("Lightweight fusion-gate parameter count is incorrect.")
    require_equal(
        "zero gate weight",
        multiscale.fusion_gate.weight,
        torch.zeros_like(multiscale.fusion_gate.weight),
    )
    expected_bias = torch.full_like(multiscale.fusion_gate.bias, math.log(9.0))
    require_equal("logit(0.9) gate bias", multiscale.fusion_gate.bias, expected_bias)

    fused_query, fusion_weights = multiscale._build_anomaly_query(
        golden["scaled_visual"], return_gate=True
    )
    if list(fused_query.shape) != [2, 8, 4, 4] or list(fusion_weights.shape) != [2, 1, 4, 4]:
        raise AssertionError("FB-MAQ query or gate shape is incorrect.")
    torch.testing.assert_close(
        fusion_weights,
        torch.full_like(fusion_weights, 0.9),
        atol=1e-6,
        rtol=0.0,
    )
    if torch.equal(fused_query, golden["pooled_query"]):
        raise AssertionError("FB-MAQ fused query unexpectedly equals the single-scale query.")

    multiscale.zero_grad(set_to_none=True)
    nonuniform = torch.linspace(-1.0, 2.0, steps=2 * 8 * 32 * 32).reshape(2, 8, 32, 32)
    spatial_weights = torch.linspace(0.2, 1.7, steps=16).reshape(1, 1, 4, 4)
    channel_weights = torch.linspace(0.5, 1.5, steps=8).reshape(1, 8, 1, 1)
    gradient_query, _ = multiscale._build_anomaly_query(nonuniform, return_gate=True)
    loss = (gradient_query * spatial_weights * channel_weights).square().mean()
    loss.backward()
    gradient_norms = {}
    before_step = {}
    for name, parameter in multiscale.fusion_gate.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Fusion gate has missing or non-finite gradient: {name}")
        norm = float(parameter.grad.float().norm().item())
        if norm <= 1e-10:
            raise AssertionError(f"Fusion gate gradient is ineffective: {name} norm={norm}")
        gradient_norms[name] = norm
        before_step[name] = parameter.detach().clone()
    optimizer = torch.optim.SGD(multiscale.fusion_gate.parameters(), lr=1e-2)
    optimizer.step()
    changed = [
        name
        for name, parameter in multiscale.fusion_gate.named_parameters()
        if not torch.equal(parameter.detach(), before_step[name])
    ]
    if sorted(changed) != ["bias", "weight"]:
        raise AssertionError(f"Optimizer did not update both fusion-gate tensors: {changed}")

    with torch.inference_mode():
        multiscale_anomaly, multiscale_diff = multiscale(
            golden["visual_features"],
            golden["merged_visual_features"],
            golden["grid_thw"],
            diff_mode=True,
        )
    if list(multiscale_anomaly.shape) != [2, 16, 3584]:
        raise AssertionError("FB-MAQ did not preserve 16 Ano tokens.")
    require_equal("Diff branch invariance", multiscale_diff, golden["diff_token"])

    try:
        AnomalyQformer(hidden_size=8, num_pooling_size=4, embed_dim=4, anomaly_query_mode="invalid")
    except ValueError:
        invalid_mode_rejected = True
    else:
        invalid_mode_rejected = False
    if not invalid_mode_rejected:
        raise AssertionError("Invalid anomaly query mode was not rejected.")
    try:
        AnomalyQformer(hidden_size=8, num_pooling_size=2, embed_dim=4, anomaly_query_mode="multiscale")
    except ValueError:
        invalid_budget_rejected = True
    else:
        invalid_budget_rejected = False
    if not invalid_budget_rejected:
        raise AssertionError("Multiscale mode accepted a non-16-token budget.")
    if Qwen2_5_VLVisionConfig().anomaly_query_mode != "single":
        raise AssertionError("Legacy/default vision configuration does not resolve to single mode.")
    try:
        Qwen2_5_VLVisionConfig(anomaly_query_mode="invalid")
    except ValueError:
        invalid_config_rejected = True
    else:
        invalid_config_rejected = False
    if not invalid_config_rejected:
        raise AssertionError("Invalid anomaly query mode was accepted by the vision configuration.")

    del multiscale
    gc.collect()
    torch.manual_seed(constructor_seed)
    switched = new_qformer("single")
    rng_before_switch = torch.get_rng_state().clone()
    switched.set_anomaly_query_mode("multiscale")
    require_equal("dynamic mode RNG isolation", torch.get_rng_state(), rng_before_switch)

    result = {
        "status": "SUCCESS",
        "golden_tensor_sha256": manifest["tensor_file_sha256"],
        "single_state_dict_keys_unchanged": True,
        "single_pooled_query_bitwise_equal": True,
        "single_anomaly_token_bitwise_equal": True,
        "single_diff_token_bitwise_equal": True,
        "constructor_rng_isolated": True,
        "dynamic_mode_rng_isolated": True,
        "multiscale_added_keys": added_keys,
        "multiscale_output_tokens": 16,
        "initial_lambda_min": float(fusion_weights.min().item()),
        "initial_lambda_max": float(fusion_weights.max().item()),
        "gate_gradient_norms": gradient_norms,
        "gate_updated_parameters": sorted(changed),
        "diff_branch_bitwise_invariant": True,
        "invalid_mode_rejected": True,
        "invalid_budget_rejected": True,
        "legacy_config_defaults_to_single": True,
        "invalid_config_rejected": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
