#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from reproduction.stage2h.evidence import compute_evidence_margin_loss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import AnomalyQformer


TARGET_SUFFIXES = (
    "abnormal_prompt",
    "normal_prompt",
    "anomaly_attention.query_proj.weight",
    "anomaly_attention.query_proj.bias",
    "anomaly_attention.key_proj.weight",
    "anomaly_attention.key_proj.bias",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_bitwise(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if not torch.equal(left, right):
        maximum = float((left.float() - right.float()).abs().max().item())
        raise AssertionError(f"{name} changed; max_abs_diff={maximum}")


def new_qformer() -> AnomalyQformer:
    return AnomalyQformer(
        hidden_size=8,
        num_pooling_size=4,
        embed_dim=4,
        spatial_merge_size=2,
        diff_only_mode=False,
        anomaly_query_mode="single",
    ).cpu().float().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads((args.fixture_dir / "golden_manifest.json").read_text(encoding="utf-8"))
    tensor_path = args.fixture_dir / manifest["tensor_file"]
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise RuntimeError("Golden tensor fixture SHA256 mismatch.")
    golden = load_file(str(tensor_path), device="cpu")
    torch.manual_seed(int(manifest["constructor_seed"]))
    module = new_qformer()

    default_anomaly, default_diff = module(
        golden["visual_features"],
        golden["merged_visual_features"],
        golden["grid_thw"],
        diff_mode=True,
    )
    explicit_anomaly, explicit_diff, evidence = module(
        golden["visual_features"],
        golden["merged_visual_features"],
        golden["grid_thw"],
        diff_mode=True,
        return_evidence=True,
    )
    assert_bitwise("lambda-zero default anomaly token", default_anomaly, golden["anomaly_token"])
    assert_bitwise("lambda-zero default diff token", default_diff, golden["diff_token"])
    assert_bitwise("evidence request anomaly token", explicit_anomaly, default_anomaly)
    assert_bitwise("evidence request diff token", explicit_diff, default_diff)
    fixture_layers = int(golden["visual_features"].shape[0])
    if list(evidence.shape) != [2, fixture_layers, 1024]:
        raise AssertionError(f"Unexpected evidence shape: {list(evidence.shape)}")
    if evidence.dtype != torch.float32 or not torch.isfinite(evidence).all():
        raise AssertionError("Evidence dtype/range precondition failed.")
    if float(evidence.min()) < -1.0 or float(evidence.max()) > 1.0:
        raise AssertionError("sigmoid difference escaped [-1,1].")

    module.zero_grad(set_to_none=True)
    _, _, evidence_for_grad = module(
        golden["visual_features"],
        golden["merged_visual_features"],
        golden["grid_thw"],
        return_evidence=True,
    )
    aux = compute_evidence_margin_loss(evidence_for_grad, torch.tensor([0, 1]))["loss"]
    aux.backward()
    gradients = {}
    parameters = dict(module.named_parameters())
    for name in TARGET_SUFFIXES:
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"Missing/non-finite target gradient: {name}")
        maximum = float(gradient.detach().float().abs().max().item())
        if maximum <= 0.0:
            raise AssertionError(f"Ineffective target gradient: {name}")
        gradients[name] = {"max_abs": maximum, "l2_norm": float(gradient.float().norm().item())}

    isolated = (
        "gate_scale",
        "q_former.query_proj.weight",
        "q_former.key_proj.weight",
        "post_ffn.1.weight",
        "post_ffn.3.weight",
        "diff_q_former.query_proj.weight",
        "diff_post_ffn.1.weight",
    )
    isolation = {}
    for name in isolated:
        gradient = parameters[name].grad
        maximum = 0.0 if gradient is None else float(gradient.detach().float().abs().max().item())
        if gradient is not None and (not torch.isfinite(gradient).all() or maximum != 0.0):
            raise AssertionError(f"Auxiliary loss leaked into {name}: max_abs={maximum}")
        isolation[name] = {"gradient_is_none": gradient is None, "max_abs": maximum}

    before = {name: parameters[name].detach().clone() for name in TARGET_SUFFIXES}
    optimizer = torch.optim.SGD((parameters[name] for name in TARGET_SUFFIXES), lr=1e-2)
    optimizer.step()
    changed = [name for name in TARGET_SUFFIXES if not torch.equal(before[name], parameters[name])]
    if not changed:
        raise AssertionError("Optimizer did not update any evidence-producing target parameter.")
    if any("last_evidence" in name for name, _ in module.named_modules()):
        raise AssertionError("A stale evidence module state was introduced.")
    if hasattr(module, "last_evidence"):
        raise AssertionError("AnomalyQformer exposes forbidden last_evidence state.")

    payload = {
        "status": "SUCCESS",
        "golden_tensor_sha256": manifest["tensor_file_sha256"],
        "lambda_zero_prechange_b0_bitwise_parity": True,
        "evidence_request_preserves_anomaly_tokens_bitwise": True,
        "evidence_shape": list(evidence.shape),
        "evidence_dtype": str(evidence.dtype).removeprefix("torch."),
        "evidence_range": [float(evidence.min()), float(evidence.max())],
        "evidence_loss_finite": bool(torch.isfinite(aux).item()),
        "target_gradients": gradients,
        "auxiliary_gradient_isolation": isolation,
        "optimizer_changed_target_parameters": changed,
        "no_stale_evidence_state": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
