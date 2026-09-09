#!/usr/bin/env python3

import argparse
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import set_seed

from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
from qwenvl.train.trainable_checkpoint import load_trainable_checkpoint
from reproduction.stage2f.adapter_schema import (
    require_metadata_fields,
    stage1_architecture_parameter_names,
    strict_adapter_metadata,
    normalize_query_mode,
)


ANSWER_PREFIXES = ("final answer is", "final answer:", "answer is", "answer:")
REQUIRED_ADAPTER_FIELDS = (
    "base_model_revision",
    "base_checkpoint_fingerprint",
    "dataset_manifest_sha256",
    "implementation_source_fingerprint",
    "source_base_commit",
    "anomaly_query_mode",
    "output_token_count",
    "method_id",
    "gate_type",
    "run_id",
    "repeat",
    "stage",
    "seed",
    "data_seed",
    "global_step",
)


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"<answer>(.*?)</answer>", r"\1", text, flags=re.DOTALL)
    text = text.replace("\n", " ")
    for prefix in ANSWER_PREFIXES:
        if prefix in text:
            text = text.split(prefix, 1)[1]
            break
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def closed_prediction(response: str):
    candidates = {token for token in normalize_answer(response).split() if token in {"yes", "no"}}
    return next(iter(candidates)) if len(candidates) == 1 else None


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalize_answer(reference).split()
    prediction_tokens = normalize_answer(prediction).split()
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_summary(predictions, allow_partial: bool = False):
    closed = [record for record in predictions if record["answer_type"] == "closed"]
    open_records = [record for record in predictions if record["answer_type"] == "open"]
    if (not closed or not open_records) and not allow_partial:
        raise RuntimeError("Both closed and open validation records are required.")
    closed_correct = sum(bool(record["closed_correct"]) for record in closed)
    open_exact = sum(bool(record["open_exact_match"]) for record in open_records)
    return {
        "total": len(predictions),
        "closed_total": len(closed),
        "open_total": len(open_records),
        "closed_correct": closed_correct,
        "open_exact_correct": open_exact,
        "overall_exact_match_accuracy": (closed_correct + open_exact) / len(predictions),
        "closed_accuracy": closed_correct / len(closed) if closed else None,
        "open_exact_match": open_exact / len(open_records) if open_records else None,
        "open_token_f1": (
            statistics.fmean(record["open_token_f1"] for record in open_records)
            if open_records
            else None
        ),
        "invalid_closed_responses": sum(record["closed_prediction"] is None for record in closed),
        "empty_responses": sum(not record["normalized_response"] for record in predictions),
    }


def assert_mode_contract(model, mode: str) -> None:
    observed_model = getattr(model.visual, "anomaly_query_mode", None)
    observed_config = getattr(model.config.vision_config, "anomaly_query_mode", None)
    observed_qformer = getattr(model.visual.anomaly_qformer, "anomaly_query_mode", None)
    observed = {observed_model, observed_config, observed_qformer}
    if observed != {mode}:
        raise RuntimeError(
            f"Evaluator mode contract failed: CLI={mode}, model={observed_model}, "
            f"config={observed_config}, qformer={observed_qformer}"
        )


def initialize_model(
    base_model: Path,
    adapter: Path,
    seed: int,
    mode: str,
    max_pixels: int,
    expected_global_step: int,
    run_id: str,
    repeat: int,
):
    set_seed(seed)
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
    wrapper = Qwen2_5_VL(str(base_model), wrapper_args)
    model = wrapper.llm
    model.config.vision_config.vpt_tokens_number = 10
    model.config.vision_config.num_pooling_size = 4
    model.config.vision_config.anomaly_query_mode = mode
    model.visual.re_init_vpt(10)
    model.visual.re_init_anomaly(4, mode)
    model.visual.set_num_pooling_size(4)
    model.visual.set_anomaly_query_mode(mode)

    visual_reference = next(model.visual.patch_embed.parameters())
    visual_device = visual_reference.device
    visual_dtype = visual_reference.dtype
    model.visual.deep_prompt_embeddings.to(device=visual_device, dtype=visual_dtype)
    model.visual.anomaly_qformer.to(device=visual_device, dtype=visual_dtype)
    assert_mode_contract(model, mode)

    expected_names = stage1_architecture_parameter_names(model, mode)
    expected_metadata = strict_adapter_metadata(mode)
    expected_metadata.update(
        {
            "seed": seed,
            "data_seed": seed,
            "global_step": expected_global_step,
            "train_type": "default",
            "full_determinism": False,
            "run_id": run_id,
            "repeat": repeat,
            "stage": "2F",
        }
    )
    adapter_manifest = load_trainable_checkpoint(
        model,
        str(adapter),
        strict_schema=True,
        expected_metadata=expected_metadata,
        expected_parameter_names=expected_names,
    )
    require_metadata_fields(adapter_manifest.get("metadata", {}), REQUIRED_ADAPTER_FIELDS)
    assert_mode_contract(model, mode)

    for name in expected_names:
        parameter = dict(model.named_parameters())[name]
        if parameter.device != visual_device or parameter.dtype != visual_dtype:
            raise RuntimeError(
                f"Adapter parameter placement mismatch: {name} is "
                f"{parameter.device}/{parameter.dtype}, expected {visual_device}/{visual_dtype}"
            )
    model.eval()

    processor = wrapper.processor.image_processor
    processor.max_pixels = max_pixels
    processor.min_pixels = 784
    processor.size["longest_edge"] = max_pixels
    processor.size["shortest_edge"] = 784
    return wrapper, adapter_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--anomaly-query-mode", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--expected-global-step", type=int, default=32)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    mode = normalize_query_mode(args.anomaly_query_mode)

    for path in (args.output, args.predictions_jsonl):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 2F evidence: {path}")
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCESS":
        raise RuntimeError("Dataset manifest is not successful.")
    if manifest.get("official_test_downloaded_or_used") is not False:
        raise RuntimeError("Official test data must remain unused.")
    annotations = json.loads((args.data_root / "validation.json").read_text(encoding="utf-8"))
    if len(annotations) != 16:
        raise RuntimeError(f"Expected 16 validation records, got {len(annotations)}")
    if args.max_samples > 0:
        annotations = annotations[: args.max_samples]
    if args.engineering_smoke and args.max_samples <= 0:
        raise ValueError("Engineering smoke evaluation requires a positive --max-samples limit.")
    manifest_by_id = {record["id"]: record for record in manifest["records"]}

    wrapper, adapter_manifest = initialize_model(
        args.base_model,
        args.adapter,
        args.seed,
        mode,
        args.max_pixels,
        args.expected_global_step,
        args.run_id,
        args.repeat,
    )
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    started = time.perf_counter()
    for annotation in annotations:
        source = manifest_by_id[annotation["id"]]
        image_path = args.data_root / "images" / annotation["image"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            prompt = annotation["conversations"][0]["value"]
            if prompt.startswith("<image>"):
                prompt = prompt[len("<image>") :].lstrip()
            sample_started = time.perf_counter()
            with torch.inference_mode():
                response, _ = wrapper.generate_output(
                    {"prompt": prompt, "image": image},
                    tune_mode="default",
                    diff_mode=False,
                )
            latency = time.perf_counter() - sample_started

        answer = annotation["conversations"][1]["value"]
        answer_type = source["answer_type"]
        normalized_answer = normalize_answer(answer)
        normalized_response = normalize_answer(response)
        record = {
            "id": annotation["id"],
            "answer_type": answer_type,
            "question": source["question"],
            "answer": answer,
            "response": response,
            "normalized_answer": normalized_answer,
            "normalized_response": normalized_response,
            "latency_seconds": latency,
            "closed_prediction": None,
            "closed_correct": None,
            "open_exact_match": None,
            "open_token_f1": None,
        }
        if answer_type == "closed":
            prediction = closed_prediction(response)
            record["closed_prediction"] = prediction
            record["closed_correct"] = prediction == normalized_answer
        else:
            record["open_exact_match"] = normalized_response == normalized_answer
            record["open_token_f1"] = token_f1(answer, response)
        predictions.append(record)

    elapsed = time.perf_counter() - started
    metrics = metric_summary(predictions, allow_partial=args.engineering_smoke)
    numeric_metrics = (
        "overall_exact_match_accuracy",
        "closed_accuracy",
        "open_exact_match",
        "open_token_f1",
    )
    if not all(
        metrics[key] is None or math.isfinite(float(metrics[key]))
        for key in numeric_metrics
    ):
        raise RuntimeError("A Stage 2F generation metric is non-finite.")

    result = {
        "status": "SUCCESS",
        "stage": "2F",
        "evaluation_role": "engineering_smoke" if args.engineering_smoke else "exploratory_vqa_guardrail",
        "anomaly_effectiveness_evaluated": False,
        "run_id": args.run_id,
        "seed": args.seed,
        "repeat": args.repeat,
        "anomaly_query_mode": mode,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_manifest["checkpoint_sha256"],
        "adapter_schema_version": adapter_manifest["schema_version"],
        "dataset_revision": manifest["dataset_revision"],
        "official_test_downloaded_or_used": False,
        "validation_records": len(predictions),
        "max_pixels": args.max_pixels,
        "greedy_generation": True,
        "max_new_tokens": 16,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "metrics": metrics,
        "predictions_jsonl": str(args.predictions_jsonl),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in predictions),
        encoding="utf-8",
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
