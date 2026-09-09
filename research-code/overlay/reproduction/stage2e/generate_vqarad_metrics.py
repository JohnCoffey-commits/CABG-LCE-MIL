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


ANSWER_PREFIXES = (
    "final answer is",
    "final answer:",
    "answer is",
    "answer:",
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
    tokens = normalize_answer(response).split()
    candidates = {token for token in tokens if token in {"yes", "no"}}
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


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


def metric_summary(predictions):
    closed = [record for record in predictions if record["answer_type"] == "closed"]
    open_records = [record for record in predictions if record["answer_type"] == "open"]
    if not closed or not open_records:
        raise RuntimeError("Both closed and open validation records are required.")
    closed_correct = sum(bool(record["closed_correct"]) for record in closed)
    open_exact = sum(bool(record["open_exact_match"]) for record in open_records)
    overall_correct = closed_correct + open_exact
    return {
        "total": len(predictions),
        "closed_total": len(closed),
        "open_total": len(open_records),
        "closed_correct": closed_correct,
        "open_exact_correct": open_exact,
        "overall_exact_match_accuracy": overall_correct / len(predictions),
        "closed_accuracy": closed_correct / len(closed),
        "open_exact_match": open_exact / len(open_records),
        "open_token_f1": statistics.fmean(record["open_token_f1"] for record in open_records),
        "invalid_closed_responses": sum(record["closed_prediction"] is None for record in closed),
    }


def initialize_model(base_model: Path, seed: int, adapter_path: Path | None, max_pixels: int):
    set_seed(seed)
    args = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        output_attentions=False,
    )
    wrapper = Qwen2_5_VL(str(base_model), args)
    model = wrapper.llm
    model.config.vision_config.vpt_tokens_number = 10
    model.config.vision_config.num_pooling_size = 4
    model.visual.re_init_vpt(10)
    model.visual.re_init_anomaly(4)
    model.visual.set_num_pooling_size(4)
    visual_reference = next(model.visual.patch_embed.parameters())
    visual_device = visual_reference.device
    visual_dtype = visual_reference.dtype
    model.visual.deep_prompt_embeddings.to(device=visual_device, dtype=visual_dtype)
    model.visual.anomaly_qformer.to(device=visual_device, dtype=visual_dtype)
    for name, parameter in model.visual.named_parameters():
        if name.startswith(("deep_prompt_embeddings.", "anomaly_qformer.")):
            if parameter.device != visual_device or parameter.dtype != visual_dtype:
                raise RuntimeError(
                    f"Reinitialized visual parameter placement mismatch: {name} "
                    f"is {parameter.device}/{parameter.dtype}, expected {visual_device}/{visual_dtype}"
                )
    model.eval()

    adapter_manifest = None
    if adapter_path is not None:
        adapter_manifest = load_trainable_checkpoint(model, str(adapter_path))
        metadata = adapter_manifest.get("metadata", {})
        if metadata.get("vpt_tokens_number") != 10 or metadata.get("num_pooling_size") != 4:
            raise RuntimeError("Adapter architecture metadata does not match the evaluator.")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=50176)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite generation evidence: {args.output}")
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
    manifest_by_id = {record["id"]: record for record in manifest["records"]}

    wrapper, adapter_manifest = initialize_model(
        base_model=args.base_model,
        seed=args.seed,
        adapter_path=args.adapter,
        max_pixels=args.max_pixels,
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
    metrics = metric_summary(predictions)
    if not all(
        math.isfinite(float(metrics[key]))
        for key in (
            "overall_exact_match_accuracy",
            "closed_accuracy",
            "open_exact_match",
            "open_token_f1",
        )
    ):
        raise RuntimeError("A generation metric is non-finite.")
    result = {
        "status": "SUCCESS",
        "mode": "trained_adapter" if args.adapter else "untrained_reference",
        "seed": args.seed,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": adapter_manifest.get("checkpoint_sha256") if adapter_manifest else None,
        "dataset_revision": manifest["dataset_revision"],
        "official_test_downloaded_or_used": False,
        "max_pixels": args.max_pixels,
        "greedy_generation": True,
        "max_new_tokens": 16,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "metrics": metrics,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
