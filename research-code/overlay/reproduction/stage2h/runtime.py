import random
from types import SimpleNamespace

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)

from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
from qwenvl.data.data_qwen import DataCollatorForSupervisedDataset, LazySupervisedDataset
from qwenvl.train.train_qwen import set_model
from reproduction.stage2f.adapter_schema import audit_trainable_scope
from reproduction.stage2h.adapter_schema import assert_stage2h_architecture
from reproduction.stage2h.state_audit import trainable_state_record


def model_arguments():
    return SimpleNamespace(
        tune_mm_llm=False,
        tune_mm_mlp=False,
        tune_mm_vision=False,
        tune_mm_vision_decoder=False,
        tune_mm_vpt=True,
        tune_mm_anomaly=True,
        tune_mm_diff=False,
        diff_only_mode=False,
        reset_vpt=True,
        reset_anomaly=True,
        vpt_tokens_number=10,
        num_pooling_size=4,
        anomaly_query_mode="single",
    )


def load_stage2h_runtime(base_model, *, seed=42, training=False, gradient_checkpointing=False):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(str(base_model))
    set_model(model_arguments(), model)
    visual_reference = next(model.visual.patch_embed.parameters())
    visual_dtype = visual_reference.dtype
    model.visual.deep_prompt_embeddings.to(device=visual_reference.device, dtype=visual_dtype)
    model.visual.anomaly_qformer.to(device=visual_reference.device, dtype=visual_dtype)
    step0_state = trainable_state_record(model)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2H runtime requires CUDA.")
    device = torch.device("cuda:0")
    model.to(device)
    wrapper = object.__new__(Qwen2_5_VL)
    wrapper.llm = model
    wrapper.processor = processor
    wrapper.device = device
    wrapper.temperature = 0.0
    wrapper.top_p = 1.0
    wrapper.repetition_penalty = 1.0
    wrapper.max_new_tokens = 16
    wrapper.output_attentions = False
    architecture = assert_stage2h_architecture(model)
    scope = audit_trainable_scope(model, "single")
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = not training
    model.train(training)
    processor = wrapper.processor.image_processor
    processor.max_pixels = 50176
    processor.min_pixels = 784
    processor.size["longest_edge"] = 50176
    processor.size["shortest_edge"] = 784
    return wrapper, {
        "architecture": architecture,
        "trainable_scope": scope,
        "step0_state": step0_state,
        "gradient_checkpointing": bool(gradient_checkpointing),
    }


def make_stage2h_dataset(base_model, dataset_use, *, shuffle=False):
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), model_max_length=512, padding_side="right", use_fast=False
    )
    image_processor = AutoImageProcessor.from_pretrained(str(base_model))
    data_args = SimpleNamespace(
        dataset_use=dataset_use,
        image_processor=image_processor,
        max_pixels=50176,
        min_pixels=784,
        load_masks=False,
        require_anomaly_labels=True,
        strict_sample_loading=True,
        use_diff_token=False,
        diff_only_mode=False,
        model_type="qwen2.5vl",
        video_max_total_pixels=1664 * 28 * 28,
        video_min_total_pixels=256 * 28 * 28,
        video_max_frame_pixels=32 * 28 * 28,
        video_min_frame_pixels=4 * 28 * 28,
        video_max_frames=8,
        video_min_frames=4,
        base_interval=2,
    )
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        use_anomaly_token=True,
        num_pooling_size=4,
        dataset_role=dataset_use,
        shuffle=shuffle,
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        load_masks=False,
        require_anomaly_labels=True,
    )
    return tokenizer, dataset, collator


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if value is not None
    }
