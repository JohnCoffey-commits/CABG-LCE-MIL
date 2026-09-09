import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_vision_decoder: bool = field(default=False)
    tune_mm_vpt: bool = field(default=False)
    tune_mm_anomaly: bool = field(default=False)
    tune_mm_diff: bool = field(default=False)
    diff_only_mode: bool = field(default=False, metadata={"help": "Use only diff tokens (no anomaly tokens) for VPT ablation."})
    reset_vpt: bool = field(default=True)
    reset_anomaly: bool = field(default=True)
    reset_vision_decoder: bool = field(default=False)
    vpt_tokens_number: int = field(default=0)
    log_interval: int = field(default=500)
    # Jake
    segment_alpha: float = field(default=0.99)
    segment_embed_dim: Optional[int] = field(default=None)
    segment_upsample_factor: Optional[int] = field(
        default=4,
        metadata={"help": "Upsampling factor applied inside the ConvNeXt segmentation head."},
    )
    segment_use_llm_query: bool = field(default=False, metadata={"help": "Enable LLM-hidden-state guided visual decoding."})
    segment_cross_attn_heads: Optional[int] = field(default=None, metadata={"help": "Override number of heads for LLM-guided cross-attention (defaults to vision num_heads)."})
    segment_cross_attn_dropout: float = field(default=0.0, metadata={"help": "Dropout for the LLM-guided cross-attention output."})
    segment_head_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint path to initialize segmentation_head weights (only the segmentation_head)."},
    )
    train_type: str = field(default="default")
    attn_implementation: str = field(default="flash_attention_2")
    segment_vis_interval: int = field(default=0, metadata={"help": "Print segmentation preview metrics every N steps (0 to disable)."})
    segment_vis_save_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store segmentation preview PNGs."})
    seg_decoder_type: str = field(default=None)
    num_pooling_size: int = field(default=2, metadata={"help": "anomaly token pooling size"})
    anomaly_query_mode: str = field(
        default="single",
        metadata={"help": "Anomaly query construction: single or multiscale (FB-MAQ)."},
    )
    gradient_audit_interval: int = field(
        default=0,
        metadata={"help": "Audit every N optimizer steps; 0 disables the gradient audit."},
    )
    gradient_audit_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path for per-parameter gradient audit records."},
    )
    gradient_audit_strict: bool = field(
        default=False,
        metadata={
            "help": (
                "Fail at train end if an audited trainable gradient is missing or non-finite. "
                "Zero values at sparse audit steps are retained as diagnostics."
            )
        },
    )
    loss_trace_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path for unrounded training-loss diagnostics."},
    )
    evidence_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Stage 2H pre-gate evidence-loss weight; B0-AS uses 0."},
    )
    evidence_trace_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Stage 2H per-forward LM/evidence/total loss JSONL."},
    )
    stage2h_step_timing_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Stage 2H optimizer-step wall-time JSONL."},
    )
    diagnostic_eval_only: bool = field(
        default=False,
        metadata={"help": "Run evaluation after Trainer setup without entering the training loop."},
    )
    diagnostic_train_engine_eval_only: bool = field(
        default=False,
        metadata={"help": "Run eval_on_start with the training engine, then stop before the first epoch."},
    )
    diagnostic_eval_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON path for diagnostic evaluation metrics."},
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset_use: str = field(
        default="",
        metadata={"help": "Optional separately registered evaluation dataset."},
    )
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    load_masks: bool = field(default=False, metadata={"help": "Load segmentation masks with samples."})
    require_anomaly_labels: bool = field(
        default=False,
        metadata={"help": "Require one explicit binary anomaly_label per image."},
    )
    strict_sample_loading: bool = field(
        default=False,
        metadata={"help": "Disable retries and fallback sample replacement."},
    )
    use_diff_token: bool = field(default=False, metadata={"help": "Include diff tokens for paired-image inputs."})
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    skip_final_model_save: bool = field(
        default=False,
        metadata={"help": "Skip only the final full-model serialization for diagnostic runs."},
    )
    trainable_state_output: Optional[str] = field(
        default=None,
        metadata={"help": "Optional safetensors path for a trainable-parameters-only checkpoint."},
    )
