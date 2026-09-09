# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import gc
import logging
import math
import os
import pathlib
import sys
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from transformers import TrainerCallback
from transformers.modeling_utils import load_state_dict


class DiagnosticEvaluationComplete(RuntimeError):
    def __init__(self, metrics):
        super().__init__("Diagnostic evaluation completed before the first training epoch.")
        self.metrics = dict(metrics)


class StopAfterInitialEvaluationCallback(TrainerCallback):
    def __init__(self):
        self.metrics = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics = dict(metrics)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.metrics is not None:
            raise DiagnosticEvaluationComplete(self.metrics)
        return control


def wandb_is_available() -> bool:
    return wandb is not None and hasattr(wandb, "log") and hasattr(wandb, "run")


def has_active_wandb_run() -> bool:
    return (
        wandb_is_available()
        and getattr(wandb, "run", None) is not None
    )


class TrainableParamSumCallback(TrainerCallback):
    def __init__(self, log_interval: int = 500, group_depth: int = 0, log_to_wandb: bool = True):
        super().__init__()
        try:
            interval_val = int(log_interval)
        except (TypeError, ValueError):
            interval_val = 500
        self.log_interval = max(interval_val, 1)
        try:
            depth_val = int(group_depth)
        except (TypeError, ValueError):
            depth_val = 0
        self.group_depth = max(depth_val, 0)
        self.log_to_wandb = bool(log_to_wandb) and wandb_is_available()
        self._last_logged_step: Optional[int] = None
        self._module_info: Optional[Dict[str, str]] = None

    @staticmethod
    def _world_is_initialized() -> bool:
        return (
            hasattr(torch, "distributed")
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    @staticmethod
    def _rank() -> int:
        if TrainableParamSumCallback._world_is_initialized():
            return torch.distributed.get_rank()
        return 0

    @staticmethod
    def _is_main_process() -> bool:
        return TrainableParamSumCallback._rank() == 0

    @staticmethod
    def _reduce_device_from_param(param: torch.nn.Parameter) -> torch.device:
        device = getattr(param, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            return device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @staticmethod
    def _plain_param_stats(param: torch.nn.Parameter):
        if not isinstance(param, torch.nn.Parameter):
            return None
        device = getattr(param, "device", None)
        if device is None or getattr(device, "type", None) == "meta":
            return None
        numel = param.numel()
        if numel == 0:
            ds_numel = getattr(param, "ds_numel", None)
            return (0.0, int(ds_numel) if ds_numel is not None else 0)
        with torch.no_grad():
            total_sum = param.detach().sum(dtype=torch.float64).item()
        return float(total_sum), int(numel)

    @staticmethod
    def _ds_param_stats(param: torch.nn.Parameter):
        if not hasattr(param, "ds_status"):
            return None
        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        except Exception:
            return None

        status = getattr(param, "ds_status", None)
        if status == ZeroParamStatus.AVAILABLE:
            return TrainableParamSumCallback._plain_param_stats(param)

        reduce_device = TrainableParamSumCallback._reduce_device_from_param(param)
        sum_count = torch.zeros(2, device=reduce_device, dtype=torch.float64)
        with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
            if TrainableParamSumCallback._is_main_process():
                stats = TrainableParamSumCallback._plain_param_stats(param)
                if stats is not None:
                    sum_count[0] = stats[0]
                    sum_count[1] = float(stats[1])
                else:
                    ds_numel = getattr(param, "ds_numel", 0)
                    sum_count[0] = 0.0
                    sum_count[1] = float(ds_numel)
        if TrainableParamSumCallback._world_is_initialized():
            torch.distributed.broadcast(sum_count, src=0)
        return float(sum_count[0].cpu().item()), int(sum_count[1].cpu().item())

    @staticmethod
    def _param_stats(param: torch.nn.Parameter):
        stats = TrainableParamSumCallback._ds_param_stats(param)
        if stats is not None:
            return stats
        return TrainableParamSumCallback._plain_param_stats(param)

    @staticmethod
    def _unwrap(model):
        return getattr(model, "module", model)

    def _module_key(self, module_path: str) -> str:
        if module_path in ("", None):
            parts = []
        else:
            parts = module_path.split(".")
        if not parts:
            parts = ["<root>"]
        if self.group_depth == 0:
            key = ".".join(parts)
        else:
            depth = min(len(parts), self.group_depth)
            key = ".".join(parts[:depth])
        return key or "<root>"

    def _ensure_module_info(self, model) -> None:
        if self._module_info is not None:
            return
        info: Dict[str, str] = {}
        for name, module in model.named_modules():
            key = name or "<root>"
            info[key] = module.__class__.__name__
        if "<root>" not in info:
            info["<root>"] = model.__class__.__name__
        self._module_info = info

    def _collect_stats(self, model) -> Optional[Dict[str, Dict[str, float]]]:
        is_main = self._is_main_process()
        module_sums: Dict[str, float] = {} if is_main else {}
        module_counts: Dict[str, int] = {} if is_main else {}
        module_params: Dict[str, list] = {} if is_main else {}
        for name, param in model.named_parameters():
            if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                continue
            stats = self._param_stats(param)
            if stats is None:
                continue
            total_sum, total_count = stats
            if not is_main:
                # Still execute for synchronization but skip accumulation on non-main ranks.
                continue
            module_path = name.rsplit(".", 1)[0] if "." in name else "<root>"
            module_sums[module_path] = module_sums.get(module_path, 0.0) + total_sum
            module_counts[module_path] = module_counts.get(module_path, 0) + total_count
            short_name = name[len(module_path) + 1 :] if module_path not in ("<root>", "") else name
            module_params.setdefault(module_path, []).append(
                {
                    "name": name,
                    "short_name": short_name,
                    "sum": total_sum,
                    "count": total_count,
                }
            )
        if not is_main or not module_sums:
            return None
        if self.group_depth == 0:
            return {
                module_path: {
                    "sum": module_sums[module_path],
                    "count": module_counts[module_path],
                    "modules": [module_path],
                    "parameters": sorted(
                        module_params.get(module_path, []),
                        key=lambda x: x["name"],
                    ),
                }
                for module_path in module_sums
            }

        grouped_sums: Dict[str, float] = {}
        grouped_counts: Dict[str, int] = {}
        grouped_members: Dict[str, list] = {}
        grouped_params: Dict[str, list] = {}
        for module_path, total_sum in module_sums.items():
            key = self._module_key(module_path)
            grouped_sums[key] = grouped_sums.get(key, 0.0) + total_sum
            grouped_counts[key] = grouped_counts.get(key, 0) + module_counts[module_path]
            members = grouped_members.setdefault(key, [])
            if module_path not in members:
                members.append(module_path)
            grouped_params.setdefault(key, []).extend(module_params.get(module_path, []))
        return {
            key: {
                "sum": grouped_sums[key],
                "count": grouped_counts[key],
                "modules": sorted(grouped_members.get(key, [])),
                "parameters": sorted(
                    grouped_params.get(key, []),
                    key=lambda x: x["name"],
                ),
            }
            for key in grouped_sums
        }

    def _log_stats(self, state, model):
        bare_model = self._unwrap(model)
        self._ensure_module_info(bare_model)
        stats = self._collect_stats(bare_model)
        if not stats or not self._is_main_process():
            return
        step = getattr(state, "global_step", 0) if state is not None else 0
        if self._last_logged_step == step:
            return
        self._last_logged_step = step
        rank0_print(f"[TrainableParamSum] step {step}")
        module_info = self._module_info or {}
        for key in sorted(stats.keys()):
            info = stats[key]
            modules = info.get("modules") or []
            if modules:
                resolved = []
                for module_path in modules:
                    label = module_path if module_path != "<root>" else "<root>"
                    cls_name = module_info.get(module_path, "")
                    if cls_name:
                        resolved.append(f"{label} ({cls_name})")
                    else:
                        resolved.append(label)
                module_line = ", ".join(resolved)
                rank0_print(f"  - {key}: sum={info['sum']:.6f}, params={info['count']} | modules: {module_line}")
            else:
                rank0_print(f"  - {key}: sum={info['sum']:.6f}, params={info['count']}")
            param_details = info.get("parameters") or []
            for param_info in param_details:
                pname = param_info.get("short_name") or param_info.get("name")
                psum = param_info.get("sum")
                pcount = param_info.get("count")
                rank0_print(f"    · {pname}: sum={psum:.6f}, params={pcount}")
        if self.log_to_wandb and has_active_wandb_run():
            payload = {f"trainable_sum/{key}": info["sum"] for key, info in stats.items()}
            payload.update({f"trainable_count/{key}": info["count"] for key, info in stats.items()})
            payload["train/global_step"] = step
            # Let W&B manage the internal step, mirroring the Hugging Face integration.
            wandb.log(payload)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        self._log_stats(state, model)

    def on_step_end(self, args, state, control, **kwargs):
        if state is None or state.global_step == 0 or (state.global_step % self.log_interval) != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        self._log_stats(state, model)


class TargetGradientAuditCallback(TrainerCallback):
    """Audit full trainable gradients, including DeepSpeed ZeRO partitions."""

    def __init__(self, interval: int = 0, output_path: Optional[str] = None, strict: bool = False):
        super().__init__()
        try:
            interval_value = int(interval)
        except (TypeError, ValueError):
            interval_value = 0
        self.interval = max(interval_value, 0)
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.strict = bool(strict)
        self._expected: Dict[str, int] = {}
        self._seen_gradient: Dict[str, bool] = {}
        self._seen_finite: Dict[str, bool] = {}
        self._seen_nonfinite: Dict[str, bool] = {}
        self._seen_nonzero: Dict[str, bool] = {}

    @staticmethod
    def _world_is_initialized() -> bool:
        return (
            hasattr(torch, "distributed")
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    @staticmethod
    def _is_main_process() -> bool:
        if TargetGradientAuditCallback._world_is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _unwrap(model):
        return getattr(model, "module", model)

    @staticmethod
    def _full_gradient(param: torch.nn.Parameter):
        if hasattr(param, "ds_id") or hasattr(param, "_hp_mapping"):
            try:
                from deepspeed.utils import safe_get_full_grad
            except Exception as exc:  # pragma: no cover - requires DeepSpeed runtime
                raise RuntimeError("DeepSpeed gradient audit helper is unavailable.") from exc
            return safe_get_full_grad(param)
        return getattr(param, "grad", None)

    def _write_jsonl(self, record: Dict) -> None:
        if self.output_path is None or not self._is_main_process():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("Gradient audit requires the Trainer model.")
        bare_model = self._unwrap(model)
        self._expected = {
            name: int(param.numel() or getattr(param, "ds_numel", 0))
            for name, param in bare_model.named_parameters()
            if isinstance(param, torch.nn.Parameter) and param.requires_grad
        }
        if not self._expected:
            raise RuntimeError("Gradient audit found no trainable parameters.")
        self._seen_gradient = {name: False for name in self._expected}
        self._seen_finite = {name: False for name in self._expected}
        self._seen_nonfinite = {name: False for name in self._expected}
        self._seen_nonzero = {name: False for name in self._expected}
        self._write_jsonl(
            {
                "event": "gradient_audit_start",
                "trainable_parameter_tensors": len(self._expected),
                "trainable_parameter_elements": sum(self._expected.values()),
            }
        )

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return
        optimizer_step = int(getattr(state, "global_step", 0)) + 1
        if optimizer_step % self.interval != 0:
            return
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("Gradient audit requires the Trainer model.")
        bare_model = self._unwrap(model)
        parameter_stats = []
        for name, param in bare_model.named_parameters():
            if name not in self._expected:
                continue
            grad = self._full_gradient(param)
            if grad is None:
                parameter_stats.append(
                    {
                        "name": name,
                        "elements": self._expected[name],
                        "gradient_present": False,
                        "finite": False,
                        "nonzero": False,
                        "norm": None,
                        "max_abs": None,
                    }
                )
                continue
            detached = grad.detach().float()
            finite = bool(torch.isfinite(detached).all().item())
            norm = float(torch.linalg.vector_norm(detached).item())
            max_abs = float(detached.abs().max().item()) if detached.numel() else 0.0
            nonzero = bool(norm > 0.0)
            self._seen_gradient[name] = True
            self._seen_finite[name] = self._seen_finite[name] or finite
            self._seen_nonfinite[name] = self._seen_nonfinite[name] or (not finite)
            self._seen_nonzero[name] = self._seen_nonzero[name] or (finite and nonzero)
            parameter_stats.append(
                {
                    "name": name,
                    "elements": self._expected[name],
                    "gradient_present": True,
                    "finite": finite,
                    "nonzero": nonzero,
                    "norm": norm,
                    "max_abs": max_abs,
                }
            )
            del detached, grad

        present_count = sum(item["gradient_present"] for item in parameter_stats)
        finite_count = sum(item["finite"] for item in parameter_stats)
        nonzero_count = sum(item["nonzero"] for item in parameter_stats)
        rank0_print(
            f"[TargetGradientAudit] optimizer_step {optimizer_step}: "
            f"present={present_count}/{len(parameter_stats)}, "
            f"finite={finite_count}/{len(parameter_stats)}, "
            f"nonzero={nonzero_count}/{len(parameter_stats)}"
        )
        self._write_jsonl(
            {
                "event": "gradient_audit_step",
                "optimizer_step": optimizer_step,
                "present_tensors": present_count,
                "finite_tensors": finite_count,
                "nonzero_tensors": nonzero_count,
                "parameters": parameter_stats,
            }
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return
        missing = sorted(name for name, seen in self._seen_gradient.items() if not seen)
        never_finite = sorted(name for name, seen in self._seen_finite.items() if not seen)
        nonfinite = sorted(name for name, seen in self._seen_nonfinite.items() if seen)
        never_nonzero = sorted(name for name, seen in self._seen_nonzero.items() if not seen)
        integrity_failures = bool(missing or never_finite or nonfinite)
        summary = {
            "event": "gradient_audit_end",
            "optimizer_steps": int(getattr(state, "global_step", 0)),
            "trainable_parameter_tensors": len(self._expected),
            "trainable_parameter_elements": sum(self._expected.values()),
            "missing_gradient": missing,
            "never_finite": never_finite,
            "nonfinite": nonfinite,
            "never_nonzero": never_nonzero,
            # A sparse audit can establish gradient presence and finiteness at the
            # sampled optimizer steps. It cannot establish that a parameter was
            # inactive for the entire run merely because the sampled value was
            # zero. Keep that observation as a diagnostic, not an integrity
            # failure. Method-specific effective-gradient checks are performed
            # separately (for example, the Stage 2F fusion gate check).
            "sampled_activity_status": "ALL_NONZERO" if not never_nonzero else "SPARSE_ZERO_OBSERVED",
            "status": "SUCCESS" if not integrity_failures else "FAILED",
        }
        self._write_jsonl(summary)
        rank0_print(
            "[TargetGradientAudit] final: "
            f"status={summary['status']}, missing={len(missing)}, "
            f"never_finite={len(never_finite)}, nonfinite={len(nonfinite)}, "
            f"sampled_zero={len(never_nonzero)}"
        )
        if self.strict and summary["status"] != "SUCCESS":
            raise RuntimeError("Strict trainable-gradient audit failed: " + json.dumps(summary, sort_keys=True))

class SegmentationPreviewCallback(TrainerCallback):
    def __init__(self, interval: int = 0, save_dir: Optional[str] = None):
        super().__init__()
        self.interval = max(int(interval), 0)
        self.save_dir = save_dir
        self._dir_ready = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return
        if state.global_step == 0 or (state.global_step % self.interval) != 0:
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        mm = getattr(model, "module", model)
        preview = getattr(mm, "_last_segmentation_preview", None)
        if preview is None:
            return

        if not isinstance(preview, dict):
            return

        probs = preview.get("probs")
        masks = preview.get("masks")
        if probs is None or masks is None:
            return

        if hasattr(torch, "distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = torch.distributed.get_rank() == 0
        else:
            is_main = True

        with torch.no_grad():
            preds = (probs > 0.5).float()
            targets = (masks > 0.5).float()
            intersection = (preds * targets).sum().item()
            pred_sum = preds.sum().item()
            target_sum = targets.sum().item()
            eps = 1e-6
            dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
            pred_ratio = pred_sum / (preds.numel() + eps)
            target_ratio = target_sum / (targets.numel() + eps)

        wandb_ready = is_main and has_active_wandb_run() and hasattr(wandb, "Image")
        prob_np = pred_np = mask_np = None
        if is_main and (self.save_dir or wandb_ready):
            prob_np = probs.detach().float().cpu().squeeze().numpy()
            pred_np = preds.detach().float().cpu().squeeze().numpy()
            mask_np = targets.detach().float().cpu().squeeze().numpy()

        if is_main and self.save_dir:
            if not self._dir_ready:
                os.makedirs(self.save_dir, exist_ok=True)
                self._dir_ready = True
            step_dir = os.path.join(self.save_dir, f"step_{state.global_step:06d}")
            os.makedirs(step_dir, exist_ok=True)

            prob_img = Image.fromarray((np.clip(prob_np, 0.0, 1.0) * 255).astype(np.uint8))
            pred_img = Image.fromarray((np.clip(pred_np, 0.0, 1.0) * 255).astype(np.uint8))
            mask_img = Image.fromarray((np.clip(mask_np, 0.0, 1.0) * 255).astype(np.uint8))

            prob_img.save(os.path.join(step_dir, "prob.png"))
            pred_img.save(os.path.join(step_dir, "pred_mask.png"))
            mask_img.save(os.path.join(step_dir, "gt_mask.png"))

            # Input previews are skipped to avoid problematic reshaping.
        if wandb_ready and prob_np is not None:
            log_payload = {"train/dice_preview": dice, "train/pred_positive": pred_ratio, "train/gt_positive": target_ratio}
            images = [
                wandb.Image(prob_np, caption="prob"),
                wandb.Image(pred_np, caption="pred"),
                wandb.Image(mask_np, caption="gt"),
            ]
            log_payload["train/seg_preview"] = images
            wandb.log(log_payload, step=state.global_step)
            rank0_print(
                f"[SegPreview] step {state.global_step}: dice={dice:.4f}, pred_positive={pred_ratio:.4f}, gt_positive={target_ratio:.4f}"
            )

        mm._last_segmentation_preview = None

class SegmentationGradNormCallback(TrainerCallback):
    def __init__(self, log_interval: int = 500, log_to_wandb: bool = True):
        super().__init__()
        try:
            interval_val = int(log_interval)
        except (TypeError, ValueError):
            interval_val = 500
        self.log_interval = max(interval_val, 1)
        self.log_to_wandb = bool(log_to_wandb) and wandb_is_available()
        self._last_logged_step: Optional[int] = None
        self._pending_norm_sq: Dict[str, float] = {}

    @staticmethod
    def _world_is_initialized() -> bool:
        return (
            hasattr(torch, "distributed")
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    @staticmethod
    def _rank() -> int:
        if SegmentationGradNormCallback._world_is_initialized():
            return torch.distributed.get_rank()
        return 0

    @staticmethod
    def _is_main_process() -> bool:
        return SegmentationGradNormCallback._rank() == 0

    @staticmethod
    def _unwrap(model):
        return getattr(model, "module", model)

    @staticmethod
    def _reduce_device_from_param(param: torch.nn.Parameter) -> torch.device:
        device = getattr(param, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            return device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @staticmethod
    def _plain_grad_square_sum(param: torch.nn.Parameter) -> float:
        grad = getattr(param, "grad", None)
        if grad is None:
            return 0.0
        with torch.no_grad():
            g = grad.detach().float()
            return float(torch.sum(g * g).item())

    @staticmethod
    def _ds_grad_square_sum(param: torch.nn.Parameter) -> float:
        if not hasattr(param, "ds_status"):
            return 0.0
        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        except Exception:
            return 0.0

        status = getattr(param, "ds_status", None)
        reduce_device = SegmentationGradNormCallback._reduce_device_from_param(param)
        buffer = torch.zeros(1, device=reduce_device, dtype=torch.float64)
        gather_needed = status != ZeroParamStatus.AVAILABLE
        # Gather grad partitions on rank 0 to measure the norm.
        with deepspeed.zero.GatheredParameters([param], modifier_rank=0, enabled=gather_needed):
            if SegmentationGradNormCallback._is_main_process():
                grad = getattr(param, "grad", None)
                if grad is not None:
                    g = grad.detach().float()
                    buffer[0] = torch.sum(g * g).to(dtype=torch.float64)
        if SegmentationGradNormCallback._world_is_initialized():
            torch.distributed.broadcast(buffer, src=0)
        return float(buffer.cpu().item())

    @staticmethod
    def _grad_square_sum(param: torch.nn.Parameter) -> float:
        plain = SegmentationGradNormCallback._plain_grad_square_sum(param)
        if plain > 0.0:
            return plain
        return SegmentationGradNormCallback._ds_grad_square_sum(param)

    def _accumulate_grad_norms(self, model) -> None:
        bare_model = self._unwrap(model)
        for name, param in bare_model.named_parameters():
            if not name.startswith("segmentation_head.blocks."):
                continue
            block_name = ".".join(name.split(".", 3)[:3])
            norm_sq = self._grad_square_sum(param)
            if norm_sq == 0.0:
                continue
            self._pending_norm_sq[block_name] = self._pending_norm_sq.get(block_name, 0.0) + norm_sq

    def on_backward_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        self._accumulate_grad_norms(model)

    def on_step_end(self, args, state, control, **kwargs):
        if state is None:
            self._pending_norm_sq.clear()
            return
        step = getattr(state, "global_step", 0)
        if step == 0:
            self._pending_norm_sq.clear()
            return
        if step == self._last_logged_step:
            self._pending_norm_sq.clear()
            return
        if step % self.log_interval != 0:
            self._pending_norm_sq.clear()
            return
        if not self._is_main_process():
            self._pending_norm_sq.clear()
            return
        norms = {key: value ** 0.5 for key, value in self._pending_norm_sq.items()}
        self._pending_norm_sq.clear()
        if not norms:
            return
        self._last_logged_step = step
        rank0_print(f"[SegGradNorm] step {step}")
        for key in sorted(norms.keys()):
            rank0_print(f"  - {key}: grad_norm={norms[key]:.6e}")
        if self.log_to_wandb and has_active_wandb_run():
            payload = {f"grad_norm/{key}": value for key, value in norms.items()}
            payload["train/global_step"] = step
            wandb.log(payload)

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.trainable_checkpoint import save_trainable_checkpoint
from qwenvl.train.anomaly_evidence_trainer import (
    AnomalyEvidenceTrainer,
    Stage2HStepTimingCallback,
)
from transformers import AutoImageProcessor, AutoTokenizer, Trainer

local_rank = None

def is_main_process() -> bool:
    if hasattr(torch, "distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return local_rank in (None, -1, 0)


def rank0_print(*args):
    if is_main_process():
        print(*args)


def set_requires_grad(module, flag: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = flag


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    diff_only_mode = bool(getattr(model_args, "diff_only_mode", False))
    anomaly_query_mode = str(getattr(model_args, "anomaly_query_mode", "single") or "single").strip().lower()
    if anomaly_query_mode not in {"single", "multiscale"}:
        raise ValueError(
            f"Unsupported anomaly_query_mode: {anomaly_query_mode!r}; expected 'single' or 'multiscale'."
        )
    if anomaly_query_mode == "multiscale" and int(model_args.num_pooling_size) != 4:
        raise ValueError("FB-MAQ multiscale mode requires num_pooling_size=4 (16 Ano tokens).")
    if hasattr(model, "config"):
        model.config.diff_only_mode = diff_only_mode
        if hasattr(model.config, "vision_config"):
            model.config.vision_config.diff_only_mode = diff_only_mode
            model.config.vision_config.anomaly_query_mode = anomaly_query_mode
    visual_module = getattr(model, "visual", None)
    if visual_module is None and hasattr(model, "model"):
        visual_module = getattr(model.model, "visual", None)
    if visual_module is not None and hasattr(visual_module, "set_diff_only_mode"):
        visual_module.set_diff_only_mode(diff_only_mode)
    if visual_module is not None and hasattr(visual_module, "set_num_pooling_size"):
        visual_module.set_num_pooling_size(model_args.num_pooling_size)
    if visual_module is not None and hasattr(visual_module, "set_anomaly_query_mode"):
        visual_module.set_anomaly_query_mode(anomaly_query_mode)

    def _set_module_grads(module, flag: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    set_requires_grad(model.visual, model_args.tune_mm_vision)
    set_requires_grad(model.segmentation_head, model_args.tune_mm_vision_decoder)

    fusion_modules = [
        getattr(model, "segment_query_proj", None),
        getattr(model, "segment_key_proj", None),
        getattr(model, "segment_value_proj", None),
        getattr(model, "segment_out_proj", None),
        getattr(model, "segment_cross_attn", None),
        getattr(model, "segment_norm", None),
    ]
    for module in fusion_modules:
        _set_module_grads(module, model_args.tune_mm_vision_decoder)

    set_requires_grad(model.visual.merger, model_args.tune_mm_mlp)
    set_requires_grad(model.model, model_args.tune_mm_llm)
    model.lm_head.requires_grad = bool(model_args.tune_mm_llm)
    model.lm_head.weight.requires_grad = bool(model_args.tune_mm_llm)

    if model_args.tune_mm_vpt:
        if model_args.reset_vpt or model.config.vision_config.vpt_tokens_number != model_args.vpt_tokens_number:
            model.config.vision_config.vpt_tokens_number = model_args.vpt_tokens_number
            model.visual.re_init_vpt(model_args.vpt_tokens_number)
        set_requires_grad(model.visual.deep_prompt_embeddings, True)
        model.visual.set_vpt_dropout(0.1)
        model.visual.prompt_dropout.requires_grad = True
    else:
        set_requires_grad(model.visual.deep_prompt_embeddings, False)

    if model_args.tune_mm_anomaly:
        model.config.vision_config.num_pooling_size = model_args.num_pooling_size
        model.config.vision_config.anomaly_query_mode = anomaly_query_mode
        if (
            model_args.reset_anomaly
            or model_args.num_pooling_size != model.visual.num_pooling_size
            or anomaly_query_mode != getattr(model.visual, "anomaly_query_mode", "single")
        ):
            model.visual.re_init_anomaly(model_args.num_pooling_size, anomaly_query_mode)
        set_requires_grad(model.visual.anomaly_qformer, True)
    else:
        set_requires_grad(model.visual.anomaly_qformer, False)

    if model_args.tune_mm_diff:
        model.visual.anomaly_qformer.re_init_diff_q_former()
        set_requires_grad(model.visual.anomaly_qformer.diff_q_former, True)
        set_requires_grad(model.visual.anomaly_qformer.diff_post_ffn, True)
    else:
        set_requires_grad(model.visual.anomaly_qformer.diff_q_former, False)
        set_requires_grad(model.visual.anomaly_qformer.diff_post_ffn, False)



def _resolve_checkpoint_files(checkpoint_path: str):
    path = Path(checkpoint_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        return []

    index_candidates = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    for name in index_candidates:
        index_path = path / name
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            weight_map = data.get("weight_map", {})
            files = sorted({str(path / fname) for fname in weight_map.values()})
            return [fname for fname in files if Path(fname).exists()]

    for name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        file_path = path / name
        if file_path.exists():
            return [str(file_path)]

    shard_files = sorted(path.glob("model-*.safetensors"))
    if shard_files:
        return [str(p) for p in shard_files]
    shard_files = sorted(path.glob("pytorch_model-*.bin"))
    if shard_files:
        return [str(p) for p in shard_files]
    return []


def _extract_state_dict(raw_state):
    if isinstance(raw_state, dict):
        for key in ("state_dict", "model", "module"):
            candidate = raw_state.get(key)
            if isinstance(candidate, dict) and candidate:
                if all(isinstance(k, str) for k in candidate.keys()):
                    return candidate
    return raw_state


def _normalize_segment_key(key: str):
    if key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("segmentation_head."):
        return key[len("segmentation_head."):]
    if key.startswith("model.segmentation_head."):
        return key[len("model.segmentation_head."):]
    return key


def load_segmentation_head_from_checkpoint(model, checkpoint_path: str, print_fn=print):
    if not checkpoint_path:
        return
    checkpoint_files = _resolve_checkpoint_files(checkpoint_path)
    if not checkpoint_files:
        print_fn(f"[seg-head] checkpoint not found: {checkpoint_path}")
        return

    target_state = model.segmentation_head.state_dict()
    loaded_state = {}
    skipped_shape = 0
    for ckpt_file in checkpoint_files:
        try:
            state = load_state_dict(ckpt_file, map_location="cpu", weights_only=True)
        except TypeError:
            state = load_state_dict(ckpt_file)
        state = _extract_state_dict(state)
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if not isinstance(key, str):
                continue
            norm_key = _normalize_segment_key(key)
            if norm_key not in target_state:
                continue
            try:
                if tuple(value.shape) != tuple(target_state[norm_key].shape):
                    skipped_shape += 1
                    continue
            except Exception:
                pass
            loaded_state[norm_key] = value
        del state
        gc.collect()

    if not loaded_state:
        print_fn(f"[seg-head] no matching segmentation_head keys found in {checkpoint_path}")
        return

    incompatible = model.segmentation_head.load_state_dict(loaded_state, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print_fn(
        f"[seg-head] loaded {len(loaded_state)}/{len(target_state)} keys from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatch={skipped_shape})."
    )

def _resolve_param_shape(param: torch.nn.Parameter):
    ds_shape = getattr(param, "ds_shape", None)
    if ds_shape is not None:
        try:
            resolved = tuple(int(x) for x in ds_shape)
            if all(dim >= 0 for dim in resolved):
                return resolved
        except Exception:
            pass
    shape = tuple(param.shape)
    if shape == (0,) and hasattr(param, "ds_numel"):
        try:
            ds_numel = int(getattr(param, "ds_numel"))
            if ds_numel > 0:
                return (ds_numel,)
        except Exception:
            pass
    return shape


def print_learnable_tree(module, prefix=""):
    """Print trainable parameter names and shapes."""
    rank0_print("#####################")
    rank0_print("Trainable Parameters")
    rank0_print("#####################")
    for name, param in module.named_parameters():
        if param.requires_grad:
            shape = _resolve_param_shape(param)
            rank0_print(f"{prefix}├─ {name} {shape}")

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Build the dataset only after all ranks share the same Python, NumPy, and
    # PyTorch seed. The dataset constructor samples and shuffles records before
    # Trainer is instantiated, so relying on Trainer's later seed setup can
    # produce rank-dependent dataset order in distributed runs.
    transformers.set_seed(training_args.seed)

    diff_only_mode = bool(getattr(model_args, "diff_only_mode", False))
    setattr(data_args, "diff_only_mode", diff_only_mode)
    if diff_only_mode:
        data_args.use_diff_token = True

    if getattr(model_args, "train_type", "default") != "segment":
        data_args.load_masks = False
    if getattr(model_args, "train_type", "default") == "anomaly_evidence":
        data_args.require_anomaly_labels = True
        data_args.strict_sample_loading = True
        if data_args.data_flatten or data_args.data_packing:
            raise ValueError("Stage 2H anomaly-evidence training prohibits flattening and packing.")
        if training_args.remove_unused_columns:
            raise ValueError("Stage 2H requires --remove_unused_columns False.")
        if not math.isfinite(float(model_args.evidence_loss_weight)) or model_args.evidence_loss_weight < 0:
            raise ValueError("Stage 2H evidence_loss_weight must be finite and non-negative.")
        if str(model_args.anomaly_query_mode).strip().lower() != "single":
            raise ValueError("Stage 2H requires anomaly_query_mode=single.")
        if int(model_args.num_pooling_size) != 4:
            raise ValueError("Stage 2H requires num_pooling_size=4 (16 Ano tokens).")
        if int(model_args.vpt_tokens_number) != 10:
            raise ValueError("Stage 2H requires the official B0 vpt_tokens_number=10.")
        if not model_args.tune_mm_vpt or not model_args.tune_mm_anomaly:
            raise ValueError("Stage 2H requires exactly the VPT and AnomalyQformer trainable scopes.")
        forbidden_tuning = {
            "tune_mm_llm": model_args.tune_mm_llm,
            "tune_mm_mlp": model_args.tune_mm_mlp,
            "tune_mm_vision": model_args.tune_mm_vision,
            "tune_mm_vision_decoder": model_args.tune_mm_vision_decoder,
            "tune_mm_diff": model_args.tune_mm_diff,
            "diff_only_mode": model_args.diff_only_mode,
        }
        enabled_forbidden = sorted(name for name, enabled in forbidden_tuning.items() if enabled)
        if enabled_forbidden:
            raise ValueError(f"Stage 2H found forbidden trainable/method flags: {enabled_forbidden}")
        if os.environ.get("MEDIC_AD_STAGE2H_STRICT_SCHEMA") == "1":
            strict_requirements = {
                "max_steps": (int(training_args.max_steps), 2),
                "per_device_train_batch_size": (int(training_args.per_device_train_batch_size), 1),
                "gradient_accumulation_steps": (int(training_args.gradient_accumulation_steps), 1),
                "seed": (int(training_args.seed), 42),
                "data_seed": (int(training_args.data_seed or training_args.seed), 42),
                "gradient_checkpointing": (bool(training_args.gradient_checkpointing), True),
                "bf16": (bool(training_args.bf16), True),
                "dataset_use": (str(data_args.dataset_use), "medic_ad_stage2h_train"),
            }
            mismatches = {
                name: {"observed": observed, "expected": expected}
                for name, (observed, expected) in strict_requirements.items()
                if observed != expected
            }
            if mismatches:
                raise ValueError(f"Stage 2H-E strict run configuration mismatch: {mismatches}")

    attn_implementation = "flash_attention_2"

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    target_config = getattr(model, 'model', None)
    target_config = getattr(target_config, 'config', model.config)

    target_config.segment_use_llm_query = model_args.segment_use_llm_query
    target_config.segment_cross_attn_heads = model_args.segment_cross_attn_heads
    target_config.segment_cross_attn_dropout = model_args.segment_cross_attn_dropout
    target_config.segment_upsample_factor = model_args.segment_upsample_factor

    model.config.segment_use_llm_query = model_args.segment_use_llm_query
    model.config.segment_cross_attn_heads = model_args.segment_cross_attn_heads
    model.config.segment_cross_attn_dropout = model_args.segment_cross_attn_dropout
    model.config.seg_decoder_type = model_args.seg_decoder_type
    model.config.segment_upsample_factor = model_args.segment_upsample_factor
    rank0_print(f"[Config] segmentation decoder: {model_args.seg_decoder_type}")
    if model_args.seg_decoder_type == "DETR":
        model.re_init_detr()

    needs_reinit = False
    if model_args.segment_embed_dim is not None:
        target_config.segment_embed_dim = model_args.segment_embed_dim
        model.config.segment_embed_dim = model_args.segment_embed_dim
        needs_reinit = True
    if model_args.segment_use_llm_query:
        needs_reinit = True
    if model_args.segment_cross_attn_heads is not None:
        needs_reinit = True
    if model_args.segment_cross_attn_dropout:
        needs_reinit = True
    if model_args.segment_upsample_factor is not None:
        needs_reinit = True

    if needs_reinit and model_args.reset_vision_decoder:
        model.config.segment_upsample_factor = model_args.segment_upsample_factor
        model.segment_upsample_factor = model_args.segment_upsample_factor
        model.re_init_convnext(upsample_factor=model_args.segment_upsample_factor)
    else:
        target_flag = bool(model_args.segment_use_llm_query)
        current_flag = bool(getattr(model, "segment_use_llm_query", False))
        have_modules = getattr(model, "segment_query_proj", None) is not None
        if (target_flag != current_flag) or (target_flag and not have_modules) or (not target_flag and have_modules):
            # Refresh LLM-guided fusion modules when toggling without a full decoder reset.
            model.segment_use_llm_query = target_flag
            model.segment_cross_attn_heads = model_args.segment_cross_attn_heads
            model.segment_cross_attn_dropout = float(model_args.segment_cross_attn_dropout or 0.0)
            model._init_llm_guided_modules()

    seg_head_ckpt = getattr(model_args, "segment_head_checkpoint", None)
    if isinstance(seg_head_ckpt, str) and seg_head_ckpt.strip().lower() in {"none", "null"}:
        seg_head_ckpt = None
    if seg_head_ckpt:
        load_segmentation_head_from_checkpoint(model, seg_head_ckpt, print_fn=rank0_print)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)
    if model_args.train_type == "anomaly_evidence":
        # `re_init_anomaly` creates a fresh module in the framework default
        # dtype/device. Place both newly initialized Stage 1 modules exactly
        # with the pinned vision tower before capturing step-0 state or
        # constructing the DeepSpeed engine.
        stage2h_reference = next(model.visual.patch_embed.parameters())
        model.visual.deep_prompt_embeddings.to(
            device=stage2h_reference.device,
            dtype=stage2h_reference.dtype,
        )
        model.visual.anomaly_qformer.to(
            device=stage2h_reference.device,
            dtype=stage2h_reference.dtype,
        )
    stage2h_initial_state_output = os.environ.get("MEDIC_AD_STAGE2H_INITIAL_STATE_OUTPUT")
    if stage2h_initial_state_output:
        if model_args.train_type != "anomaly_evidence":
            raise ValueError("Stage 2H initial-state capture requires train_type=anomaly_evidence.")
        from reproduction.stage2h.state_audit import save_initial_trainable_state

        initial_state = save_initial_trainable_state(model, stage2h_initial_state_output)
        rank0_print(json.dumps({"stage2h_initial_state": initial_state}, indent=2, sort_keys=True))
    trainable_scope_output = os.environ.get("MEDIC_AD_TRAINABLE_SCOPE_OUTPUT")
    if trainable_scope_output:
        from reproduction.stage2f.adapter_schema import audit_trainable_scope

        scope_path = Path(trainable_scope_output).expanduser()
        if scope_path.exists():
            raise FileExistsError(f"Refusing to overwrite trainable-scope evidence: {scope_path}")
        scope_result = audit_trainable_scope(model, model_args.anomaly_query_mode)
        scope_result["source_base_commit"] = "580abb4"
        scope_path.parent.mkdir(parents=True, exist_ok=True)
        scope_path.write_text(
            json.dumps(scope_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rank0_print(json.dumps({"trainable_scope": scope_result}, indent=2, sort_keys=True))
    current_pooling = getattr(getattr(model, "visual", None), "num_pooling_size", None)
    config_pooling = getattr(getattr(model.config, "vision_config", None), "num_pooling_size", None)
    rank0_print(
        f"[Config] visual num_pooling_size={current_pooling} "
        f"(config={config_pooling}, arg={model_args.num_pooling_size})"
    )
    rank0_print(
        f"[Config] anomaly_query_mode={getattr(model.visual, 'anomaly_query_mode', None)} "
        f"(config={getattr(model.config.vision_config, 'anomaly_query_mode', None)}, "
        f"arg={model_args.anomaly_query_mode})"
    )
    print_learnable_tree(model)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if getattr(model_args, "train_type", "default") == "diff":
        data_args.use_diff_token = True

    use_anomaly_token_flag = (
        (model_args.tune_mm_anomaly or model_args.train_type == "segment")
        or (data_args.use_diff_token and not diff_only_mode)
    )

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(
            tokenizer=tokenizer,
            data_args=data_args,
            use_anomaly_token=use_anomaly_token_flag,
            num_pooling_size=model_args.num_pooling_size
        )
    else:
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            use_anomaly_token=use_anomaly_token_flag,
            num_pooling_size=model_args.num_pooling_size
        )

    monitor_interval = getattr(model_args, "trainable_monitor_interval", None)
    try:
        monitor_interval_val = int(monitor_interval) if monitor_interval is not None else int(model_args.log_interval)
    except (TypeError, ValueError):
        monitor_interval_val = int(model_args.log_interval)

    monitor_depth = getattr(model_args, "trainable_monitor_depth", 0)
    try:
        monitor_depth_val = int(monitor_depth) if monitor_depth is not None else 0
    except (TypeError, ValueError):
        monitor_depth_val = 0

    monitor_to_wandb = getattr(model_args, "trainable_monitor_to_wandb", True)

    callbacks = [
        TrainableParamSumCallback(
            log_interval=monitor_interval_val,
            group_depth=monitor_depth_val,
            log_to_wandb=bool(monitor_to_wandb),
        ),
        SegmentationGradNormCallback(
            log_interval=monitor_interval_val,
            log_to_wandb=bool(monitor_to_wandb),
        ),
    ]
    if model_args.stage2h_step_timing_output:
        callbacks.append(Stage2HStepTimingCallback(model_args.stage2h_step_timing_output))
    activation_diagnostic_output = os.environ.get(
        "MEDIC_AD_ACTIVATION_DIAGNOSTIC_OUTPUT"
    )
    if activation_diagnostic_output:
        from reproduction.stage2f.training_diagnostics import (
            Stage2FActivationDiagnosticCallback,
        )

        callbacks.append(
            Stage2FActivationDiagnosticCallback(
                model=model,
                output_path=activation_diagnostic_output,
                mode=model_args.anomaly_query_mode,
                expected_steps=training_args.max_steps,
            )
        )
    if model_args.diagnostic_train_engine_eval_only:
        if not training_args.eval_on_start:
            raise ValueError("diagnostic_train_engine_eval_only requires eval_on_start=True.")
        callbacks.append(StopAfterInitialEvaluationCallback())
    if int(getattr(model_args, "gradient_audit_interval", 0) or 0) > 0:
        callbacks.append(
            TargetGradientAuditCallback(
                interval=model_args.gradient_audit_interval,
                output_path=model_args.gradient_audit_output,
                strict=model_args.gradient_audit_strict,
            )
        )
    if getattr(model_args, "segment_vis_interval", 0) > 0 and model_args.train_type == "segment":
        save_dir = model_args.segment_vis_save_dir
        if save_dir:
            save_dir = os.path.abspath(save_dir)
        callbacks.append(
            SegmentationPreviewCallback(
                interval=model_args.segment_vis_interval,
                save_dir=save_dir,
            )
        )

    class SegmentationTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            inputs["tune_mode"] = "segment"
            inputs["segment_alpha"] = model_args.segment_alpha
            outputs = model(**inputs)
            metrics = {}
            if hasattr(outputs, "seg_loss") and outputs.seg_loss is not None:
                seg_value = outputs.seg_loss.item() if hasattr(outputs.seg_loss, "item") else float(outputs.seg_loss)
                metrics["train_seg_loss"] = seg_value
            if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
                lm_value = outputs.lm_loss.item() if hasattr(outputs.lm_loss, "item") else float(outputs.lm_loss)
                metrics["train_lm_loss"] = lm_value
            if metrics:
                self.log(metrics)

            loss = outputs.lm_loss * (1 - model_args.segment_alpha) + outputs.seg_loss * model_args.segment_alpha
            return (loss, outputs) if return_outputs else loss
    
    class DefaultTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            inputs["tune_mode"] = "default"
            outputs = model(**inputs)
            loss = outputs.loss
            loss_trace_output = getattr(model_args, "loss_trace_output", None)
            if loss_trace_output and model.training and self.is_world_process_zero():
                output_path = Path(loss_trace_output).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "global_step_before_update": int(self.state.global_step),
                    "loss": float(loss.detach().float().item()),
                }
                with output_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            return (loss, outputs) if return_outputs else loss

    class DiffTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            inputs["tune_mode"] = "diff"
            inputs["diff_mode"] = True
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    if model_args.train_type == "default":
        trainer = DefaultTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            callbacks=callbacks,
            **data_module,
        )
    elif model_args.train_type == "diff":
        trainer = DiffTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            callbacks=callbacks,
            **data_module,
        )
    elif model_args.train_type == "segment":
        trainer = SegmentationTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            callbacks=callbacks,
            **data_module,
        )
    elif model_args.train_type == "anomaly_evidence":
        trainer = AnomalyEvidenceTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            callbacks=callbacks,
            evidence_loss_weight=model_args.evidence_loss_weight,
            evidence_trace_output=model_args.evidence_trace_output,
            **data_module,
        )
    else:
        raise ValueError(f"Unsupported train_type: {model_args.train_type}")

    if model_args.diagnostic_eval_only:
        if model_args.diagnostic_train_engine_eval_only:
            raise ValueError("Only one diagnostic evaluation mode may be enabled.")
        if trainer.eval_dataset is None:
            raise ValueError("diagnostic_eval_only requires a separately configured eval_dataset.")
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            result = {
                "status": "SUCCESS",
                "mode": "diagnostic_eval_only",
                "eval_samples": len(trainer.eval_dataset),
                "metrics": metrics,
            }
            rank0_print(json.dumps(result, indent=2, sort_keys=True))
            if model_args.diagnostic_eval_output:
                output_path = Path(model_args.diagnostic_eval_output).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        model.config.use_cache = True
        return

    try:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except DiagnosticEvaluationComplete as completed:
        if not model_args.diagnostic_train_engine_eval_only:
            raise
        if trainer.state.global_step != 0:
            raise RuntimeError(
                "Training-engine diagnostic performed an unexpected optimizer step: "
                f"global_step={trainer.state.global_step}"
            )
        if trainer.is_world_process_zero():
            result = {
                "status": "SUCCESS",
                "mode": "diagnostic_train_engine_eval_only",
                "eval_samples": len(trainer.eval_dataset),
                "global_step": int(trainer.state.global_step),
                "optimizer_step_performed": False,
                "metrics": completed.metrics,
            }
            rank0_print(json.dumps(result, indent=2, sort_keys=True))
            if model_args.diagnostic_eval_output:
                output_path = Path(model_args.diagnostic_eval_output).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        model.config.use_cache = True
        return
    trainer.save_state()
    stage2h_train_memory_output = os.environ.get("MEDIC_AD_STAGE2H_TRAIN_MEMORY_OUTPUT")
    if stage2h_train_memory_output:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 2H training memory audit requires CUDA.")
        torch.cuda.synchronize()
        memory_path = Path(stage2h_train_memory_output).expanduser()
        if memory_path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 2H training memory evidence: {memory_path}")
        memory_record = {
            "status": "SUCCESS",
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            "global_step": int(trainer.state.global_step),
        }
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(memory_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rank0_print(json.dumps({"stage2h_training_memory": memory_record}, indent=2, sort_keys=True))
    data_args.image_processor.save_pretrained(training_args.output_dir)

    if training_args.trainable_state_output:
        normalized_query_mode = str(model_args.anomaly_query_mode or "single").strip().lower()
        stage2f_repeat = os.environ.get("MEDIC_AD_REPEAT")
        stage2f_environment_metadata = {
            "base_model_revision": os.environ.get("MEDIC_AD_BASE_MODEL_REVISION"),
            "base_checkpoint_fingerprint": os.environ.get("MEDIC_AD_BASE_CHECKPOINT_FINGERPRINT"),
            "dataset_manifest_sha256": os.environ.get("MEDIC_AD_DATASET_MANIFEST_SHA256"),
            "implementation_source_fingerprint": os.environ.get("MEDIC_AD_IMPLEMENTATION_SOURCE_FINGERPRINT"),
            "run_id": os.environ.get("MEDIC_AD_RUN_ID"),
            "repeat": int(stage2f_repeat) if stage2f_repeat else None,
            "stage": os.environ.get("MEDIC_AD_STAGE"),
        }
        if os.environ.get("MEDIC_AD_STAGE2F_STRICT_SCHEMA") == "1":
            missing_environment_metadata = sorted(
                key for key, value in stage2f_environment_metadata.items() if not value
            )
            if missing_environment_metadata:
                raise RuntimeError(
                    "Stage 2F strict adapter export is missing environment metadata: "
                    f"{missing_environment_metadata}"
                )
        stage2h_environment_metadata = {}
        if model_args.train_type == "anomaly_evidence":
            stage2h_environment_metadata = {
                "protocol_version": os.environ.get("MEDIC_AD_PROTOCOL_VERSION", "2.1"),
                "training_manifest_sha256": os.environ.get(
                    "MEDIC_AD_TRAINING_MANIFEST_SHA256"
                ),
                "evaluation_manifest_sha256": os.environ.get(
                    "MEDIC_AD_EVALUATION_MANIFEST_SHA256"
                ),
                "claim_scope": "image_level_internal_engineering_only",
            }
            if os.environ.get("MEDIC_AD_STAGE2H_STRICT_SCHEMA") == "1":
                required_stage2h_metadata = {
                    key: stage2f_environment_metadata[key]
                    for key in (
                        "base_model_revision",
                        "base_checkpoint_fingerprint",
                        "dataset_manifest_sha256",
                        "implementation_source_fingerprint",
                        "run_id",
                        "stage",
                    )
                }
                required_stage2h_metadata.update(stage2h_environment_metadata)
                missing_stage2h_metadata = sorted(
                    key for key, value in required_stage2h_metadata.items() if value is None or value == ""
                )
                if missing_stage2h_metadata:
                    raise RuntimeError(
                        "Stage 2H strict adapter export is missing environment metadata: "
                        f"{missing_stage2h_metadata}"
                    )
                if stage2f_environment_metadata["stage"] != "2H-E":
                    raise RuntimeError("Stage 2H strict adapter export requires MEDIC_AD_STAGE=2H-E.")
        stage2h_method_id = None
        if model_args.train_type == "anomaly_evidence":
            stage2h_method_id = (
                "medic-ad-b0-as"
                if float(model_args.evidence_loss_weight) == 0.0
                else "lad-mil-v2"
            )
        checkpoint_manifest = save_trainable_checkpoint(
            trainer=trainer,
            output_path=training_args.trainable_state_output,
            metadata={
                "base_model": model_args.model_name_or_path,
                "global_step": int(trainer.state.global_step),
                "seed": int(training_args.seed),
                "data_seed": int(training_args.data_seed or training_args.seed),
                "train_type": model_args.train_type,
                "vpt_tokens_number": int(model_args.vpt_tokens_number),
                "num_pooling_size": int(model_args.num_pooling_size),
                "anomaly_query_mode": normalized_query_mode,
                "output_token_count": int(model_args.num_pooling_size) ** 2,
                "method_id": stage2h_method_id or (
                    "fb-maq-stage2f" if normalized_query_mode == "multiscale" else "medic-ad-b0"
                ),
                "gate_type": "spatial_conv1x1" if normalized_query_mode == "multiscale" else "none",
                "gate_initial_lambda": 0.9 if normalized_query_mode == "multiscale" else None,
                "evidence_loss_weight": (
                    float(model_args.evidence_loss_weight)
                    if model_args.train_type == "anomaly_evidence"
                    else None
                ),
                "evidence_definition": (
                    "pre_gate_sigmoid_difference"
                    if model_args.train_type == "anomaly_evidence"
                    else None
                ),
                "evidence_margin": 0.1 if model_args.train_type == "anomaly_evidence" else None,
                "evidence_topk_fraction": 0.01 if model_args.train_type == "anomaly_evidence" else None,
                "evidence_topk_k": 11 if model_args.train_type == "anomaly_evidence" else None,
                **stage2h_environment_metadata,
                **stage2f_environment_metadata,
                "source_base_commit": "580abb4",
                "tune_mm_vpt": bool(model_args.tune_mm_vpt),
                "tune_mm_anomaly": bool(model_args.tune_mm_anomaly),
                "full_determinism": bool(training_args.full_determinism),
            },
        )
        rank0_print(json.dumps({"trainable_checkpoint": checkpoint_manifest}, indent=2, sort_keys=True))

    model.config.use_cache = True

    if training_args.skip_final_model_save:
        rank0_print("[Save] Skipping final full-model serialization for this diagnostic run.")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
