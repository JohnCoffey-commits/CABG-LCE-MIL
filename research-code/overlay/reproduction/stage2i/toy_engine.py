"""Small deterministic CPU engine used only for Gate D2 state tests."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from reproduction.stage2i.cabg_math import (
    BiasCorrectedEMA,
    aggregate_class_balanced_gradients,
    aggregate_lm_gradients,
    combine_and_clip_gradients,
    compute_budget,
    per_image_class_balanced_rms,
)
from reproduction.stage2i.constants import CHECKPOINT_SCHEMA_VERSION
from reproduction.stage2i.fingerprint import canonical_json_sha256
from reproduction.stage2i.rng_state import restore_rng_state, snapshot_rng_state


TOY_PARAMETER_NAMES = ("shared_weight", "lm_only_weight")
TOY_SHARED_NAMES = ("shared_weight",)
TOY_PROVENANCE = {
    "source_commit": "0" * 40,
    "implementation_fingerprint": "1" * 64,
    "base_checkpoint_fingerprint": "2" * 64,
    "dataset_manifest_sha256": "3" * 64,
    "configuration_sha256": "4" * 64,
}


class ToyCABGModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_weight = nn.Parameter(torch.tensor([0.25, -0.125], dtype=torch.bfloat16).float())
        self.lm_only_weight = nn.Parameter(torch.tensor([0.0625, 0.1875], dtype=torch.bfloat16).float())


class ToyCABGEngine:
    def __init__(self) -> None:
        self.model = ToyCABGModel()
        parameters = dict(self.model.named_parameters())
        self.masters = {
            name: nn.Parameter(parameters[name].detach().to(torch.bfloat16).float().clone())
            for name in TOY_PARAMETER_NAMES
        }
        self.optimizer = torch.optim.AdamW(
            list(self.masters.values()),
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, 2) / 2.0)),
        )
        self.ema = BiasCorrectedEMA()
        self.completed_blocks = 0
        self.last_budget = {"lambda_raw": 0.0, "lambda_cap": 0.0, "lambda_final": 0.0}
        self.last_record_sha256 = "0" * 64

    @staticmethod
    def block(block_index: int):
        blocks = (
            (
                (torch.tensor([1.0, 0.5]), 0.4, "normal"),
                (torch.tensor([-0.5, 1.5]), -0.2, "abnormal"),
                (torch.tensor([0.75, -1.0]), 0.1, "abnormal"),
            ),
            (
                (torch.tensor([0.25, 1.25]), -0.3, "abnormal"),
                (torch.tensor([1.5, -0.25]), 0.6, "normal"),
                (torch.tensor([-1.0, 0.75]), 0.25, "abnormal"),
            ),
        )
        return blocks[block_index]

    def run_block(self, block_index: int) -> dict[str, float]:
        if block_index != self.completed_blocks:
            raise RuntimeError("Toy CABG block cursor mismatch.")
        parameters = dict(self.model.named_parameters())
        lm_per_image = []
        auxiliary_per_image = []
        labels = []
        for features, target, label in self.block(block_index):
            noise = torch.rand((), dtype=torch.float32) * 0.01
            prediction = (
                (parameters["shared_weight"] * features).sum()
                + (parameters["lm_only_weight"] * features).sum()
                + noise
            )
            lm_loss = (prediction - target).square()
            score = (parameters["shared_weight"] * features).sum()
            sign = 1.0 if label == "abnormal" else -1.0
            auxiliary_loss = torch.nn.functional.softplus(-sign * score)
            lm_gradients = torch.autograd.grad(
                lm_loss,
                [parameters[name] for name in TOY_PARAMETER_NAMES],
                retain_graph=True,
                allow_unused=False,
            )
            auxiliary_gradient = torch.autograd.grad(
                auxiliary_loss,
                parameters["shared_weight"],
                retain_graph=False,
                allow_unused=False,
            )[0]
            lm_per_image.append(dict(zip(TOY_PARAMETER_NAMES, lm_gradients)))
            auxiliary_per_image.append({"shared_weight": auxiliary_gradient})
            labels.append(label)
        lm_rms, _ = per_image_class_balanced_rms(
            [{"shared_weight": value["shared_weight"]} for value in lm_per_image], labels
        )
        auxiliary_rms, _ = per_image_class_balanced_rms(auxiliary_per_image, labels)
        aggregate_lm = aggregate_lm_gradients(lm_per_image)
        aggregate_auxiliary = aggregate_class_balanced_gradients(auxiliary_per_image, labels)
        budget = compute_budget(
            lm_rms=float(lm_rms),
            auxiliary_rms=float(auxiliary_rms),
            lm_shared_gradient={"shared_weight": aggregate_lm["shared_weight"]},
            auxiliary_shared_gradient=aggregate_auxiliary,
            ema=self.ema,
        )
        combined, diagnostics = combine_and_clip_gradients(
            aggregate_lm,
            aggregate_auxiliary,
            shared_names=TOY_SHARED_NAMES,
            lambda_value=budget["lambda_final"],
        )
        self.optimizer.zero_grad(set_to_none=True)
        for name in TOY_PARAMETER_NAMES:
            self.masters[name].grad = combined[name].to(dtype=torch.float32).clone()
        self.optimizer.step()
        self.scheduler.step()
        with torch.no_grad():
            for name in TOY_PARAMETER_NAMES:
                parameters[name].copy_(self.masters[name].detach().to(torch.bfloat16).float())
        self.completed_blocks += 1
        self.last_budget = {
            key: float(budget[key]) for key in ("lambda_raw", "lambda_cap", "lambda_final")
        }
        self.last_record_sha256 = canonical_json_sha256(
            {
                "block": block_index,
                "completed": self.completed_blocks,
                "budget": self.last_budget,
                "rng_tail": torch.rand(4).tolist(),
            }
        )
        return {**self.last_budget, **diagnostics}

    def checkpoint_payload(self) -> dict[str, object]:
        parameters = dict(self.model.named_parameters())
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "adapter_state": {
                name: parameters[name].detach().to(torch.bfloat16).cpu().clone()
                for name in TOY_PARAMETER_NAMES
            },
            "master_state": {
                name: self.masters[name].detach().float().cpu().clone()
                for name in TOY_PARAMETER_NAMES
            },
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "cabg_state": {
                "m_lm": self.ema.lm,
                "m_auxiliary": self.ema.auxiliary,
                "valid_blocks": self.ema.valid_blocks,
                "last_lambda_raw": self.last_budget["lambda_raw"],
                "last_lambda_cap": self.last_budget["lambda_cap"],
                "last_lambda_final": self.last_budget["lambda_final"],
            },
            "sampler_state": {
                "seed": 123,
                "epoch": 0,
                "cursor": self.completed_blocks,
                "block_permutation_hash": "5" * 64,
            },
            "rng_state": snapshot_rng_state(),
            "provenance": dict(TOY_PROVENANCE),
            "trace_state": {
                "last_record_sha256": self.last_record_sha256,
                "completed_blocks": self.completed_blocks,
            },
        }

    def load_payload(self, payload: Mapping[str, object]) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name in TOY_PARAMETER_NAMES:
                parameters[name].copy_(payload["adapter_state"][name].float())
                self.masters[name].copy_(payload["master_state"][name].float())
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.ema.load_state_dict(
            {
                "beta": self.ema.beta,
                "lm": payload["cabg_state"]["m_lm"],
                "auxiliary": payload["cabg_state"]["m_auxiliary"],
                "valid_blocks": payload["cabg_state"]["valid_blocks"],
            }
        )
        self.last_budget = {
            "lambda_raw": float(payload["cabg_state"]["last_lambda_raw"]),
            "lambda_cap": float(payload["cabg_state"]["last_lambda_cap"]),
            "lambda_final": float(payload["cabg_state"]["last_lambda_final"]),
        }
        self.completed_blocks = int(payload["trace_state"]["completed_blocks"])
        self.last_record_sha256 = str(payload["trace_state"]["last_record_sha256"])
        if int(payload["sampler_state"]["cursor"]) != self.completed_blocks:
            raise RuntimeError("Toy CABG sampler/checkpoint cursor mismatch.")
        restore_rng_state(payload["rng_state"])

    def result_state(self) -> dict[str, object]:
        parameters = dict(self.model.named_parameters())
        return {
            "model": {name: parameters[name].detach().cpu().clone() for name in TOY_PARAMETER_NAMES},
            "masters": {name: self.masters[name].detach().cpu().clone() for name in TOY_PARAMETER_NAMES},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "completed_blocks": self.completed_blocks,
            "last_budget": dict(self.last_budget),
            "last_record_sha256": self.last_record_sha256,
        }
