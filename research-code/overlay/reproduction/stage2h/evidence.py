import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


EVIDENCE_MARGIN = 0.1
TOPK_FRACTION = 0.01
LAMBDA_CANDIDATES = (0.01, 0.03, 0.1, 0.3)
CALIBRATION_TARGET_RATIO = 0.1
CALIBRATION_RATIO_BAND = (0.05, 0.20)


def compute_evidence_margin_loss(
    anomaly_evidence: torch.Tensor,
    anomaly_labels: torch.Tensor,
    *,
    margin: float = EVIDENCE_MARGIN,
    topk_fraction: float = TOPK_FRACTION,
) -> Dict[str, torch.Tensor | int]:
    """Compute LAD-MIL v2 on pre-gate sigmoid-difference evidence.

    `anomaly_evidence` must be `[images, layers, spatial_positions]` and
    `anomaly_labels` must have one explicit binary label per image. All loss
    arithmetic is FP32 while preserving the gradient graph to evidence.
    """

    if anomaly_evidence is None:
        raise RuntimeError("Stage 2H requires explicit anomaly evidence in the model output.")
    if anomaly_evidence.ndim != 3:
        raise RuntimeError(
            "Stage 2H evidence must have shape [images,layers,positions], "
            f"got {tuple(anomaly_evidence.shape)}."
        )
    if anomaly_evidence.shape[0] <= 0 or anomaly_evidence.shape[1] <= 0 or anomaly_evidence.shape[2] <= 0:
        raise RuntimeError("Stage 2H evidence dimensions must all be positive.")
    if not anomaly_evidence.is_floating_point():
        raise RuntimeError("Stage 2H evidence must be floating point.")
    if not torch.isfinite(anomaly_evidence).all():
        raise RuntimeError("Stage 2H evidence contains NaN or Inf.")

    labels = anomaly_labels
    if labels is None:
        raise RuntimeError("Stage 2H anomaly_labels are missing.")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.numel() != anomaly_evidence.shape[0]:
        raise RuntimeError(
            "Stage 2H evidence/label count mismatch: "
            f"{anomaly_evidence.shape[0]} evidence rows vs {labels.numel()} labels."
        )
    labels = labels.to(device=anomaly_evidence.device)
    if labels.dtype == torch.bool or labels.is_floating_point():
        raise RuntimeError("Stage 2H anomaly_labels must be integer 0 or 1.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise RuntimeError("Stage 2H anomaly_labels contain a value outside {0,1}.")

    evidence_fp32 = anomaly_evidence.float()
    layer_mean = evidence_fp32.mean(dim=1)
    positions = int(layer_mean.shape[1])
    k = max(1, int(math.ceil(float(topk_fraction) * positions)))
    topk_scores = torch.topk(layer_mean, k=k, dim=1, largest=True, sorted=False).values
    image_scores = topk_scores.mean(dim=1)
    targets = labels.to(dtype=torch.float32).mul(2.0).sub(1.0)
    per_image_loss = F.relu(float(margin) - targets * image_scores).square()
    loss = per_image_loss.mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Stage 2H evidence loss is NaN or Inf.")

    saturation_count = (evidence_fp32.abs() >= 0.95).sum()
    saturation_total = torch.tensor(evidence_fp32.numel(), device=evidence_fp32.device)
    return {
        "loss": loss,
        "image_scores": image_scores,
        "layer_mean": layer_mean,
        "per_image_loss": per_image_loss,
        "k": k,
        "positions": positions,
        "saturation_count": saturation_count,
        "saturation_total": saturation_total,
    }


def _fp32_gradient_norm(gradient: torch.Tensor) -> torch.Tensor:
    flat = gradient.detach().float().reshape(-1)
    return torch.linalg.vector_norm(flat, ord=2)


def calibration_parameter_items(model) -> Tuple[Tuple[str, torch.nn.Parameter], ...]:
    prefix = "model.visual.anomaly_qformer."
    suffixes = (
        "abnormal_prompt",
        "normal_prompt",
        "anomaly_attention.query_proj.weight",
        "anomaly_attention.query_proj.bias",
        "anomaly_attention.key_proj.weight",
        "anomaly_attention.key_proj.bias",
    )
    parameters = dict(model.named_parameters())
    items = []
    for suffix in suffixes:
        name = prefix + suffix
        parameter = parameters.get(name)
        if parameter is None:
            raise RuntimeError(f"Stage 2H calibration parameter is missing: {name}")
        items.append((name, parameter))
    return tuple(items)


class LambdaCalibrationAccumulator:
    """Accumulate exact mean-loss gradients over sequential microbatches."""

    def __init__(self, model):
        self.named_parameters = calibration_parameter_items(model)
        self._gradient_sums = {
            loss_name: {
                name: torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float32)
                for name, parameter in self.named_parameters
            }
            for loss_name in ("lm", "evidence")
        }
        self._missing_counts = {
            loss_name: {name: 0 for name, _ in self.named_parameters}
            for loss_name in ("lm", "evidence")
        }
        self._nonfinite_counts = {
            loss_name: {name: 0 for name, _ in self.named_parameters}
            for loss_name in ("lm", "evidence")
        }
        self._loss_sums = {"lm": 0.0, "evidence": 0.0}
        self.microbatch_count = 0

    def add(self, *, lm_loss: torch.Tensor, evidence_loss: torch.Tensor) -> None:
        if not bool(torch.isfinite(lm_loss).detach().cpu().item()):
            raise RuntimeError("Stage 2H LM loss is NaN or Inf during calibration.")
        if not bool(torch.isfinite(evidence_loss).detach().cpu().item()):
            raise RuntimeError("Stage 2H evidence loss is NaN or Inf during calibration.")

        parameters = [parameter for _, parameter in self.named_parameters]
        gradient_sets = {
            "lm": torch.autograd.grad(
                lm_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            ),
            "evidence": torch.autograd.grad(
                evidence_loss,
                parameters,
                retain_graph=False,
                allow_unused=True,
            ),
        }
        for loss_name, gradients in gradient_sets.items():
            for (name, _), gradient in zip(self.named_parameters, gradients):
                if gradient is None:
                    self._missing_counts[loss_name][name] += 1
                    continue
                gradient_fp32 = gradient.detach().to(device="cpu", dtype=torch.float32)
                if not bool(torch.isfinite(gradient_fp32).all().item()):
                    self._nonfinite_counts[loss_name][name] += 1
                    continue
                self._gradient_sums[loss_name][name].add_(gradient_fp32)
        self._loss_sums["lm"] += float(lm_loss.detach().float().cpu().item())
        self._loss_sums["evidence"] += float(evidence_loss.detach().float().cpu().item())
        self.microbatch_count += 1

    def finalize(self, candidates: Iterable[float] = LAMBDA_CANDIDATES) -> Dict:
        if self.microbatch_count <= 0:
            raise RuntimeError("Stage 2H lambda calibration received no microbatches.")
        records = []
        common_lm_squared = 0.0
        common_evidence_squared = 0.0
        common_count = 0
        for name, _ in self.named_parameters:
            row = {"name": name}
            norms = {}
            included = True
            for loss_name in ("lm", "evidence"):
                missing_count = self._missing_counts[loss_name][name]
                nonfinite_count = self._nonfinite_counts[loss_name][name]
                complete = missing_count == 0 and nonfinite_count == 0
                norm = None
                if complete:
                    mean_gradient = self._gradient_sums[loss_name][name].div(
                        float(self.microbatch_count)
                    )
                    norm = float(_fp32_gradient_norm(mean_gradient).item())
                    complete = math.isfinite(norm)
                row[f"{loss_name}_gradient_missing_count"] = missing_count
                row[f"{loss_name}_gradient_nonfinite_count"] = nonfinite_count
                row[f"{loss_name}_mean_gradient_finite"] = complete
                row[f"{loss_name}_mean_gradient_norm_fp32"] = norm
                norms[loss_name] = norm
                included = included and complete
            row["included_in_global_norm"] = included
            if included:
                common_lm_squared += norms["lm"] * norms["lm"]
                common_evidence_squared += norms["evidence"] * norms["evidence"]
                common_count += 1
            records.append(row)

        if common_count == 0:
            raise RuntimeError("Stage 2H lambda calibration has no common finite gradients.")
        lm_global_norm = math.sqrt(common_lm_squared)
        evidence_global_norm = math.sqrt(common_evidence_squared)
        if not math.isfinite(lm_global_norm) or lm_global_norm == 0.0:
            raise RuntimeError("Stage 2H lambda calibration LM denominator is zero or non-finite.")
        if not math.isfinite(evidence_global_norm):
            raise RuntimeError("Stage 2H lambda calibration evidence norm is non-finite.")

        rows = []
        lower, upper = CALIBRATION_RATIO_BAND
        for candidate in sorted(float(value) for value in candidates):
            ratio = candidate * evidence_global_norm / lm_global_norm
            eligible = math.isfinite(ratio) and lower <= ratio <= upper
            rows.append(
                {
                    "lambda": candidate,
                    "gradient_ratio": ratio,
                    "distance_to_target": abs(ratio - CALIBRATION_TARGET_RATIO),
                    "eligible": eligible,
                }
            )
        eligible_rows = [row for row in rows if row["eligible"]]
        selected = (
            min(
                eligible_rows,
                key=lambda row: (row["distance_to_target"], row["lambda"]),
            )
            if eligible_rows
            else None
        )
        for row in rows:
            row["selected"] = selected is not None and row is selected

        return {
            "status": "SUCCESS" if selected is not None else "NO_ELIGIBLE_CANDIDATE",
            "logical_batch_size": self.microbatch_count,
            "microbatch_size": 1,
            "lm_loss_mean": self._loss_sums["lm"] / self.microbatch_count,
            "evidence_loss_mean": self._loss_sums["evidence"] / self.microbatch_count,
            "common_finite_parameter_count": common_count,
            "lm_global_gradient_norm_fp32": lm_global_norm,
            "evidence_global_gradient_norm_fp32": evidence_global_norm,
            "lm_denominator_zero": False,
            "parameter_gradients": records,
            "candidates": rows,
            "selected_lambda": selected["lambda"] if selected is not None else None,
            "target_ratio": CALIBRATION_TARGET_RATIO,
            "ratio_band": list(CALIBRATION_RATIO_BAND),
        }


def calibrate_lambda_from_losses(
    *,
    model,
    lm_loss: torch.Tensor,
    evidence_loss: torch.Tensor,
    candidates: Iterable[float] = LAMBDA_CANDIDATES,
) -> Dict:
    accumulator = LambdaCalibrationAccumulator(model)
    accumulator.add(lm_loss=lm_loss, evidence_loss=evidence_loss)
    return accumulator.finalize(candidates=candidates)
