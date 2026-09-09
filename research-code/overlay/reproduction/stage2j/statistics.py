"""Pure statistics and support logic for CABG-MIL v1.2 R0."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from reproduction.stage2i.cabg_math import CABGContractError


QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
GRADIENT_FLOOR = 1e-12


def distribution_record(value: torch.Tensor, *, kind: str) -> dict[str, object]:
    working = value.detach().float().reshape(-1)
    if working.numel() == 0 or not torch.isfinite(working).all():
        raise CABGContractError(f"R0 {kind} distribution is empty or non-finite.")
    q = torch.quantile(working, torch.tensor(QUANTILES, device=working.device))
    record: dict[str, object] = {
        "count": int(working.numel()),
        "min": float(working.min().cpu()),
        "max": float(working.max().cpu()),
        "mean": float(working.mean().cpu()),
        "median": float(working.median().cpu()),
        "std_population": float(working.std(unbiased=False).cpu()),
        "quantiles": {f"q{int(level * 100):02d}": float(number.cpu()) for level, number in zip(QUANTILES, q)},
    }
    if kind == "probability":
        saturated = (working <= 0.01) | (working >= 0.99)
        derivative = working * (1.0 - working)
        record.update(
            {
                "saturation_fraction": float(saturated.float().mean().cpu()),
                "sigmoid_derivative_mean": float(derivative.mean().cpu()),
                "sigmoid_derivative_median": float(derivative.median().cpu()),
                "sigmoid_derivative_le_1e-3_fraction": float((derivative <= 1e-3).float().mean().cpu()),
            }
        )
    elif kind == "evidence":
        record["saturation_fraction"] = float((working.abs() >= 0.95).float().mean().cpu())
    elif kind not in {"raw_logit", "raw_gap", "score", "loss"}:
        raise CABGContractError(f"Unknown R0 distribution kind: {kind}")
    return record


def layer_distribution_records(value: torch.Tensor, *, kind: str) -> list[dict[str, object]]:
    if value.ndim != 3 or value.shape[0] != 1:
        raise CABGContractError("R0 layer statistics require [1,layers,positions].")
    return [
        {"layer_index": index, **distribution_record(value[:, index, :], kind=kind)}
        for index in range(value.shape[1])
    ]


def gradient_row(name: str, gradient: torch.Tensor | None) -> dict[str, object]:
    if gradient is None:
        return {"name": name, "present": False, "finite": True, "effective": False, "max_abs": 0.0, "l2_norm": 0.0}
    value = gradient.detach().float()
    finite = bool(torch.isfinite(value).all().item())
    maximum = float(value.abs().max().cpu()) if finite else None
    norm = float(torch.linalg.vector_norm(value.reshape(-1)).cpu()) if finite else None
    return {
        "name": name,
        "present": True,
        "finite": finite,
        "effective": bool(finite and maximum is not None and maximum > GRADIENT_FLOOR),
        "max_abs": maximum,
        "l2_norm": norm,
    }


def gradient_rows(names: Sequence[str], gradients: Sequence[torch.Tensor | None]) -> list[dict[str, object]]:
    if len(names) != len(gradients) or len(set(names)) != len(names):
        raise CABGContractError("R0 gradient names and tensors are misaligned.")
    return [gradient_row(name, gradient) for name, gradient in zip(names, gradients)]


def effective_support(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(str(row["name"]) for row in rows if row.get("effective") is True)


def support_intersection(
    lm_rows: Sequence[Mapping[str, object]], auxiliary_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    lm = set(effective_support(lm_rows))
    auxiliary = set(effective_support(auxiliary_rows))
    shared = lm & auxiliary
    return {
        "lm_effective": sorted(lm),
        "auxiliary_effective": sorted(auxiliary),
        "actual_intersection": sorted(shared),
        "lm_tensor_count": len(lm),
        "auxiliary_tensor_count": len(auxiliary),
        "intersection_tensor_count": len(shared),
    }


def named_global_norm(rows: Sequence[Mapping[str, object]], names: Sequence[str] | None = None) -> float:
    selected = set(names) if names is not None else None
    square = 0.0
    for row in rows:
        if selected is not None and str(row["name"]) not in selected:
            continue
        norm = float(row["l2_norm"] or 0.0)
        if not math.isfinite(norm):
            raise CABGContractError("R0 gradient norm is non-finite.")
        square += norm * norm
    return math.sqrt(square)


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        raise CABGContractError("R0 Pearson inputs are misaligned or too short.")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))
