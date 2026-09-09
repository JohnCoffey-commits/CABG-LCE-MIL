"""Canonical file, source, support, and dataset fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_dataset_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    required = {"sample_id", "sha256", "relative_path", "label"}
    canonical = []
    identities = set()
    sample_ids = set()
    content_hashes = set()
    relative_paths = set()
    for source in records:
        if not required.issubset(source):
            raise CABGContractError(f"Dataset record is missing keys: {sorted(required - set(source))}")
        record = dict(source)
        sample_id = str(record["sample_id"])
        sha256 = str(record["sha256"]).lower()
        relative_path = Path(str(record["relative_path"])).as_posix()
        if not sample_id or not HEX64.fullmatch(sha256):
            raise CABGContractError("Dataset sample ID or SHA-256 is invalid.")
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            raise CABGContractError("Dataset path must be canonical and relative.")
        identity = (sample_id, sha256, relative_path)
        if (
            identity in identities
            or sample_id in sample_ids
            or sha256 in content_hashes
            or relative_path in relative_paths
        ):
            raise CABGContractError(f"Duplicate dataset identity component: {identity}")
        identities.add(identity)
        sample_ids.add(sample_id)
        content_hashes.add(sha256)
        relative_paths.add(relative_path)
        record["sample_id"] = sample_id
        record["sha256"] = sha256
        record["relative_path"] = relative_path
        record["label"] = normalize_label(record["label"])
        canonical.append(record)
    canonical.sort(key=lambda value: (value["label"], value["sample_id"], value["sha256"]))
    return canonical


def dataset_fingerprint(records: Iterable[Mapping[str, object]]) -> str:
    return canonical_json_sha256(canonical_dataset_records(records))


def implementation_source_record(repo_root: Path, relative_paths: Sequence[str]) -> dict[str, object]:
    if not relative_paths or len(set(relative_paths)) != len(relative_paths):
        raise CABGContractError("Source inventory paths must be non-empty and unique.")
    files = []
    for relative_path in sorted(relative_paths):
        normalized = Path(relative_path).as_posix()
        if normalized.startswith("/") or ".." in Path(normalized).parts:
            raise CABGContractError(f"Invalid source inventory path: {relative_path}")
        path = repo_root / normalized
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": normalized, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "fingerprint": canonical_json_sha256(files)}


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parameter_inventory(
    named_parameters: Mapping[str, torch.Tensor], names: Sequence[str]
) -> dict[str, object]:
    if len(set(names)) != len(names) or not names:
        raise CABGContractError("Parameter inventory names must be non-empty and unique.")
    missing = sorted(set(names) - set(named_parameters))
    if missing:
        raise CABGContractError(f"Parameter inventory is missing names: {missing}")
    rows = []
    for name in names:
        tensor = named_parameters[name]
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "elements": int(tensor.numel()),
                "tensor_sha256": tensor_sha256(tensor),
            }
        )
    rows.sort(key=lambda value: value["name"])
    return {
        "parameters": rows,
        "tensor_count": len(rows),
        "element_count": sum(int(value["elements"]) for value in rows),
        "fingerprint": canonical_json_sha256(rows),
    }


def git_identity(repo_root: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    )
    return {"commit": commit, "dirty": bool(status.strip()), "status_sha256": hashlib.sha256(status.encode()).hexdigest()}
