#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from reproduction.stage2g.evaluation_core import (
    load_jsonl,
    sha256_file,
    validate_evaluation_manifest,
)
from reproduction.stage2g.source_fingerprint import stage2g_source_fingerprint


MIN_FREE_BYTES = 15 * 1024**3


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def checkpoint_shard_audit(model_dir: Path):
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    shards = []
    for name in shard_names:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = "".join(f"{record['file']}:{record['sha256']}\n" for record in shards).encode()
    return {
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "shard_count": len(shards),
        "physical_shard_bytes": sum(record["bytes"] for record in shards),
        "checkpoint_shard_list_fingerprint": hashlib.sha256(payload).hexdigest(),
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    parser.add_argument("--r0-source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--r0-model", type=Path, required=True)
    parser.add_argument("--r0-revision-file", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--r0-checkpoint-audit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.r0_checkpoint_audit_output, args.output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite preflight evidence: {path}")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "LOCKED" or len(registry.get("runs", [])) != 13:
        raise RuntimeError("Stage 2G run registry is not locked and complete.")
    current_fingerprint = stage2g_source_fingerprint(args.repo_root)
    if current_fingerprint != registry["stage2g_source_fingerprint"]:
        raise RuntimeError("Stage 2G implementation source fingerprint changed after registry lock.")
    if git_head(args.adapter_source) != registry["runs"][1]["source_commit"]:
        raise RuntimeError("Adapter-source worktree commit mismatch.")
    fingerprint_env = os.environ.copy()
    fingerprint_env["PYTHONPATH"] = (
        f"{args.adapter_source}:{args.adapter_source / 'qwen-vl-finetune'}"
    )
    adapter_fingerprint = subprocess.check_output(
        [
            "/home/envs/medic-ad-train/bin/python",
            str(args.adapter_source / "reproduction/stage2f/source_fingerprint.py"),
            "--repo-root",
            str(args.adapter_source),
        ],
        cwd=args.adapter_source,
        text=True,
        env=fingerprint_env,
    ).strip()
    if adapter_fingerprint != registry["runs"][1]["source_fingerprint"]:
        raise RuntimeError("Adapter-source fingerprint mismatch.")
    if git_head(args.r0_source) != registry["runs"][0]["source_commit"]:
        raise RuntimeError("R0 source worktree commit mismatch.")

    dataset_audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    if dataset_audit.get("status") != "SUCCESS" or dataset_audit.get("unique_records") != 228:
        raise RuntimeError("Dataset audit is not successful and complete.")
    manifest_sha256 = sha256_file(args.evaluation_manifest)
    if manifest_sha256 != dataset_audit.get("evaluation_manifest_sha256"):
        raise RuntimeError("Evaluation manifest differs from dataset audit.")
    manifest = load_jsonl(args.evaluation_manifest)
    validate_evaluation_manifest(manifest)
    for record in manifest:
        path = args.dataset_root / record["relative_path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Dataset preflight artifact mismatch: {record['relative_path']}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Dataset preflight hash mismatch: {record['relative_path']}")

    for run in registry["runs"][1:]:
        adapter = Path(run["adapter_path"])
        manifest_path = Path(run["adapter_manifest_path"])
        if sha256_file(adapter) != run["adapter_sha256"]:
            raise RuntimeError(f"Adapter hash mismatch: {run['run_id']}")
        if sha256_file(manifest_path) != run["adapter_manifest_sha256"]:
            raise RuntimeError(f"Adapter manifest hash mismatch: {run['run_id']}")
        adapter_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if adapter_manifest.get("checkpoint_sha256") != run["adapter_sha256"]:
            raise RuntimeError(f"Adapter manifest content mismatch: {run['run_id']}")

    base_audit = checkpoint_shard_audit(args.base_model)
    if base_audit["checkpoint_shard_list_fingerprint"] != registry["runs"][1][
        "base_checkpoint_fingerprint"
    ]:
        raise RuntimeError("Pinned Lingshu base checkpoint fingerprint mismatch.")
    r0_revision = args.r0_revision_file.read_text(encoding="utf-8").strip()
    if r0_revision != registry["runs"][0]["model_revision"]:
        raise RuntimeError("Published R0 revision file mismatch.")
    r0_audit = {
        "status": "SUCCESS",
        "stage": "2G",
        "model": "wooohyeooon/MEDIC-AD",
        "model_path": str(args.r0_model),
        "model_revision": r0_revision,
        **checkpoint_shard_audit(args.r0_model),
    }
    disk = shutil.disk_usage("/home")
    if disk.free < MIN_FREE_BYTES:
        raise RuntimeError(f"Remote /home free space below 15GiB: {disk.free}")
    args.r0_checkpoint_audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.r0_checkpoint_audit_output.write_text(
        json.dumps(r0_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "status": "SUCCESS",
        "stage": "2G",
        "gate": "PREEXPERIMENT_READY",
        "registry": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "stage2g_source_fingerprint": current_fingerprint,
        "adapter_source_commit": git_head(args.adapter_source),
        "adapter_source_fingerprint": adapter_fingerprint,
        "r0_source_commit": git_head(args.r0_source),
        "dataset_audit_sha256": sha256_file(args.dataset_audit),
        "evaluation_manifest_sha256": manifest_sha256,
        "evaluation_records": len(manifest),
        "verified_adapter_count": 12,
        "base_checkpoint_fingerprint": base_audit["checkpoint_shard_list_fingerprint"],
        "r0_checkpoint_audit": str(args.r0_checkpoint_audit_output),
        "r0_checkpoint_audit_sha256": sha256_file(args.r0_checkpoint_audit_output),
        "home_free_bytes": disk.free,
        "home_free_gib": disk.free / 1024**3,
        "quantization": False,
        "cpu_offloading": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
