#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from reproduction.stage2g.evaluation_core import sha256_file
from reproduction.stage2g.source_fingerprint import stage2g_source_fingerprint


R0_SOURCE_COMMIT = "ad62e7c910f4febad7b07030bd1c11796ae064e7"
ADAPTER_SOURCE_COMMIT = "28e72711c38e9c47865657a009a2867f7205c330"
R0_REVISION = "9374b660aade05e190471c5501fefac548982169"
BASE_REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
BASE_FINGERPRINT = "9bc1b0de134f6a94f3ddaa6e6b0f3566c4ee55a2dbdccea27e2cb8432a02e16d"
ADAPTER_SOURCE_FINGERPRINT = "fb1fa1efb3f5c63fb0ae56dbe6674656b2286997e3f073f06839120837745d64"


PAIR_ORDER = (
    (42, 1, "B0"),
    (42, 1, "A3"),
    (42, 2, "A3"),
    (42, 2, "B0"),
    (123, 1, "A3"),
    (123, 1, "B0"),
    (123, 2, "B0"),
    (123, 2, "A3"),
    (2026, 1, "B0"),
    (2026, 1, "A3"),
    (2026, 2, "A3"),
    (2026, 2, "B0"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage2f-adapter-evidence", type=Path, required=True)
    parser.add_argument("--remote-adapter-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite run registry: {args.output}")

    runs = [
        {
            "order": 0,
            "run_id": "stage2g-r0",
            "run_role": "published_reference",
            "method": "R0",
            "source_commit": R0_SOURCE_COMMIT,
            "model_path": "/home/checkpoints/MEDIC-AD",
            "model_revision": R0_REVISION,
            "excluded_from_fb_maq_decision": True,
        }
    ]
    for order, (seed, repeat, method) in enumerate(PAIR_ORDER, start=1):
        train_run_id = f"seed-{seed}-repeat-{repeat}-{method.lower()}"
        local_adapter = args.stage2f_adapter_evidence / f"{train_run_id}.safetensors"
        local_manifest = args.stage2f_adapter_evidence / f"{train_run_id}.manifest.json"
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        adapter_sha256 = sha256_file(local_adapter)
        if manifest.get("checkpoint_sha256") != adapter_sha256:
            raise RuntimeError(f"Local adapter/manifest mismatch: {train_run_id}")
        metadata = manifest.get("metadata", {})
        expected_mode = "single" if method == "B0" else "multiscale"
        required = {
            "run_id": train_run_id,
            "seed": seed,
            "data_seed": seed,
            "repeat": repeat,
            "global_step": 32,
            "stage": "2F",
            "anomaly_query_mode": expected_mode,
            "implementation_source_fingerprint": ADAPTER_SOURCE_FINGERPRINT,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(f"Adapter metadata mismatch {train_run_id}: {key}")
        runs.append(
            {
                "order": order,
                "run_id": f"stage2g-{train_run_id}",
                "run_role": "paired_method",
                "method": method,
                "mode": expected_mode,
                "seed": seed,
                "repeat": repeat,
                "adapter_train_run_id": train_run_id,
                "adapter_path": str(args.remote_adapter_root / f"{train_run_id}.safetensors"),
                "adapter_manifest_path": str(
                    args.remote_adapter_root / f"{train_run_id}.manifest.json"
                ),
                "adapter_sha256": adapter_sha256,
                "adapter_manifest_sha256": sha256_file(local_manifest),
                "source_commit": ADAPTER_SOURCE_COMMIT,
                "source_fingerprint": ADAPTER_SOURCE_FINGERPRINT,
                "base_model_path": "/home/checkpoints/Lingshu-7B",
                "base_model_revision": BASE_REVISION,
                "base_checkpoint_fingerprint": BASE_FINGERPRINT,
            }
        )
    if len(runs) != 13 or [record["order"] for record in runs] != list(range(13)):
        raise RuntimeError("Stage 2G registry must contain exactly 13 ordered runs.")
    document = {
        "status": "LOCKED",
        "stage": "2G",
        "protocol_approved_date": "2026-08-31",
        "decision_margin": 0.02,
        "primary_endpoint": "balanced_accuracy",
        "evaluation_records_per_run": 228,
        "evaluation_counts": {"good": 87, "ungood": 141},
        "fresh_process_required": True,
        "quantization": False,
        "cpu_offloading": False,
        "stage2g_source_fingerprint": stage2g_source_fingerprint(args.repo_root),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**document, "registry_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
