#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path


SOURCE_FILES = (
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py",
    "qwen-vl-finetune/qwenvl/train/argument.py",
    "qwen-vl-finetune/qwenvl/train/train_qwen.py",
    "qwen-vl-finetune/qwenvl/train/trainable_checkpoint.py",
    "reproduction/stage2f/adapter_schema.py",
    "reproduction/stage2f/evaluate_vqarad_exploratory.py",
    "reproduction/stage2f/training_diagnostics.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_source_fingerprint(repo_root: Path) -> str:
    records = []
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(f"{relative}:{sha256_file(path)}")
    payload = ("\n".join(records) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(implementation_source_fingerprint(args.repo_root.resolve()))


if __name__ == "__main__":
    main()
