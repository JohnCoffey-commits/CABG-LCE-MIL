#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from qwenvl.data.data_qwen import DataCollatorForSupervisedDataset


class ToyTokenizer:
    pad_token_id = 0
    model_max_length = 32


def sample(label=1, images=1):
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
        "position_ids": torch.arange(3).reshape(1, 1, 3).expand(3, 1, 3),
        "pixel_values": torch.zeros((images * 4, 3), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * images, dtype=torch.long),
        "anomaly_labels": torch.tensor([label] * images, dtype=torch.long),
        "masks": None,
    }


def expect_failure(callable_, fragment):
    try:
        callable_()
    except RuntimeError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"Expected {fragment!r}, got {str(exc)!r}") from exc
        return str(exc)
    raise AssertionError(f"Expected failure containing {fragment!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    collator = DataCollatorForSupervisedDataset(
        tokenizer=ToyTokenizer(), load_masks=False, require_anomaly_labels=True
    )
    batch = collator([sample(0), sample(1)])
    if batch["anomaly_labels"].tolist() != [0, 1]:
        raise AssertionError("Collator changed explicit anomaly-label order.")
    if int(batch["image_grid_thw"].shape[0]) != batch["anomaly_labels"].numel():
        raise AssertionError("Collator image/label alignment failed.")

    missing = sample()
    missing.pop("anomaly_labels")
    mismatch = sample()
    mismatch["anomaly_labels"] = torch.tensor([], dtype=torch.long)
    floating = sample()
    floating["anomaly_labels"] = floating["anomaly_labels"].float()
    payload = {
        "status": "SUCCESS",
        "valid_batch_labels": batch["anomaly_labels"].tolist(),
        "valid_batch_image_rows": int(batch["image_grid_thw"].shape[0]),
        "multi_image_rejected": expect_failure(lambda: collator([sample(images=2)]), "exactly one image"),
        "missing_label_rejected": expect_failure(lambda: collator([missing]), "missing anomaly_labels"),
        "count_mismatch_rejected": expect_failure(lambda: collator([mismatch]), "label/image mismatch"),
        "floating_label_rejected": expect_failure(lambda: collator([floating]), "integer tensors"),
        "invalid_label_rejected": expect_failure(lambda: collator([sample(2)]), "outside"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
