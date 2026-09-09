#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path


def max_abs_difference(left, right):
    if len(left) != len(right):
        raise RuntimeError(f"Sequence length mismatch: {len(left)} != {len(right)}")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def absolute_differences(left, right):
    if len(left) != len(right):
        raise RuntimeError(f"Sequence length mismatch: {len(left)} != {len(right)}")
    return [abs(float(a) - float(b)) for a, b in zip(left, right)]


def main() -> None:
    log_dir = Path(os.environ.get("MEDIC_AD_LOG_DIR", "/home/logs/medic-ad/training-multisample"))
    run_ids = (
        os.environ.get("MEDIC_AD_REPEAT_RUN_A", "run-a-poststep-eval"),
        os.environ.get("MEDIC_AD_REPEAT_RUN_B", "run-b-poststep-eval"),
    )
    summaries = {
        run_id: json.loads((log_dir / f"vqarad-8step-{run_id}-verification.json").read_text())
        for run_id in run_ids
    }
    for run_id, summary in summaries.items():
        if summary.get("status") != "SUCCESS":
            raise RuntimeError(f"{run_id} verification was not successful.")

    train_loss_difference = max_abs_difference(
        summaries[run_ids[0]]["training_losses"],
        summaries[run_ids[1]]["training_losses"],
    )
    eval_loss_difference = max_abs_difference(
        summaries[run_ids[0]]["eval_losses"],
        summaries[run_ids[1]]["eval_losses"],
    )
    gradient_norm_difference = max_abs_difference(
        summaries[run_ids[0]]["gradient_norms"],
        summaries[run_ids[1]]["gradient_norms"],
    )

    tolerance = 1e-10
    if not all(math.isfinite(value) for value in (train_loss_difference, eval_loss_difference, gradient_norm_difference)):
        raise RuntimeError("Repeatability difference is non-finite.")

    strict_repeatability = all(
        value <= tolerance
        for value in (train_loss_difference, eval_loss_difference, gradient_norm_difference)
    )

    result = {
        "status": "SUCCESS" if strict_repeatability else "FAILED",
        "strict_repeatability": strict_repeatability,
        "absolute_tolerance": tolerance,
        "max_abs_training_loss_difference": train_loss_difference,
        "max_abs_eval_loss_difference": eval_loss_difference,
        "max_abs_gradient_norm_difference": gradient_norm_difference,
        "training_loss_absolute_differences": absolute_differences(
            summaries[run_ids[0]]["training_losses"],
            summaries[run_ids[1]]["training_losses"],
        ),
        "eval_loss_absolute_differences": absolute_differences(
            summaries[run_ids[0]]["eval_losses"],
            summaries[run_ids[1]]["eval_losses"],
        ),
        "gradient_norm_absolute_differences": absolute_differences(
            summaries[run_ids[0]]["gradient_norms"],
            summaries[run_ids[1]]["gradient_norms"],
        ),
        "run_ids": list(run_ids),
        "run_a_first_final_eval_loss": [
            summaries[run_ids[0]]["first_eval_loss"],
            summaries[run_ids[0]]["final_eval_loss"],
        ],
        "run_b_first_final_eval_loss": [
            summaries[run_ids[1]]["first_eval_loss"],
            summaries[run_ids[1]]["final_eval_loss"],
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not strict_repeatability:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
