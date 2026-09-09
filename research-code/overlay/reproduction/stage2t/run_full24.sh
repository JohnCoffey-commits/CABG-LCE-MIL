#!/usr/bin/env bash
# This script is the authorized full24 campaign only. It has no causal-arm runner.
set -euo pipefail
set -o noclobber
commit="$1"
root="$2"
python=/home/envs/medic-ad-train/bin/python
export PYTHONPATH="$PWD:$PWD/qwen-vl-finetune"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
train() {
  local phase="$1"; shift
  "$python" -m reproduction.stage2t.train --campaign "$root" --output "$root/$phase" --code-commit "$commit" "$@" > "$root/$phase.log" 2>&1
}
train R0 --run-id R0 --mode train --end 24
train R1a --run-id R1 --mode train --end 12
train R1b --run-id R1 --mode train --start 12 --end 24 --resume "$root/R1a/block12-checkpoint/checkpoint.pt"
train replay-R0-b13 --run-id R0 --mode replay --start 12 --end 13 --resume "$root/R0/block12-checkpoint/checkpoint.pt"
train replay-R0-b24 --run-id R0 --mode replay --start 23 --end 24 --resume "$root/R0/block23-checkpoint/checkpoint.pt"
train replay-R1-b24 --run-id R1 --mode replay --start 23 --end 24 --resume "$root/R1b/block23-checkpoint/checkpoint.pt"
for run in 0 1; do
  for step in 12 24; do
    phase=R0
    if [[ "$run" == 1 ]]; then
      phase=R1a
      if [[ "$step" == 24 ]]; then phase=R1b; fi
    fi
    "$python" -m reproduction.stage2t.evaluate --campaign "$root" --output "$root/eval-R$run-b$step" --run-id "R$run" --cursor "$step" --code-commit "$commit" --checkpoint "$root/$phase/block$step-checkpoint/checkpoint.pt" > "$root/eval-R$run-b$step.log" 2>&1
  done
done
"$python" -m reproduction.stage2t.verify --campaign "$root" --output "$root/verification-v1" --code-commit "$commit" > "$root/verification-v1.log" 2>&1
