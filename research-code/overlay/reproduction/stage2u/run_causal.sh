#!/usr/bin/env bash
# DO NOT RUN without new explicit user approval for this multi-arm GPU experiment.
set -euo pipefail
set -o noclobber
[[ "${1:-}" == --execute ]] || { echo 'Preparation only: explicit user approval and --execute required' >&2; exit 2; }
commit="$2"; plan="$3"; root="$4"
[[ ! -e "$root" ]] || { echo 'Campaign root already exists' >&2; exit 2; }
python=/home/envs/medic-ad-train/bin/python
export PYTHONPATH="$PWD:$PWD/qwen-vl-finetune" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
mkdir "$root"
for arm in dynamic frozen_ratio lce_off; do
  "$python" -m reproduction.stage2u.train --execute --code-commit "$commit" --plan "$plan" --campaign "$root" --arm "$arm" > "$root/$arm.log" 2>&1
done
for arm in dynamic frozen_ratio lce_off; do
  "$python" -m reproduction.stage2u.evaluate --execute --code-commit "$commit" --plan "$plan" --campaign "$root" --arm "$arm" --output "$root/eval-$arm" --checkpoint "$root/$arm/block24-checkpoint/checkpoint.pt" > "$root/eval-$arm.log" 2>&1
done
"$python" -m reproduction.stage2u.verify --code-commit "$commit" --plan "$plan" --campaign "$root" --output "$root/verification.json" > "$root/verification.log" 2>&1
