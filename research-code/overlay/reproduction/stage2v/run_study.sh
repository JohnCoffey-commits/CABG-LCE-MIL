#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
[[ "${1:-}" == --execute ]] || { echo 'Explicit authorization and --execute required' >&2; exit 2; }
commit="$2"; root="$3"
[[ ! -e "$root" ]] || { echo 'New campaign root required' >&2; exit 2; }
python=/home/envs/medic-ad-train/bin/python
export PYTHONPATH="$PWD:$PWD/qwen-vl-finetune" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="" "$python" -m reproduction.stage2v.preflight --campaign "$root" --code-commit "$commit"
for phase in bridge-b13 bridge-b24; do
  "$python" -m reproduction.stage2v.train --execute --campaign "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
done
"$python" -m reproduction.stage2v.verify --campaign "$root" --code-commit "$commit" --stage bridge-train > "$root/bridge-train-verification.log" 2>&1
"$python" -m reproduction.stage2v.evaluate --execute --campaign "$root" --code-commit "$commit" --phase bridge-eval-b24 > "$root/bridge-eval-b24.log" 2>&1
"$python" -m reproduction.stage2v.verify --campaign "$root" --code-commit "$commit" --stage bridge > "$root/bridge-verification.log" 2>&1
for seed in 43 44; do
  for arm in prefix dynamic lce_off; do
    phase="s$seed-$arm"
    "$python" -m reproduction.stage2v.train --execute --campaign "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
  done
done
for seed in 43 44; do
  for endpoint in parent12 dynamic24 lce_off24; do
    phase="eval-s$seed-$endpoint"
    "$python" -m reproduction.stage2v.evaluate --execute --campaign "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
  done
done
"$python" -m reproduction.stage2v.verify --campaign "$root" --code-commit "$commit" --stage final > "$root/verification.log" 2>&1
