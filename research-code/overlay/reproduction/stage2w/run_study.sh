#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
[[ "${1:-}" == --execute ]] || { echo 'Explicit --execute required' >&2; exit 2; }
commit="$2"; root="$3"
[[ ! -e "$root" ]] || { echo 'New evidence root required' >&2; exit 2; }
python=/home/envs/medic-ad-train/bin/python
export PYTHONPATH="$PWD:$PWD/qwen-vl-finetune" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="" "$python" -m reproduction.stage2w.analyze --output "$root" --code-commit "$commit"
for phase in reference corners; do
  CUDA_VISIBLE_DEVICES=0 "$python" -m reproduction.stage2w.gpu --execute --output "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
done
CUDA_VISIBLE_DEVICES="" "$python" -m reproduction.stage2w.verify --output "$root" --code-commit "$commit" > "$root/verification.log" 2>&1
