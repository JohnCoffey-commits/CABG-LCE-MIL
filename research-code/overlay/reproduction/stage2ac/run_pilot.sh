#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
[[ $# == 2 && $1 == --execute ]] || exit 2
root=$2
python=/home/envs/medic-ad-train/bin/python
export PYTHONPATH="$PWD:$PWD/qwen-vl-finetune" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
for phase in s43-none s43-full s43-early s44-none s44-full s44-early features; do
  "$python" -B -m reproduction.stage2ac.run --execute --campaign "$root" --phase "$phase" > "$root/$phase.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' "$python" -B -m reproduction.stage2ac.fit --campaign "$root" > "$root/linear-fit.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" -B -m reproduction.stage2ac.verify --campaign "$root" > "$root/independent-verification.log" 2>&1
