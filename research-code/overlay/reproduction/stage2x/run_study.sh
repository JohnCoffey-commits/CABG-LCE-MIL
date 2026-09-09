#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && $1 == --execute ]] || { echo 'usage: run_study.sh --execute COMMIT NEW_ROOT'; exit 2; }
commit=$2
root=$3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PWD}:${PWD}/qwen-vl-finetune"
export TOKENIZERS_PARALLELISM=false
python=/home/envs/medic-ad-train/bin/python
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2x.prepare --output "$root" --code-commit "$commit"
for anchor in s43-initial s43-prefix8 s43-parent12 s43-dynamic20 s43-off20 s44-initial s44-prefix8 s44-parent12 s44-dynamic20 s44-off20; do
  "$python" -m reproduction.stage2x.run --output "$root" --code-commit "$commit" --anchor "$anchor" --execute > "$root/$anchor.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2x.math --output "$root" > "$root/response-analysis.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2x.verify --output "$root" --code-commit "$commit" > "$root/verification.log" 2>&1
