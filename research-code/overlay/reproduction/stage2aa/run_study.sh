#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
[[ $# == 3 && $1 == --execute ]] || { echo 'usage: run_study.sh --execute COMMIT NEW_ROOT'; exit 2; }
commit=$2
root=$3
[[ ! -e "$root" ]] || { echo 'New campaign root required'; exit 2; }
python=/home/envs/medic-ad-train/bin/python
export PYTHONPATH="$PWD:$PWD/qwen-vl-finetune" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2aa.preflight --campaign "$root" --code-commit "$commit"
for phase in s43-late_only s44-late_only replay-s43-b21 replay-s44-b21; do
  "$python" -m reproduction.stage2aa.train --execute --campaign "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2aa.verify --campaign "$root" --code-commit "$commit" --stage train > "$root/training-verification.log" 2>&1
for phase in eval-ref-s43-lm_only24 eval-ref-s44-lm_only24; do
  "$python" -m reproduction.stage2aa.evaluate --execute --campaign "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2aa.verify --campaign "$root" --code-commit "$commit" --stage reference > "$root/reference-verification.log" 2>&1
for phase in eval-s43-late_only24 eval-s44-late_only24; do
  "$python" -m reproduction.stage2aa.evaluate --execute --campaign "$root" --code-commit "$commit" --phase "$phase" > "$root/$phase.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2aa.analysis --campaign "$root" > "$root/comparison-analysis.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" -m reproduction.stage2aa.verify --campaign "$root" --code-commit "$commit" --stage final > "$root/verification.log" 2>&1
