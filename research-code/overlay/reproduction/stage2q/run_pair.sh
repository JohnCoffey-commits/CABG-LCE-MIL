#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then exit 2; fi
mode="$1"; setting="$2"; blocks="$3"; pair_root="$4"
case "$pair_root" in /home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-reproducibility-v1/*) ;; *) exit 3;; esac
if [ -e "$pair_root" ]; then echo 'Refuse overwrite' >&2; exit 4; fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=/home/envs/medic-ad-train/bin/python
root=/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-reproducibility-v1
export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune"
if [ "$setting" = strict_deterministic ]; then export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
mkdir -p "$pair_root"
run_pair() {
  for process in p0 p1; do
    "$python_bin" -m reproduction.stage2q.run --mode "$mode" --setting "$setting" --blocks "$blocks" --preflight "$root/preflight.json" --output "$pair_root/$process"
  done
  if [ "$mode" = diagnostic ]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python_bin" -m reproduction.stage2q.verify --root "$pair_root" --output "$pair_root/comparison.json"
  fi
}
set +e
(set -euo pipefail; run_pair) 2>&1 | tee "$pair_root/run.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$pair_root/run.exitcode"
exit "$rc"
