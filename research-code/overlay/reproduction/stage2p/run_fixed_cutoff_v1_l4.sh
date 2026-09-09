#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 0 ]; then exit 2; fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
split_audit="/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json"
original_root="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-v1"
causal_verified="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-causal-branch-evaluator-repair-v1"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-fixed-cutoff-v1"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/d4-scout-fixed-cutoff-v1"
prereg="/home/Medic-AD/incoming/fixed-cutoff-preregistration.md"
run_log="$log_dir/run.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"
export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
for path in "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then echo "refuse overwrite: $path" >&2; exit 4; fi
done
for path in "$python_bin" "$base_model" "$split_audit" "$prereg" "$original_root/artifact-inventory.sha256" "$causal_verified/artifact-inventory.sha256"; do
  if [ ! -e "$path" ]; then echo "missing prerequisite: $path" >&2; exit 3; fi
done
mkdir -p "$output_dir" "$log_dir"
mark_stage() { printf '%s\n' "$1" > "$active_stage"; printf 'FIXED_CUTOFF_STAGE=%s\n' "$1"; }
assert_gpu_idle() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$1"
  if [ -s "$1" ]; then echo "GPU not idle" >&2; return 20; fi
}
run_arm() {
  local arm="$1"
  local arm_dir="$output_dir/$arm"
  mkdir -p "$arm_dir/checkpoints"
  mark_stage "${arm}_fresh_initialization_blocks1_12"
  "$python_bin" -m reproduction.stage2m.train_scout --arm "$arm" --phase phase1 \
    --run-root "$output_dir" --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --split-audit "$split_audit" --trace "$arm_dir/trace.jsonl" --checkpoint-output "$arm_dir/checkpoints/mid-block12.pt"
  assert_gpu_idle "$log_dir/$arm-post-phase1-gpu.csv"
  mark_stage "${arm}_own_midpoint_resume_blocks13_24"
  "$python_bin" -m reproduction.stage2m.train_scout --arm "$arm" --phase phase2 \
    --run-root "$output_dir" --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --split-audit "$split_audit" --trace "$arm_dir/trace.jsonl" \
    --resume-checkpoint "$arm_dir/checkpoints/mid-block12.pt" --checkpoint-output "$arm_dir/checkpoints/final-block24.pt"
  assert_gpu_idle "$log_dir/$arm-post-phase2-gpu.csv"
  mark_stage "${arm}_fixed_block24_evaluation"
  "$python_bin" -m reproduction.stage2m.evaluate_scout --arm "$arm" --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" --split-audit "$split_audit" \
    --checkpoint "$arm_dir/checkpoints/final-block24.pt" --expected-cursor 24 \
    --repo-root "$repo_root" --output-dir "$output_dir/evaluation/$arm"
  assert_gpu_idle "$log_dir/$arm-post-eval-gpu.csv"
}
run_pipeline() {
  cd "$repo_root"
  mark_stage prerequisite_rehash
  printf '6bd7819a35ff0ee1143ef5078ec444e83ffe23b5d6e96095d0a351c599629e97  %s\n' "$prereg" | sha256sum -c -
  cp "$prereg" "$output_dir/preregistration.md"
  sha256sum -c "$original_root/artifact-inventory.sha256" > "$log_dir/original-inventory-check.log"
  sha256sum -c "$causal_verified/producer-inventory.sha256" > "$log_dir/causal-producer-inventory-check.log"
  sha256sum -c "$causal_verified/artifact-inventory.sha256" > "$log_dir/causal-evaluator-inventory-check.log"
  assert_gpu_idle "$log_dir/pre-execution-gpu.csv"
  mark_stage import_and_tests
  "$python_bin" -c 'import qwenvl; import reproduction.stage2m.train_scout; import reproduction.stage2m.evaluate_scout; import reproduction.stage2p.verify_fixed_cutoff; print("IMPORT_GATE_SUCCESS")' > "$log_dir/import-gate.txt"
  "$python_bin" -m reproduction.stage2p.run_unit_tests --output "$log_dir/unit-tests.json"
  mark_stage preflight
  "$python_bin" -m reproduction.stage2m.preflight --repo-root "$repo_root" --base-model "$base_model" --split-audit "$split_audit" --output "$output_dir/preflight.json"
  "$python_bin" -m reproduction.stage2p.verify_fixed_cutoff --run-root "$output_dir" --split-audit "$split_audit" --data-only --output "$output_dir/data-preflight.json"
  run_arm cabg_original_control
  run_arm cabg_fixed_cutoff
  mark_stage independent_verification
  "$python_bin" -m reproduction.stage2p.verify_fixed_cutoff --run-root "$output_dir" --split-audit "$split_audit" --original-root "$original_root" --repo-root "$repo_root" --output "$output_dir/comparison.json"
  terminal_decision="$(jq -r '.decision' "$output_dir/comparison.json")"
  case "$terminal_decision" in PASS_FIXED_CUTOFF|NOT_VALIDATED|INCONCLUSIVE) ;; *) return 21;; esac
  printf '%s\n' "$terminal_decision" > "$output_dir/terminal-decision.txt"
  mark_stage complete
  inventory="$output_dir/artifact-inventory.sha256"
  find "$log_dir" "$output_dir" -type f ! -path "$inventory" ! -path "$run_log" ! -path "$run_status" ! -path "$output_dir/artifact-inventory-check.log" -print0 | sort -z | xargs -0 sha256sum > "$inventory"
  sha256sum -c "$inventory" > "$output_dir/artifact-inventory-check.log"
}
set +e
( set -euo pipefail; run_pipeline ) 2>&1 | tee "$run_log"
pipeline_status=${PIPESTATUS[0]}
set -e
if [ "$pipeline_status" -ne 0 ]; then
  printf 'INVALID_QUARANTINED:%s:%s\n' "$(cat "$active_stage")" "$pipeline_status" > "$run_status"
  exit "$pipeline_status"
fi
if [ "$(cat "$active_stage")" != complete ]; then exit 5; fi
sha256sum -c "$output_dir/artifact-inventory.sha256" > "$output_dir/post-wrapper-inventory-check.log"
cat "$output_dir/terminal-decision.txt" > "$run_status"
