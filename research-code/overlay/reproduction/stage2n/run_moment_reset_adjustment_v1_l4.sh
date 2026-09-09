#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Moment-reset adjustment uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
split_audit="/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json"
original_root="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-v1"
registered_comparison="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-stability-repair-v1-evaluator-repair/comparison.json"
registered_inventory="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-stability-repair-v1-evaluator-repair/artifact-inventory.sha256"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-stability-repair-moment-reset-v1"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/d4-scout-stability-repair-moment-reset-v1"
run_log="$log_dir/run.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"
arm="cabg_spatial_guard_moment_reset"

export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
for path in "$python_bin" "$base_model" "$split_audit" "$original_root/comparison.json" "$registered_comparison" "$registered_inventory"; do
  if [ ! -e "$path" ]; then
    echo "missing moment-reset prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite moment-reset root: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir/$arm/checkpoints" "$log_dir"

mark_stage() {
  printf '%s\n' "$1" > "$active_stage"
  printf 'D4_SCOUT_MOMENT_RESET_STAGE=%s\n' "$1"
}

assert_gpu_idle() {
  local output_file="$1"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$output_file"
  if [ -s "$output_file" ]; then
    echo "GPU compute process remained at $output_file" >&2
    return 20
  fi
}

run_pipeline() {
  cd "$repo_root"
  mark_stage prerequisite_rehash
  sha256sum -c "$registered_inventory" > "$log_dir/registered-repair-evaluator-inventory-check.log"
  jq -e '.status == "SUCCESS" and .decision == "C"' "$registered_comparison" > "$log_dir/registered-c-decision-check.txt"
  mark_stage import_and_tests
  "$python_bin" - <<'PY' > "$log_dir/import-gate.txt"
import qwenvl
import reproduction.stage2m.train_scout
import reproduction.stage2m.evaluate_scout
import reproduction.stage2n.summarize_adjustment
print("D4_SCOUT_MOMENT_RESET_IMPORT_GATE_SUCCESS")
PY
  "$python_bin" -m reproduction.stage2n.run_unit_tests --output "$log_dir/unit-tests.json"
  mark_stage preflight
  "$python_bin" -m reproduction.stage2m.preflight \
    --repo-root "$repo_root" --base-model "$base_model" --split-audit "$split_audit" \
    --output "$output_dir/preflight.json"
  trace="$output_dir/$arm/trace.jsonl"
  midpoint="$output_dir/$arm/checkpoints/mid-block12.pt"
  final="$output_dir/$arm/checkpoints/final-block24.pt"
  mark_stage phase1_blocks_1_12
  "$python_bin" -m reproduction.stage2m.train_scout \
    --arm "$arm" --phase phase1 --run-root "$output_dir" --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" --split-audit "$split_audit" \
    --trace "$trace" --checkpoint-output "$midpoint"
  assert_gpu_idle "$log_dir/post-phase1-compute-apps.csv"
  mark_stage phase2_fresh_resume_blocks_13_24
  "$python_bin" -m reproduction.stage2m.train_scout \
    --arm "$arm" --phase phase2 --run-root "$output_dir" --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" --split-audit "$split_audit" \
    --trace "$trace" --resume-checkpoint "$midpoint" --checkpoint-output "$final"
  assert_gpu_idle "$log_dir/post-phase2-compute-apps.csv"
  mark_stage fixed_block24_heldout_evaluation
  "$python_bin" -m reproduction.stage2m.evaluate_scout \
    --arm "$arm" --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --split-audit "$split_audit" --checkpoint "$final" --expected-cursor 24 \
    --repo-root "$repo_root" --output-dir "$output_dir/evaluation/$arm"
  assert_gpu_idle "$log_dir/post-eval-compute-apps.csv"
  mark_stage independent_summary
  "$python_bin" -m reproduction.stage2n.summarize_adjustment \
    --run-root "$output_dir" --split-audit "$split_audit" --original-root "$original_root" \
    --registered-comparison "$registered_comparison" --output "$output_dir/comparison.json"
  terminal_decision="$($python_bin -c 'import json; print(json.load(open("/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-stability-repair-moment-reset-v1/comparison.json"))["decision"])')"
  case "$terminal_decision" in A|B|C|D) ;; *) return 21 ;; esac
  printf '%s\n' "$terminal_decision" > "$output_dir/terminal-decision.txt"
  mark_stage complete
  inventory="$output_dir/artifact-inventory.sha256"
  find "$log_dir" "$output_dir" -type f \
    ! -path "$inventory" ! -path "$run_log" ! -path "$run_status" \
    ! -path "$output_dir/artifact-inventory-check.log" \
    -print0 | sort -z | xargs -0 sha256sum > "$inventory"
  sha256sum -c "$inventory" > "$output_dir/artifact-inventory-check.log"
}

set +e
( set -euo pipefail; run_pipeline ) 2>&1 | tee "$run_log"
pipeline_status=${PIPESTATUS[0]}
set -e
if [ "$pipeline_status" -ne 0 ]; then
  printf 'INVALID_QUARANTINED:%s:%s\n' "$(cat "$active_stage" 2>/dev/null || printf unknown)" "$pipeline_status" > "$run_status"
  exit "$pipeline_status"
fi
if [ "$(cat "$active_stage")" != complete ]; then
  printf 'INVALID_QUARANTINED:completion_marker:5\n' > "$run_status"
  exit 5
fi
set +e
sha256sum -c "$output_dir/artifact-inventory.sha256" > "$output_dir/post-wrapper-inventory-check.log"
inventory_status=$?
set -e
if [ "$inventory_status" -ne 0 ]; then
  printf 'INVALID_QUARANTINED:post_wrapper_inventory:%s\n' "$inventory_status" > "$run_status"
  exit "$inventory_status"
fi
cat "$output_dir/terminal-decision.txt" > "$run_status"
