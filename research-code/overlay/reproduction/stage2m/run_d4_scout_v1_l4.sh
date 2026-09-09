#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "D4-Scout v1 uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
source_data="/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2"
image_root="/home/data/medic-ad/official/med_anomaly"
training_development="$source_data/training-development-manifest.jsonl"
d3r_manifest="/home/data/medic-ad/cabg-lce-mil-v1.2-d3r-v1/manifest.jsonl"
d4_pilot_manifest="/home/data/medic-ad/cabg-lce-mil-v1.2-d4-pilot-v2/manifest.jsonl"
data_dir="/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-v1"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/d4-scout-v1"
run_log="$log_dir/d4-scout-v1.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"

export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
for path in "$python_bin" "$base_model" "$image_root" "$training_development" "$d3r_manifest" "$d4_pilot_manifest"; do
  if [ ! -e "$path" ]; then
    echo "missing D4-Scout prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$data_dir" "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing D4-Scout root: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir" "$log_dir"

mark_stage() {
  printf '%s\n' "$1" > "$active_stage"
  printf 'D4_SCOUT_V1_STAGE=%s\n' "$1"
}

assert_gpu_idle() {
  local output_file="$1"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$output_file"
  if [ -s "$output_file" ]; then
    echo "GPU compute process remained at $output_file" >&2
    return 20
  fi
}

train_arm() {
  local arm="$1"
  local arm_dir="$output_dir/$arm"
  local trace="$arm_dir/trace.jsonl"
  local midpoint="$arm_dir/checkpoints/mid-block12.pt"
  local final="$arm_dir/checkpoints/final-block24.pt"
  mkdir -p "$arm_dir/checkpoints"
  mark_stage "${arm}_phase1_blocks_0_11"
  "$python_bin" -m reproduction.stage2m.train_scout \
    --arm "$arm" --phase phase1 --run-root "$output_dir" --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" --split-audit "$data_dir/split-audit.json" \
    --trace "$trace" --checkpoint-output "$midpoint"
  assert_gpu_idle "$log_dir/${arm}-post-phase1-compute-apps.csv"
  mark_stage "${arm}_phase2_fresh_resume_blocks_12_23"
  "$python_bin" -m reproduction.stage2m.train_scout \
    --arm "$arm" --phase phase2 --run-root "$output_dir" --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" --split-audit "$data_dir/split-audit.json" \
    --trace "$trace" --resume-checkpoint "$midpoint" --checkpoint-output "$final"
  assert_gpu_idle "$log_dir/${arm}-post-phase2-compute-apps.csv"
  mark_stage "${arm}_heldout_evaluation"
  "$python_bin" -m reproduction.stage2m.evaluate_scout \
    --arm "$arm" --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --split-audit "$data_dir/split-audit.json" --checkpoint "$final" \
    --output-dir "$output_dir/evaluation/$arm"
  assert_gpu_idle "$log_dir/${arm}-post-eval-compute-apps.csv"
}

run_pipeline() {
  cd "$repo_root"
  mark_stage import_gate
  "$python_bin" - <<'PY' > "$log_dir/import-gate.txt"
import qwenvl
import reproduction.stage2m.build_split
import reproduction.stage2m.train_scout
import reproduction.stage2m.evaluate_scout
import reproduction.stage2m.summarize_scout
print("D4_SCOUT_IMPORT_GATE_SUCCESS")
PY
  mark_stage targeted_unit_tests
  "$python_bin" -m reproduction.stage2m.run_unit_tests --output "$log_dir/unit-tests.json"
  mark_stage split_lock
  "$python_bin" -m reproduction.stage2m.build_split \
    --training-development "$training_development" --d3r-manifest "$d3r_manifest" \
    --d4-pilot-manifest "$d4_pilot_manifest" --image-root "$image_root" --output-dir "$data_dir"
  mark_stage preflight
  "$python_bin" -m reproduction.stage2m.preflight \
    --repo-root "$repo_root" --base-model "$base_model" --split-audit "$data_dir/split-audit.json" \
    --output "$output_dir/preflight.json"
  mark_stage initial_heldout_reference
  "$python_bin" -m reproduction.stage2m.evaluate_scout \
    --arm initial --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --split-audit "$data_dir/split-audit.json" --output-dir "$output_dir/evaluation/initial"
  assert_gpu_idle "$log_dir/initial-post-eval-compute-apps.csv"
  train_arm cabg_lce
  train_arm lm_only
  mark_stage independent_summary_and_fairness
  "$python_bin" -m reproduction.stage2m.summarize_scout \
    --run-root "$output_dir" --split-audit "$data_dir/split-audit.json" \
    --initial-dir "$output_dir/evaluation/initial" --cabg-dir "$output_dir/evaluation/cabg_lce" \
    --baseline-dir "$output_dir/evaluation/lm_only" --output "$output_dir/comparison.json" \
    --table "$output_dir/comparison.csv"
  terminal_decision="$($python_bin -c 'import json; print(json.load(open("/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-v1/comparison.json"))["decision"])')"
  case "$terminal_decision" in A|B|C|D) ;; *) return 21 ;; esac
  printf '%s\n' "$terminal_decision" > "$output_dir/terminal-decision.txt"
  mark_stage complete
  inventory="$output_dir/artifact-inventory.sha256"
  find "$data_dir" "$log_dir" "$output_dir" -type f \
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
