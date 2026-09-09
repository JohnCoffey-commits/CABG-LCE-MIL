#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "D4-Scout causal branch v1 uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
split_audit="/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json"
original_root="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-v1"
midpoint_root="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-midpoint-supplement-v1"
parent_checkpoint="$original_root/cabg_lce/checkpoints/mid-block12.pt"
causal_run_version="${CAUSAL_RUN_VERSION:-v1}"
if [[ ! "$causal_run_version" =~ ^v[0-9]+$ ]]; then
  echo "invalid causal run version: $causal_run_version" >&2
  exit 2
fi
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-causal-branch-$causal_run_version"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/d4-scout-causal-branch-$causal_run_version"
run_log="$log_dir/run.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"
control_primary="$output_dir/control-primary"
control_repeat="$output_dir/control-repeat"
causal_root="$output_dir/causal"

export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
for path in "$python_bin" "$base_model" "$split_audit" "$parent_checkpoint" "$parent_checkpoint.manifest.json" "$original_root/comparison.json" "$original_root/artifact-inventory.sha256" "$midpoint_root/artifact-inventory.sha256"; do
  if [ ! -e "$path" ]; then
    echo "missing causal-continuation prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite causal-continuation root: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir" "$log_dir"

mark_stage() {
  printf '%s\n' "$1" > "$active_stage"
  printf 'D4_SCOUT_CAUSAL_STAGE=%s\n' "$1"
}

assert_gpu_idle() {
  local output_file="$1"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$output_file"
  if [ -s "$output_file" ]; then
    echo "GPU compute process remained at $output_file" >&2
    return 20
  fi
}

run_branch() {
  local root="$1"
  local arm="$2"
  local arm_dir="$root/$arm"
  local final="$arm_dir/checkpoints/final-block24.pt"
  mkdir -p "$arm_dir/checkpoints"
  mark_stage "${arm}_restore_and_blocks_13_24"
  "$python_bin" -m reproduction.stage2m.train_scout \
    --arm "$arm" --phase phase2 --run-root "$root" --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" --split-audit "$split_audit" \
    --trace "$arm_dir/trace.jsonl" --resume-checkpoint "$parent_checkpoint" \
    --checkpoint-output "$final" --parent-audit "$output_dir/parent-audit.json"
  assert_gpu_idle "$log_dir/${arm}-$(basename "$root")-post-train-compute-apps.csv"
  mark_stage "${arm}_fixed_block24_heldout_evaluation"
  "$python_bin" -m reproduction.stage2m.evaluate_scout \
    --arm "$arm" --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --split-audit "$split_audit" --checkpoint "$final" --expected-cursor 24 \
    --repo-root "$repo_root" --output-dir "$root/evaluation/$arm"
  assert_gpu_idle "$log_dir/${arm}-$(basename "$root")-post-eval-compute-apps.csv"
}

run_pipeline() {
  cd "$repo_root"
  mark_stage prerequisite_rehash
  sha256sum -c "$original_root/artifact-inventory.sha256" > "$log_dir/original-scout-inventory-check.log"
  sha256sum -c "$midpoint_root/artifact-inventory.sha256" > "$log_dir/original-midpoint-inventory-check.log"
  assert_gpu_idle "$log_dir/pre-execution-compute-apps.csv"
  mark_stage import_and_tests
  "$python_bin" - <<'PY' > "$log_dir/import-gate.txt"
import qwenvl
import reproduction.stage2m.train_scout
import reproduction.stage2m.evaluate_scout
import reproduction.stage2o.audit_parent
import reproduction.stage2o.summarize_causal
print("D4_SCOUT_CAUSAL_IMPORT_GATE_SUCCESS")
PY
  "$python_bin" -m reproduction.stage2o.run_unit_tests --output "$log_dir/unit-tests.json"
  mark_stage preflight
  "$python_bin" -m reproduction.stage2m.preflight \
    --repo-root "$repo_root" --base-model "$base_model" --split-audit "$split_audit" \
    --output "$output_dir/preflight.json"
  mark_stage parent_checkpoint_audit
  "$python_bin" -m reproduction.stage2o.audit_parent \
    --checkpoint "$parent_checkpoint" --split-audit "$split_audit" \
    --original-root "$original_root" --midpoint-root "$midpoint_root" \
    --output "$output_dir/parent-audit.json"

  run_branch "$control_primary" cabg_continue_control
  mark_stage control_primary_independent_check
  "$python_bin" -m reproduction.stage2o.summarize_causal control-check \
    --branch-root "$control_primary" --parent-audit "$output_dir/parent-audit.json" \
    --split-audit "$split_audit" --original-root "$original_root" \
    --output "$output_dir/control-primary-check.json"
  primary_reproduced="$(jq -r '.reproduction_phenotype' "$output_dir/control-primary-check.json")"
  repeat_arg=()
  run_causal=false
  if [ "$primary_reproduced" = true ]; then
    run_causal=true
  else
    run_branch "$control_repeat" cabg_continue_control
    mark_stage control_repeat_independent_check
    "$python_bin" -m reproduction.stage2o.summarize_causal control-check \
      --branch-root "$control_repeat" --parent-audit "$output_dir/parent-audit.json" \
      --split-audit "$split_audit" --original-root "$original_root" \
      --output "$output_dir/control-repeat-check.json"
    repeat_arg=(--control-repeat "$control_repeat")
    if [ "$(jq -r '.reproduction_phenotype' "$output_dir/control-repeat-check.json")" = true ]; then
      run_causal=true
    fi
  fi
  causal_arg=()
  if [ "$run_causal" = true ]; then
    run_branch "$causal_root" lce_off_state_kept
    run_branch "$causal_root" lce_off_s9_exp_avg_reset
    causal_arg=(--causal-root "$causal_root")
  fi
  mark_stage final_independent_summary
  "$python_bin" -m reproduction.stage2o.summarize_causal final \
    --control-primary "$control_primary" "${repeat_arg[@]}" "${causal_arg[@]}" \
    --parent-audit "$output_dir/parent-audit.json" --split-audit "$split_audit" \
    --original-root "$original_root" --output "$output_dir/comparison.json"
  terminal_decision="$(jq -r '.decision' "$output_dir/comparison.json")"
  case "$terminal_decision" in
    CONTROLLER_REPAIR_ELIGIBLE|OPTIMIZER_REPAIR_ELIGIBLE|REDESIGN_REQUIRED|INCONCLUSIVE_REPRODUCIBILITY) ;;
    *) return 21 ;;
  esac
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
