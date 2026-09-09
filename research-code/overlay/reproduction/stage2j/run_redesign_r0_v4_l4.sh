#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "CABG-MIL v1.2 R0 v4 uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
d3_output="/home/Medic-AD/outputs/cabg-mil-v1.1/gate-d3-v1"
d3_log="/home/Medic-AD/logs/cabg-mil-v1.1/gate-d3-v1"
dataset_audit="/home/data/medic-ad/cabg-mil-v1.1-gate-d3-v1/dataset-audit.json"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/redesign-r0-v4"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/redesign-r0-v4"
run_log="$log_dir/redesign-r0-v4.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"
post_wrapper_inventory_check="$output_dir/post-wrapper-inventory-check.log"

export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"

for path in "$python_bin" "$base_model" "$d3_output/preflight.json" "$d3_output/gate-d3-verification.json" "$dataset_audit"; do
  if [ ! -e "$path" ]; then
    echo "missing R0 v4 prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing R0 v4 root: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir" "$log_dir"

mark_stage() {
  printf '%s\n' "$1" > "$active_stage"
  printf 'R0_V4_STAGE=%s\n' "$1"
}

run_pipeline() {
  cd "$repo_root"

  mark_stage import_gate
  "$python_bin" - <<'PY' > "$log_dir/import-gate.txt"
import qwenvl
import reproduction.stage2h.runtime
import reproduction.stage2j.candidate_math
import reproduction.stage2j.run_redesign_r0
import reproduction.stage2j.verify_redesign_r0
print("IMPORT_GATE_SUCCESS")
PY
  cat "$log_dir/import-gate.txt"

  mark_stage unit_tests
  "$python_bin" -m reproduction.stage2j.run_unit_tests \
    --repo-root "$repo_root" \
    --output "$log_dir/unit-tests.json"

  mark_stage d3_independent_reverification
  "$python_bin" -m reproduction.stage2i.verify_gate_d3 \
    --preflight "$d3_output/preflight.json" \
    --dataset-audit "$dataset_audit" \
    --records "$d3_output/mechanism-records.jsonl" \
    --aggregate "$d3_output/class-aggregate.json" \
    --model-load-audit "$d3_output/model-load-audit.json" \
    --state-before "$d3_output/parameter-state-before.json" \
    --state-after "$d3_output/parameter-state-after.json" \
    --file-open-audit "$d3_output/file-open-audit.json" \
    --rng-audit "$d3_output/rng-audit.json" \
    --monitor-summary "$d3_output/nvidia-smi-peak.json" \
    --post-run-gpu "$d3_log/post-run-gpu.csv" \
    --inventory "$d3_output/artifact-inventory.sha256" \
    --inventory-check "$d3_output/artifact-inventory-check.log" \
    --run-status "$d3_log/run.status" \
    --output "$output_dir/d3-independent-reverification.json"

  mark_stage real_model_diagnostic
  "$python_bin" -m reproduction.stage2j.run_redesign_r0 \
    --repo-root "$repo_root" \
    --base-model "$base_model" \
    --d3-preflight "$d3_output/preflight.json" \
    --d3-verification "$d3_output/gate-d3-verification.json" \
    --dataset-audit "$dataset_audit" \
    --records-output "$output_dir/r0-records.jsonl" \
    --model-load-audit-output "$output_dir/model-load-audit.json" \
    --state-before-output "$output_dir/parameter-state-before.json" \
    --state-after-output "$output_dir/parameter-state-after.json" \
    --file-open-audit-output "$output_dir/file-open-audit.json" \
    --rng-audit-output "$output_dir/rng-audit.json" \
    --monitor-output "$log_dir/nvidia-smi.csv" \
    --monitor-summary-output "$output_dir/nvidia-smi-peak.json" \
    --phase-markers-output "$log_dir/phase-markers.jsonl" \
    --source-inventory-output "$output_dir/source-inventory.json"

  mark_stage post_run_gpu
  nvidia-smi \
    --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "$log_dir/post-run-gpu.csv"
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$log_dir/post-run-compute-apps.csv"

  mark_stage independent_evaluator
  "$python_bin" -m reproduction.stage2j.verify_redesign_r0 \
    --records "$output_dir/r0-records.jsonl" \
    --model-load-audit "$output_dir/model-load-audit.json" \
    --state-before "$output_dir/parameter-state-before.json" \
    --state-after "$output_dir/parameter-state-after.json" \
    --file-open-audit "$output_dir/file-open-audit.json" \
    --rng-audit "$output_dir/rng-audit.json" \
    --monitor-summary "$output_dir/nvidia-smi-peak.json" \
    --post-run-gpu "$log_dir/post-run-gpu.csv" \
    --post-run-compute-apps "$log_dir/post-run-compute-apps.csv" \
    --source-inventory "$output_dir/source-inventory.json" \
    --dataset-audit "$dataset_audit" \
    --aggregate-output "$output_dir/r0-aggregate.json" \
    --output "$output_dir/r0-verification.json"

  mark_stage artifact_inventory
  inventory="$output_dir/artifact-inventory.sha256"
  inventory_check="$output_dir/artifact-inventory-check.log"
  # Freeze the final marker before hashing.  No inventoried file may change
  # after this point; run_status, run_log, and the post-wrapper recheck are
  # explicit mutable/control files excluded below.
  mark_stage complete
  find "$log_dir" "$output_dir" -type f \
    ! -path "$inventory" \
    ! -path "$inventory_check" \
    ! -path "$run_log" \
    ! -path "$run_status" \
    ! -path "$post_wrapper_inventory_check" \
    -print0 | sort -z | xargs -0 sha256sum > "$inventory"
  sha256sum -c "$inventory" > "$inventory_check"
}

set +e
( set -euo pipefail; run_pipeline ) 2>&1 | tee "$run_log"
pipeline_status=${PIPESTATUS[0]}
set -e
if [ "$pipeline_status" -ne 0 ]; then
  failed_stage="$(cat "$active_stage" 2>/dev/null || printf '%s' unknown)"
  printf 'TERMINAL_BLOCKER:%s:%s\n' "$failed_stage" "$pipeline_status" > "$run_status"
  exit "$pipeline_status"
fi

if [ "$(cat "$active_stage")" != "complete" ]; then
  printf '%s\n' 'TERMINAL_BLOCKER:completion_marker:5' > "$run_status"
  exit 5
fi

# Recheck only after the piped pipeline has fully returned.  This catches any
# finalization-order mutation that an in-pipeline inventory check cannot see.
set +e
sha256sum -c "$output_dir/artifact-inventory.sha256" > "$post_wrapper_inventory_check"
post_wrapper_inventory_status=$?
set -e
if [ "$post_wrapper_inventory_status" -ne 0 ]; then
  printf 'TERMINAL_BLOCKER:post_wrapper_inventory:%s\n' "$post_wrapper_inventory_status" > "$run_status"
  exit "$post_wrapper_inventory_status"
fi
printf '%s\n' 'SUCCESS' > "$run_status"
