#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "CABG-MIL v1.2 R0 uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
d3_output="/home/Medic-AD/outputs/cabg-mil-v1.1/gate-d3-v1"
d3_log="/home/Medic-AD/logs/cabg-mil-v1.1/gate-d3-v1"
dataset_audit="/home/data/medic-ad/cabg-mil-v1.1-gate-d3-v1/dataset-audit.json"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/redesign-r0-v1"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/redesign-r0-v1"
run_log="$log_dir/redesign-r0.log"
run_status="$log_dir/run.status"

for path in "$python_bin" "$base_model" "$d3_output/preflight.json" "$d3_output/gate-d3-verification.json" "$dataset_audit"; do
  if [ ! -e "$path" ]; then
    echo "missing R0 prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$output_dir/r0-records.jsonl" "$output_dir/r0-aggregate.json" "$output_dir/r0-verification.json" "$log_dir/unit-tests.json" "$run_log" "$run_status"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing R0 artifact: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir" "$log_dir"

run_pipeline() {
  cd "$repo_root"
  "$python_bin" -m reproduction.stage2j.run_unit_tests \
    --repo-root "$repo_root" \
    --output "$log_dir/unit-tests.json"

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

  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits > "$log_dir/post-run-gpu.csv"

  "$python_bin" -m reproduction.stage2j.verify_redesign_r0 \
    --records "$output_dir/r0-records.jsonl" \
    --model-load-audit "$output_dir/model-load-audit.json" \
    --state-before "$output_dir/parameter-state-before.json" \
    --state-after "$output_dir/parameter-state-after.json" \
    --file-open-audit "$output_dir/file-open-audit.json" \
    --rng-audit "$output_dir/rng-audit.json" \
    --monitor-summary "$output_dir/nvidia-smi-peak.json" \
    --post-run-gpu "$log_dir/post-run-gpu.csv" \
    --source-inventory "$output_dir/source-inventory.json" \
    --dataset-audit "$dataset_audit" \
    --aggregate-output "$output_dir/r0-aggregate.json" \
    --output "$output_dir/r0-verification.json"

  inventory="$output_dir/artifact-inventory.sha256"
  find "$log_dir" "$output_dir" -type f \
    ! -path "$inventory" \
    ! -path "$run_log" \
    ! -path "$run_status" \
    -print0 | sort -z | xargs -0 sha256sum > "$inventory"
  sha256sum -c "$inventory" > "$output_dir/artifact-inventory-check.log"
  printf '%s\n' 'SUCCESS' > "$run_status"
}

set +e
run_pipeline 2>&1 | tee "$run_log"
pipeline_status=${PIPESTATUS[0]}
set -e
if [ "$pipeline_status" -ne 0 ]; then
  if [ ! -e "$run_status" ]; then
    printf 'TERMINAL_BLOCKER:%s\n' "$pipeline_status" > "$run_status"
  fi
  exit "$pipeline_status"
fi
