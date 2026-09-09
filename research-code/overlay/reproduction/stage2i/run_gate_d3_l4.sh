#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Gate D3 v1 uses packet-locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
control_root="/home/Medic-AD/project-control/cabg-mil-v1.1/gate-d3-v1"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.1/gate-d3-v1"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.1/gate-d3-v1"
dataset_dir="/home/data/medic-ad/cabg-mil-v1.1-gate-d3-v1"
d2_dataset_dir="/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2"
base_model="/home/checkpoints/Lingshu-7B"
image_root="/home/data/medic-ad/official/med_anomaly"
protocol="$control_root/experiment-protocol.md"
packet="$control_root/gate-d3-execution-v1.md"
packet_lock="$control_root/gate-d3-execution-v1.lock.json"
d2_verification="/home/Medic-AD/outputs/cabg-mil-v1.1/gate-d2-final-v2/gate-d2-verification.json"
run_log="$log_dir/gate-d3.log"
run_status="$log_dir/run.status"

for path in "$log_dir" "$output_dir" "$dataset_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing Gate D3 path: $path" >&2
    exit 3
  fi
done
for path in "$python_bin" "$protocol" "$packet" "$packet_lock" "$d2_verification"; do
  if [ ! -e "$path" ]; then
    echo "missing locked Gate D3 prerequisite: $path" >&2
    exit 4
  fi
done

mkdir -p "$log_dir" "$output_dir"

run_pipeline() {
  cd "$repo_root"
  "$python_bin" -m reproduction.stage2i.run_unit_tests \
    --repo-root "$repo_root" \
    --output "$log_dir/unit-tests.json"

  "$python_bin" -m reproduction.stage2i.build_gate_d3_dataset \
    --manifest "$d2_dataset_dir/mechanism-diagnostic-manifest.jsonl" \
    --image-root "$image_root" \
    --output-dir "$dataset_dir"

  "$python_bin" -m reproduction.stage2i.gate_d3_preflight \
    --repo-root "$repo_root" \
    --protocol "$protocol" \
    --packet "$packet" \
    --packet-lock "$packet_lock" \
    --d2-verification "$d2_verification" \
    --unit-test-result "$log_dir/unit-tests.json" \
    --dataset-audit "$dataset_dir/dataset-audit.json" \
    --base-model "$base_model" \
    --source-inventory-output "$output_dir/source-inventory.json" \
    --environment-output "$output_dir/environment.json" \
    --output "$output_dir/preflight.json"

  "$python_bin" -m reproduction.stage2i.run_gate_d3 \
    --base-model "$base_model" \
    --preflight "$output_dir/preflight.json" \
    --dataset-audit "$dataset_dir/dataset-audit.json" \
    --records-output "$output_dir/mechanism-records.jsonl" \
    --aggregate-output "$output_dir/class-aggregate.json" \
    --model-load-audit-output "$output_dir/model-load-audit.json" \
    --state-before-output "$output_dir/parameter-state-before.json" \
    --state-after-output "$output_dir/parameter-state-after.json" \
    --file-open-audit-output "$output_dir/file-open-audit.json" \
    --rng-audit-output "$output_dir/rng-audit.json" \
    --monitor-output "$log_dir/nvidia-smi.csv" \
    --monitor-summary-output "$output_dir/nvidia-smi-peak.json" \
    --phase-markers-output "$log_dir/phase-markers.jsonl"

  nvidia-smi \
    --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "$log_dir/post-run-gpu.csv"

  inventory="$output_dir/artifact-inventory.sha256"
  inventory_check="$output_dir/artifact-inventory-check.log"
  verification="$output_dir/gate-d3-verification.json"
  find "$log_dir" "$output_dir" "$dataset_dir" -type f \
    ! -path "$inventory" \
    ! -path "$inventory_check" \
    ! -path "$verification" \
    ! -path "$run_status" \
    ! -path "$run_log" \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$inventory"
  sha256sum -c "$inventory" > "$inventory_check"

  "$python_bin" -m reproduction.stage2i.verify_gate_d3 \
    --preflight "$output_dir/preflight.json" \
    --dataset-audit "$dataset_dir/dataset-audit.json" \
    --records "$output_dir/mechanism-records.jsonl" \
    --aggregate "$output_dir/class-aggregate.json" \
    --model-load-audit "$output_dir/model-load-audit.json" \
    --state-before "$output_dir/parameter-state-before.json" \
    --state-after "$output_dir/parameter-state-after.json" \
    --file-open-audit "$output_dir/file-open-audit.json" \
    --rng-audit "$output_dir/rng-audit.json" \
    --monitor-summary "$output_dir/nvidia-smi-peak.json" \
    --post-run-gpu "$log_dir/post-run-gpu.csv" \
    --inventory "$inventory" \
    --inventory-check "$inventory_check" \
    --run-status "$run_status" \
    --output "$verification"

  decision="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$verification")"
  if [ "$decision" = "PASS_GATE_D3" ]; then
    printf '%s\n' 'PASS_GATE_D3' > "$run_status"
  elif [ "$decision" = "PAUSE" ]; then
    printf '%s\n' 'PAUSE' > "$run_status"
  else
    echo "unexpected Gate D3 verifier decision: $decision" >&2
    return 5
  fi
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

exit 0
