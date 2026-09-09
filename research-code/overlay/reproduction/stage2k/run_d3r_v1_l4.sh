#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "CABG-LCE-MIL v1.2 D3R v1 uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
source_data="/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2"
approved_preflight="/home/Medic-AD/outputs/cabg-mil-v1.1/gate-d3-v1/preflight.json"
image_root="/home/data/medic-ad/official/med_anomaly"
data_dir="/home/data/medic-ad/cabg-lce-mil-v1.2-d3r-v1"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/d3r-v1"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/d3r-v1"
run_log="$log_dir/d3r-v1.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"
post_wrapper_inventory_check="$output_dir/post-wrapper-inventory-check.log"

training_development="$source_data/training-development-manifest.jsonl"
threshold_validation="$source_data/threshold-validation-manifest.jsonl"
old_d3="$source_data/mechanism-diagnostic-manifest.jsonl"
stage2h_excluded="$source_data/excluded-stage2h-manifest.jsonl"

export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"

for path in "$python_bin" "$base_model" "$approved_preflight" "$image_root" "$training_development" "$threshold_validation" "$old_d3" "$stage2h_excluded" "$repo_root/reproduction/stage2k/packet-gate.json"; do
  if [ ! -e "$path" ]; then
    echo "missing D3R prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$data_dir" "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing D3R v1 root: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir" "$log_dir"

mark_stage() {
  printf '%s\n' "$1" > "$active_stage"
  printf 'D3R_V1_STAGE=%s\n' "$1"
}

run_pipeline() {
  cd "$repo_root"

  mark_stage packet_gate
  "$python_bin" - <<'PY' > "$log_dir/packet-gate-check.json"
import json
from pathlib import Path
p=Path("reproduction/stage2k/packet-gate.json")
d=json.loads(p.read_text())
assert d["status"] == "PASS"
assert d["spec_review"] == "PASS"
assert d["quality_review"] == "PASS"
assert d["tests"] == "PASS"
assert d["dataset_isolation"] == "PASS"
assert d["independent_verification"] == "PASS"
assert d["manifest_sha256"] == "7e581453de5f4abba2eafa87b982eccb751264f66e86ac10f3b92a160ae20db1"
print(json.dumps(d, indent=2, sort_keys=True))
PY

  mark_stage import_gate
  "$python_bin" - <<'PY' > "$log_dir/import-gate.txt"
import qwenvl
import reproduction.stage2h.runtime
import reproduction.stage2k.manifest
import reproduction.stage2k.build_dataset
import reproduction.stage2k.producer_math
import reproduction.stage2k.run_d3r
print("D3R_IMPORT_GATE_SUCCESS")
PY

  mark_stage unit_tests
  "$python_bin" -m reproduction.stage2k.run_unit_tests \
    --repo-root "$repo_root" \
    --output "$log_dir/unit-tests.json"

  mark_stage manifest_lock
  "$python_bin" -m reproduction.stage2k.manifest \
    --training-development "$training_development" \
    --threshold-validation "$threshold_validation" \
    --old-d3 "$old_d3" \
    --stage2h-excluded "$stage2h_excluded" \
    --output-dir "$data_dir"

  mark_stage manifest_independent_verification
  "$python_bin" -m reproduction.stage2k.verify_manifest \
    --training-development "$training_development" \
    --threshold-validation "$threshold_validation" \
    --old-d3 "$old_d3" \
    --stage2h-excluded "$stage2h_excluded" \
    --manifest "$data_dir/manifest.jsonl" \
    --dataset-audit "$data_dir/dataset-audit.json" \
    --exclusion-audit "$data_dir/exclusion-audit.json" \
    --output "$data_dir/manifest-verification.json"

  mark_stage selected_image_byte_verification
  "$python_bin" -m reproduction.stage2k.build_dataset \
    --manifest "$data_dir/manifest.jsonl" \
    --image-root "$image_root" \
    --output-dir "$data_dir/runtime"

  mark_stage real_model_diagnostic
  "$python_bin" -m reproduction.stage2k.run_d3r \
    --repo-root "$repo_root" \
    --base-model "$base_model" \
    --approved-preflight "$approved_preflight" \
    --dataset-audit "$data_dir/runtime/dataset-audit.json" \
    --records-output "$output_dir/records.jsonl" \
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
  "$python_bin" -m reproduction.stage2k.verify_d3r \
    --records "$output_dir/records.jsonl" \
    --manifest "$data_dir/manifest.jsonl" \
    --dataset-audit "$data_dir/runtime/dataset-audit.json" \
    --exclusion-audit "$data_dir/exclusion-audit.json" \
    --training-development "$training_development" \
    --threshold-validation "$threshold_validation" \
    --old-d3 "$old_d3" \
    --stage2h-excluded "$stage2h_excluded" \
    --model-load-audit "$output_dir/model-load-audit.json" \
    --state-before "$output_dir/parameter-state-before.json" \
    --state-after "$output_dir/parameter-state-after.json" \
    --file-open-audit "$output_dir/file-open-audit.json" \
    --rng-audit "$output_dir/rng-audit.json" \
    --monitor-summary "$output_dir/nvidia-smi-peak.json" \
    --post-run-gpu "$log_dir/post-run-gpu.csv" \
    --post-run-compute-apps "$log_dir/post-run-compute-apps.csv" \
    --source-inventory "$output_dir/source-inventory.json" \
    --repo-root "$repo_root" \
    --base-model "$base_model" \
    --aggregate-output "$output_dir/aggregate.json" \
    --output "$output_dir/verification.json"

  mark_stage terminal_decision_check
  terminal_decision="$($python_bin -c 'import json; print(json.load(open("/home/Medic-AD/outputs/cabg-mil-v1.2/d3r-v1/verification.json"))["decision"])')"
  case "$terminal_decision" in
    PASS_D3R|PAUSE) ;;
    *) echo "invalid D3R terminal decision: $terminal_decision" >&2; return 6 ;;
  esac
  printf '%s\n' "$terminal_decision" > "$output_dir/terminal-decision.txt"

  mark_stage artifact_inventory
  inventory="$output_dir/artifact-inventory.sha256"
  inventory_check="$output_dir/artifact-inventory-check.log"
  mark_stage complete
  find "$data_dir" "$log_dir" "$output_dir" -type f \
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
  printf 'INVALID_QUARANTINED:%s:%s\n' "$failed_stage" "$pipeline_status" > "$run_status"
  exit "$pipeline_status"
fi

if [ "$(cat "$active_stage")" != "complete" ]; then
  printf '%s\n' 'INVALID_QUARANTINED:completion_marker:5' > "$run_status"
  exit 5
fi

set +e
sha256sum -c "$output_dir/artifact-inventory.sha256" > "$post_wrapper_inventory_check"
post_wrapper_inventory_status=$?
set -e
if [ "$post_wrapper_inventory_status" -ne 0 ]; then
  printf 'INVALID_QUARANTINED:post_wrapper_inventory:%s\n' "$post_wrapper_inventory_status" > "$run_status"
  exit "$post_wrapper_inventory_status"
fi
terminal_decision="$(cat "$output_dir/terminal-decision.txt")"
printf '%s\n' "$terminal_decision" > "$run_status"
