#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "CABG-LCE-MIL v1.2 D4 pilot v1 uses locked paths and accepts no overrides." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/envs/medic-ad-train/bin/python"
base_model="/home/checkpoints/Lingshu-7B"
source_data="/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2"
image_root="/home/data/medic-ad/official/med_anomaly"
d3r_manifest="/home/data/medic-ad/cabg-lce-mil-v1.2-d3r-v1/manifest.jsonl"
data_dir="/home/data/medic-ad/cabg-lce-mil-v1.2-d4-pilot-v2"
output_dir="/home/Medic-AD/outputs/cabg-mil-v1.2/d4-pilot-v2"
log_dir="/home/Medic-AD/logs/cabg-mil-v1.2/d4-pilot-v2"
training_development="$source_data/training-development-manifest.jsonl"
packet_gate="$repo_root/reproduction/stage2l/packet-gate.json"
trace="$output_dir/block-trace.jsonl"
phase1_checkpoint="$output_dir/checkpoints/phase1-block2.pt"
final_checkpoint="$output_dir/checkpoints/final-block4.pt"
run_log="$log_dir/d4-pilot-v1.log"
run_status="$log_dir/run.status"
active_stage="$log_dir/active-stage.txt"

export PYTHONPATH="$repo_root:$repo_root/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
for path in "$python_bin" "$base_model" "$image_root" "$training_development" "$d3r_manifest" "$packet_gate"; do
  if [ ! -e "$path" ]; then
    echo "missing D4 pilot prerequisite: $path" >&2
    exit 3
  fi
done
for path in "$data_dir" "$output_dir" "$log_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing D4 pilot root: $path" >&2
    exit 4
  fi
done
mkdir -p "$output_dir/checkpoints" "$log_dir"

mark_stage() {
  printf '%s\n' "$1" > "$active_stage"
  printf 'D4_PILOT_V1_STAGE=%s\n' "$1"
}

run_pipeline() {
  cd "$repo_root"
  mark_stage packet_gate
  "$python_bin" - <<'PY' > "$log_dir/packet-gate-check.json"
import json
from pathlib import Path
d=json.loads(Path("reproduction/stage2l/packet-gate.json").read_text())
assert d["status"] == "PASS"
assert set(d["gates"]) == {"spec_review","quality_review","unit_tests","data_isolation","source_model_identity","independent_verification"}
assert all(value == "PASS" for value in d["gates"].values())
print(json.dumps(d, indent=2, sort_keys=True))
PY
  mark_stage import_gate
  "$python_bin" - <<'PY' > "$log_dir/import-gate.txt"
import qwenvl
import reproduction.stage2h.runtime
import reproduction.stage2l.controller
import reproduction.stage2l.checkpoint
import reproduction.stage2l.run_pilot
import reproduction.stage2l.verify_pilot
print("D4_PILOT_IMPORT_GATE_SUCCESS")
PY
  mark_stage unit_tests
  "$python_bin" -m reproduction.stage2l.run_unit_tests --output "$log_dir/unit-tests.json"
  mark_stage dataset_lock
  "$python_bin" -m reproduction.stage2l.build_dataset \
    --training-development "$training_development" \
    --d3r-manifest "$d3r_manifest" \
    --image-root "$image_root" \
    --output-dir "$data_dir"
  mark_stage packet_independent_verification
  "$python_bin" -m reproduction.stage2l.verify_packet \
    --repo-root "$repo_root" --unit-tests "$log_dir/unit-tests.json" \
    --dataset-audit "$data_dir/dataset-audit.json" --manifest "$data_dir/manifest.jsonl" \
    --output "$log_dir/packet-verification.json"
  mark_stage independent_preflight
  "$python_bin" -m reproduction.stage2l.preflight \
    --repo-root "$repo_root" --base-model "$base_model" \
    --dataset-audit "$data_dir/dataset-audit.json" \
    --packet-gate "$packet_gate" --output "$output_dir/preflight.json"
  mark_stage phase1_blocks_0_1
  "$python_bin" -m reproduction.stage2l.run_pilot \
    --phase phase1 --run-root "$output_dir" --repo-root "$repo_root" \
    --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --dataset-audit "$data_dir/dataset-audit.json" --trace "$trace" \
    --checkpoint-output "$phase1_checkpoint"
  mark_stage phase1_process_exited
  test -f "$phase1_checkpoint" && test -f "$phase1_checkpoint.manifest.json"
  mark_stage phase2_fresh_process_resume_blocks_2_3
  "$python_bin" -m reproduction.stage2l.run_pilot \
    --phase phase2 --run-root "$output_dir" --repo-root "$repo_root" \
    --base-model "$base_model" --preflight "$output_dir/preflight.json" \
    --dataset-audit "$data_dir/dataset-audit.json" --trace "$trace" \
    --resume-checkpoint "$phase1_checkpoint" --checkpoint-output "$final_checkpoint"
  mark_stage gpu_idle_before_independent_evaluator
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$log_dir/pre-evaluator-compute-apps.csv"
  if [ -s "$log_dir/pre-evaluator-compute-apps.csv" ]; then
    echo "GPU compute process remained before independent evaluator" >&2
    return 20
  fi
  mark_stage independent_evaluator
  "$python_bin" -m reproduction.stage2l.verify_pilot \
    --run-root "$output_dir" --trace "$trace" \
    --phase1-checkpoint "$phase1_checkpoint" --final-checkpoint "$final_checkpoint" \
    --preflight "$output_dir/preflight.json" --dataset-audit "$data_dir/dataset-audit.json" \
    --manifest "$data_dir/manifest.jsonl" --training-development "$training_development" \
    --d3r-manifest "$d3r_manifest" --output "$output_dir/verification.json"
  terminal_decision="$($python_bin -c 'import json; print(json.load(open("/home/Medic-AD/outputs/cabg-mil-v1.2/d4-pilot-v2/verification.json"))["decision"])')"
  case "$terminal_decision" in PASS_D4_PILOT|PAUSE|BLOCKED) ;; *) return 21 ;; esac
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
