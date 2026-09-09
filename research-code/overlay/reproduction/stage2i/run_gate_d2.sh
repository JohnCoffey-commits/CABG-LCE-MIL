#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 REPO_ROOT PYTHON LOG_DIR OUTPUT_DIR DATASET_OUTPUT_DIR PROTOCOL PROTOCOL_LOCK" >&2
  exit 2
fi

repo_root="$1"
python_bin="$2"
log_dir="$3"
output_dir="$4"
dataset_output_dir="$5"
protocol_path="$6"
protocol_lock_path="$7"
stage2h_root="/home/data/medic-ad/stage2h-lad-mil-v2-2-v1"

for path in "$log_dir" "$output_dir" "$dataset_output_dir"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing Gate D2 path: $path" >&2
    exit 3
  fi
done

mkdir -p "$log_dir" "$output_dir"

run_gate_d2() {
  cd "$repo_root"
  "$python_bin" -m reproduction.stage2i.run_unit_tests \
    --repo-root "$repo_root" \
    --output "$log_dir/unit-tests.json"

  "$python_bin" -m reproduction.stage2i.build_dataset_manifest \
    --source-manifest "$repo_root/reproduction/stage2g/data/evaluation-manifest.jsonl" \
    --stage2h-train-manifest "$stage2h_root/train-manifest.jsonl" \
    --internal-test-manifest "$stage2h_root/internal-test-manifest.jsonl" \
    --stage2h-audit "$stage2h_root/dataset-audit.json" \
    --output-dir "$dataset_output_dir"

  "$python_bin" -m reproduction.stage2i.verify_gate_d2 \
    --repo-root "$repo_root" \
    --protocol "$protocol_path" \
    --protocol-lock "$protocol_lock_path" \
    --unit-test-result "$log_dir/unit-tests.json" \
    --dataset-audit "$dataset_output_dir/dataset-audit.json" \
    --output "$output_dir/gate-d2-verification.json"
}

set +e
run_gate_d2 2>&1 | tee "$log_dir/gate-d2.log"
run_status=${PIPESTATUS[0]}
set -e
if [ "$run_status" -ne 0 ]; then
  printf 'FAILED:%s\n' "$run_status" > "$log_dir/run.status"
  exit "$run_status"
fi

printf '%s\n' 'SUCCESS' > "$log_dir/run.status"
inventory_path="$output_dir/artifact-inventory.sha256"
inventory_check_path="$output_dir/artifact-inventory-check.log"
find "$log_dir" "$output_dir" "$dataset_output_dir" -type f \
  ! -path "$inventory_path" \
  ! -path "$inventory_check_path" \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$inventory_path"
sha256sum -c "$inventory_path" > "$inventory_check_path"
sha256sum "$inventory_check_path" >> "$inventory_path"
sha256sum -c "$inventory_path"
