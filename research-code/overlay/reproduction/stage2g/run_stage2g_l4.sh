#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=/home/workspace/Medic-AD
readonly ADAPTER_SOURCE=/home/workspace/Medic-AD-stage2g-adapter
readonly R0_SOURCE=/home/workspace/Medic-AD-stage2g-r0
readonly LOG_ROOT=/home/logs/medic-ad/stage2g-fb-maq
readonly RUNS_ROOT="$LOG_ROOT/runs"
readonly AGGREGATE_ROOT="$LOG_ROOT/aggregate"
readonly DATA_ROOT=/home/data/medic-ad/official/med_anomaly
readonly EVALUATION_MANIFEST="$LOG_ROOT/data/evaluation-manifest.jsonl"
readonly DATASET_AUDIT="$LOG_ROOT/data/dataset-audit.json"
readonly REGISTRY="$LOG_ROOT/plan/run-registry.json"
readonly PREFLIGHT="$LOG_ROOT/preflight/preexperiment-ready.json"
readonly R0_CHECKPOINT_AUDIT="$LOG_ROOT/preflight/r0-checkpoint-audit.json"
readonly ADAPTER_PYTHON=/home/envs/medic-ad-train/bin/python
readonly R0_PYTHON=/home/envs/medic-ad/bin/python

mkdir -p "$RUNS_ROOT" "$AGGREGATE_ROOT"
if [[ -e "$LOG_ROOT/stage2g.status" ]]; then
  printf 'Refusing to overwrite Stage 2G execution evidence.\n' >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/orchestrator.log") 2>&1

current_run=''
on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'FAILED:%s:%s\n' "$rc" "$current_run" > "$LOG_ROOT/stage2g.status"
    if [[ -n "$current_run" && -d "$RUNS_ROOT/$current_run" ]]; then
      printf 'FAILED:%s\n' "$rc" > "$RUNS_ROOT/$current_run/run.status"
    fi
  fi
}
trap on_exit EXIT
printf 'RUNNING\n' > "$LOG_ROOT/stage2g.status"

"$ADAPTER_PYTHON" - "$PREFLIGHT" "$REGISTRY" "$DATASET_AUDIT" "$EVALUATION_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

preflight, registry, audit, manifest = map(Path, sys.argv[1:])
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
p = read(preflight)
r = read(registry)
if p.get("status") != "SUCCESS" or p.get("gate") != "PREEXPERIMENT_READY":
    raise SystemExit("preflight not ready")
if p.get("registry_sha256") != sha(registry):
    raise SystemExit("registry changed after preflight")
if p.get("dataset_audit_sha256") != sha(audit):
    raise SystemExit("dataset audit changed after preflight")
if p.get("evaluation_manifest_sha256") != sha(manifest):
    raise SystemExit("evaluation manifest changed after preflight")
if r.get("status") != "LOCKED" or len(r.get("runs", [])) != 13:
    raise SystemExit("registry invalid")
PY

readonly EVALUATION_MANIFEST_SHA256=$("$ADAPTER_PYTHON" -c 'import json; print(json.load(open("'"$DATASET_AUDIT"'"))["evaluation_manifest_sha256"])')

monitor_gpu() {
  local output=$1
  printf 'timestamp_utc,memory_used_mib\n' > "$output"
  while true; do
    printf '%s,' "$(date -u +%FT%T.%3NZ)" >> "$output"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$output"
    sleep 0.2
  done
}

json_field() {
  "$ADAPTER_PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}

run_registered() {
  local run_json=$1
  local run_id role method mode seed repeat train_run adapter adapter_manifest adapter_sha source_commit
  run_id=$(json_field "$run_json" run_id)
  role=$(json_field "$run_json" run_role)
  method=$(json_field "$run_json" method)
  current_run=$run_id
  local run_root="$RUNS_ROOT/$run_id"
  if [[ -e "$run_root" ]]; then
    printf 'Refusing to overwrite run directory: %s\n' "$run_root" >&2
    return 1
  fi
  mkdir -p "$run_root"
  printf 'RUNNING\n' > "$run_root/run.status"
  printf '%s\n' "$run_json" > "$run_root/registry-record.json"
  monitor_gpu "$run_root/peak-vram-samples.csv" &
  local monitor_pid=$!
  local started_ns
  started_ns=$(date +%s%N)
  set +e
  if [[ "$role" == 'published_reference' ]]; then
    (
      cd "$R0_SOURCE"
      PYTHONPATH="$R0_SOURCE:$R0_SOURCE/qwen-vl-finetune" "$R0_PYTHON" \
        "$REPO/reproduction/stage2g/evaluate_r0.py" \
        --model /home/checkpoints/MEDIC-AD \
        --checkpoint-audit "$R0_CHECKPOINT_AUDIT" \
        --data-root "$DATA_ROOT" \
        --evaluation-manifest "$EVALUATION_MANIFEST" \
        --expected-evaluation-manifest-sha256 "$EVALUATION_MANIFEST_SHA256" \
        --predictions "$run_root/predictions.jsonl" \
        --output "$run_root/result.json" \
        --run-id "$run_id" \
        --expected-source-commit ad62e7c910f4febad7b07030bd1c11796ae064e7 \
        --expected-model-revision 9374b660aade05e190471c5501fefac548982169
    ) > >(tee -a "$run_root/stdout-stderr.log") 2>&1
  else
    mode=$(json_field "$run_json" mode)
    seed=$(json_field "$run_json" seed)
    repeat=$(json_field "$run_json" repeat)
    train_run=$(json_field "$run_json" adapter_train_run_id)
    adapter=$(json_field "$run_json" adapter_path)
    adapter_manifest=$(json_field "$run_json" adapter_manifest_path)
    adapter_sha=$(json_field "$run_json" adapter_sha256)
    source_commit=$(json_field "$run_json" source_commit)
    (
      cd "$ADAPTER_SOURCE"
      PYTHONPATH="$ADAPTER_SOURCE:$ADAPTER_SOURCE/qwen-vl-finetune" "$ADAPTER_PYTHON" \
        "$REPO/reproduction/stage2g/evaluate_adapter.py" \
        --base-model /home/checkpoints/Lingshu-7B \
        --adapter "$adapter" \
        --adapter-manifest "$adapter_manifest" \
        --data-root "$DATA_ROOT" \
        --evaluation-manifest "$EVALUATION_MANIFEST" \
        --expected-evaluation-manifest-sha256 "$EVALUATION_MANIFEST_SHA256" \
        --predictions "$run_root/predictions.jsonl" \
        --output "$run_root/result.json" \
        --run-id "$run_id" \
        --adapter-train-run-id "$train_run" \
        --method "$method" \
        --mode "$mode" \
        --seed "$seed" \
        --repeat "$repeat" \
        --expected-adapter-sha256 "$adapter_sha" \
        --expected-source-commit "$source_commit"
    ) > >(tee -a "$run_root/stdout-stderr.log") 2>&1
  fi
  local eval_rc=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  if (( eval_rc != 0 )); then
    printf 'FAILED:%s\n' "$eval_rc" > "$run_root/run.status"
    return "$eval_rc"
  fi
  local ended_ns peak_mib
  ended_ns=$(date +%s%N)
  "$ADAPTER_PYTHON" -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "$started_ns" "$ended_ns" > "$run_root/wall-elapsed-seconds.txt"
  peak_mib=$(awk -F, 'NR > 1 && ($2 + 0) > max { max = $2 + 0 } END { print max + 0 }' "$run_root/peak-vram-samples.csv")
  printf '%s\n' "$peak_mib" > "$run_root/peak-vram-used-mib.txt"
  PYTHONPATH="$REPO:$REPO/qwen-vl-finetune" "$ADAPTER_PYTHON" \
    "$REPO/reproduction/stage2g/verify_run.py" \
    --registry "$REGISTRY" \
    --run-id "$run_id" \
    --evaluation-manifest "$EVALUATION_MANIFEST" \
    --result "$run_root/result.json" \
    --predictions "$run_root/predictions.jsonl" \
    --external-vram "$run_root/peak-vram-used-mib.txt" \
    --wall-elapsed "$run_root/wall-elapsed-seconds.txt" \
    --output "$run_root/run-verification.json" >> "$run_root/stdout-stderr.log" 2>&1
  printf 'SUCCESS\n' > "$run_root/run.status"
  printf '[%s] completed %s (%s)\n' "$(date -u +%FT%TZ)" "$run_id" "$method"
}

while IFS= read -r run_json; do
  run_registered "$run_json"
done < <("$ADAPTER_PYTHON" - "$REGISTRY" <<'PY'
import json
import sys
document = json.load(open(sys.argv[1]))
for run in sorted(document["runs"], key=lambda value: value["order"]):
    print(json.dumps(run, separators=(",", ":")))
PY
)

current_run='aggregate'
PYTHONPATH="$REPO:$REPO/qwen-vl-finetune" "$ADAPTER_PYTHON" \
  "$REPO/reproduction/stage2g/aggregate_pilot.py" \
  --registry "$REGISTRY" \
  --runs-root "$RUNS_ROOT" \
  --output "$AGGREGATE_ROOT/aggregate.json" \
  --transitions "$AGGREGATE_ROOT/paired-transitions.jsonl" \
  > >(tee -a "$AGGREGATE_ROOT/aggregate.log") 2>&1

PYTHONPATH="$REPO:$REPO/qwen-vl-finetune" "$ADAPTER_PYTHON" \
  "$REPO/reproduction/stage2g/verify_aggregate.py" \
  --registry "$REGISTRY" \
  --runs-root "$RUNS_ROOT" \
  --aggregate "$AGGREGATE_ROOT/aggregate.json" \
  --transitions "$AGGREGATE_ROOT/paired-transitions.jsonl" \
  --output "$AGGREGATE_ROOT/aggregate-verification.json" \
  >> "$AGGREGATE_ROOT/aggregate.log" 2>&1

printf 'SUCCESS\n' > "$AGGREGATE_ROOT/aggregate.status"
printf 'SUCCESS\n' > "$LOG_ROOT/stage2g.status"
current_run=''
trap - EXIT
