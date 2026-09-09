#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=/home/workspace/Medic-AD
readonly ADAPTER_SOURCE=/home/workspace/Medic-AD-stage2g-adapter
readonly R0_SOURCE=/home/workspace/Medic-AD-stage2g-r0
readonly LOG_ROOT=/home/logs/medic-ad/stage2g-fb-maq
readonly DATA_ROOT=/home/data/medic-ad/official/med_anomaly
readonly DATA_DIR="$LOG_ROOT/data"
readonly PLAN_DIR="$LOG_ROOT/plan"
readonly PREFLIGHT_DIR="$LOG_ROOT/preflight"
readonly REGISTRY="$PLAN_DIR/run-registry.json"
readonly PYTHON=/home/envs/medic-ad-train/bin/python

mkdir -p "$PREFLIGHT_DIR" "$PLAN_DIR"
if [[ -e "$PREFLIGHT_DIR/preflight.status" ]]; then
  printf 'Refusing to overwrite Stage 2G preflight evidence.\n' >&2
  exit 1
fi
exec > >(tee -a "$PREFLIGHT_DIR/preflight.log") 2>&1

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'FAILED:%s\n' "$rc" > "$PREFLIGHT_DIR/preflight.status"
  fi
}
trap on_exit EXIT
printf 'RUNNING\n' > "$PREFLIGHT_DIR/preflight.status"

cd "$REPO"
if [[ ! -e "$REGISTRY" ]]; then
  cp reproduction/stage2g/plan/run-registry.json "$REGISTRY"
elif ! cmp -s reproduction/stage2g/plan/run-registry.json "$REGISTRY"; then
  printf 'Persistent run registry differs from the locked repository registry.\n' >&2
  exit 1
fi
if [[ ! -d "$ADAPTER_SOURCE/.git" && ! -f "$ADAPTER_SOURCE/.git" ]]; then
  git worktree add --detach "$ADAPTER_SOURCE" 28e72711c38e9c47865657a009a2867f7205c330
fi
if [[ ! -d "$R0_SOURCE/.git" && ! -f "$R0_SOURCE/.git" ]]; then
  git worktree add --detach "$R0_SOURCE" ad62e7c910f4febad7b07030bd1c11796ae064e7
fi

export PYTHONPATH="$REPO:$REPO/qwen-vl-finetune"
"$PYTHON" reproduction/stage2g/preflight.py \
  --repo-root "$REPO" \
  --adapter-source "$ADAPTER_SOURCE" \
  --r0-source "$R0_SOURCE" \
  --registry "$REGISTRY" \
  --dataset-root "$DATA_ROOT" \
  --dataset-audit "$DATA_DIR/dataset-audit.json" \
  --evaluation-manifest "$DATA_DIR/evaluation-manifest.jsonl" \
  --r0-model /home/checkpoints/MEDIC-AD \
  --r0-revision-file /home/logs/medic-ad/stage1/checkpoint-revision.txt \
  --base-model /home/checkpoints/Lingshu-7B \
  --r0-checkpoint-audit-output "$PREFLIGHT_DIR/r0-checkpoint-audit.json" \
  --output "$PREFLIGHT_DIR/preexperiment-ready.json"

printf 'SUCCESS\n' > "$PREFLIGHT_DIR/preflight.status"
trap - EXIT
