#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/clean/Medic-AD" >&2
  exit 2
fi

target=$1
expected_base=ad62e7c910f4febad7b07030bd1c11796ae064e7
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
overlay_dir=$(cd "$script_dir/../overlay" && pwd)

if [[ ! -d "$target/.git" ]]; then
  echo "target is not a Git checkout: $target" >&2
  exit 2
fi

actual_head=$(git -C "$target" rev-parse HEAD)
if [[ "$actual_head" != "$expected_base" ]]; then
  echo "expected MEDIC-AD base $expected_base, found $actual_head" >&2
  exit 1
fi

if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
  echo "target checkout must be clean before applying the overlay" >&2
  exit 1
fi

cp -R "$overlay_dir/." "$target/"
echo "CABG-LCE-MIL source overlay applied to $target"
