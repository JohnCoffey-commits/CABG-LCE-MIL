#!/usr/bin/env bash
set -euo pipefail

export CAUSAL_RUN_VERSION=v2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_causal_branch_v1_l4.sh" "$@"
