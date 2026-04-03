#!/usr/bin/env bash
# Shared helper: runs helm-run with custom run_specs registered.
# Source project root onto PYTHONPATH, then invoke HELM via Python so that
# our @run_spec_function decorators fire in the same process.
#
# Usage (from other eval scripts):
#   source "$(dirname "$0")/_helm_run.sh"
#   helm_run "${HELM_ARGS[@]}"

_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

helm_run() {
    python -c "
import sys, run_specs
from helm.benchmark.run import main
sys.exit(main())
" "$@"
}
