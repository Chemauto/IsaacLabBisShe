#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/source/MyProject:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/generate_deploy_yaml.py" "$@"
