#!/bin/bash
# Shared CP2K workflow environment. Source after setting WORKFLOW_ROOT.
#
# Example:
#   WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
#   source "${WORKFLOW_ROOT}/cp2k_env.sh"

if [ -z "${WORKFLOW_ROOT:-}" ]; then
  echo "cp2k_env.sh: set WORKFLOW_ROOT before sourcing" >&2
  return 1 2>/dev/null || exit 1
fi

if [ -f "${WORKFLOW_ROOT}/cluster.env" ]; then
  # shellcheck source=/dev/null
  source "${WORKFLOW_ROOT}/cluster.env"
fi

CP2K_ROOT="${CP2K_ROOT:-${WORKFLOW_ROOT}/cp2k}"
CP2K_SKF_PATH="${CP2K_SKF_PATH:-${CP2K_ROOT}/skfdatafiles}"
CP2K_SIF="${CP2K_SIF:-/path/to/cp2k.sif}"
CP2K_EXE="${CP2K_EXE:-/opt/cp2k/bin/cp2k.psmp}"
APPTAINER_CMD="${APPTAINER_CMD:-$(command -v apptainer 2>/dev/null || echo apptainer)}"
HYDRO_DIR="${HYDRO_DIR:-${CP2K_ROOT}/hydroopt}"

patch_cp2k_inp() {
  local f="$1"
  sed -i "s|__CP2K_SKF_PATH__|${CP2K_SKF_PATH}|g" "$f"
}

export CP2K_ROOT CP2K_SKF_PATH CP2K_SIF CP2K_EXE APPTAINER_CMD HYDRO_DIR WORKFLOW_ROOT
