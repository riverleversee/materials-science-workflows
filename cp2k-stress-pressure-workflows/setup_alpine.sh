#!/bin/bash
# One-time CURC Alpine setup after rsync. Creates cluster.env and checks inputs.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKFLOW_ROOT
cd "${WORKFLOW_ROOT}"

if [[ ! -f cluster.env ]]; then
  cp cluster.env.example cluster.env
  echo "Created cluster.env from cluster.env.example"
fi

source cp2k_env.sh

fail=0
check() {
  if eval "$2"; then
    echo "OK  $1"
  else
    echo "FAIL $1"
    fail=1
  fi
}

check "parameter trend" "test -f \"${CP2K_PARAM_TREND_PATH}\""
check "hydro 15 GPa coords" "test -f \"${HYDRO_DIR}/15gpa/coordinates_final.xyz\""
check "SKF directory" "test -d \"${CP2K_SKF_PATH}\" && compgen -G \"${CP2K_SKF_PATH}/*.skf\" > /dev/null"
check "CP2K SIF" "test -f \"${CP2K_SIF}\""

if [[ "${fail}" -ne 0 ]]; then
  echo "Fix cluster.env paths, then re-run: bash setup_alpine.sh" >&2
  exit 1
fi

echo
echo "Setup complete. Next commands:"
echo "  module load slurm/alpine"
echo "  sbatch scfhel/run_axis_id.sbatch"
echo "  # after axis-id job finishes:"
echo "  bash uniax/deltaPramp/run_prep_on_alpine.sh"
