#!/bin/bash
# Steps 2–3 on CURC Alpine login node (after run_axis_id.sbatch completes).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORKFLOW_ROOT
cd "${WORKFLOW_ROOT}"

source cp2k_env.sh

module purge
module load slurm/alpine
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "/projects/${USER}/software/dftb_22"

test -f scfhel/minima_scale_1.0.json || {
  echo "Missing scfhel/minima_scale_1.0.json — run sbatch scfhel/run_axis_id.sbatch first" >&2
  exit 1
}

export MINIMA_JSON="${WORKFLOW_ROOT}/scfhel/minima_scale_1.0.json"
export HYDRO_COORDS="${HYDRO_DIR}/15gpa/coordinates_final.xyz"
export PRESSURE_GPA="${PRESSURE_GPA:-15}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-ucb-general}"

echo "=== make_axes_manifest.py ==="
python3 scfhel/make_axes_manifest.py | tee uniax/deltaPramp/make_manifest.log

echo "=== generate_ramp_jobs.py ==="
python3 uniax/deltaPramp/generate_ramp_jobs.py | tee uniax/deltaPramp/generate_jobs.log

P_ISO_BAR="$(python3 -c "print(int(round(float('${PRESSURE_GPA}') * 10000)))")"
OUT="uniax/deltaPramp/pIso_${P_ISO_BAR}bar"

echo
echo "Done. Test one axis:"
echo "  cd ${OUT} && sbatch axis_0/run_axis.sbatch"
echo "Then submit all:"
echo "  cd ${OUT} && bash copypastesubmit.txt"
