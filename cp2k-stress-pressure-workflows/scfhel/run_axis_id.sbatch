#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=axis-id
#SBATCH --output=axis_id.%j.out
#SBATCH --account=ucb-general

set -euo pipefail

module purge
module load slurm/alpine
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "/projects/${USER}/software/dftb_22"

WORKFLOW_ROOT="${WORKFLOW_ROOT:-/scratch/alpine/${USER}/cp2k-stress-pressure-workflows}"
cd "${WORKFLOW_ROOT}"
source cp2k_env.sh

python3 scfhel/uniax_surrogate_optimizer_test.py
