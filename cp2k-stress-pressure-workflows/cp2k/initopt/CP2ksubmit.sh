#!/bin/bash
#SBATCH --time=1:10:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=cp2k-multiorient
#SBATCH --constraint=ib
#SBATCH --output=cp2k.%j.out
#SBATCH --account=your_slurm_account

# --- Workflow paths (see cluster.env.example) ---
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WORKFLOW_ROOT}/cp2k_env.sh"


module purge
module load gcc/11.2.0
module load openmpi/4.1.1
module load cp2k/2023.1

export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1

  # Run CP2K job
  cp optimized_cell.cell oldcell.cell
  mpirun -np $SLURM_NTASKS cp2k.psmp -o CellOpt.out -i CellOpt.inp
