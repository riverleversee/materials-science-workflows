#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --job-name=perov-npt
#SBATCH --constraint=ib
#SBATCH --output=lammps.%j.out
#SBATCH --account=YOUR_SLURM_ACCOUNT

set -euo pipefail

# Path to LAMMPS binary on your cluster
LAMMPS_BIN="${LAMMPS_BIN:-lmp}"

# Expect perovin.in and lammps_data_full_supercell.data in the submit directory
infile=perovin.in

module purge
module load gcc/14.2.0 openmpi/5.0.6 fftw/3.3.10 2>/dev/null || true

mpirun -np "${SLURM_NTASKS}" "${LAMMPS_BIN}" -in "${infile}"
