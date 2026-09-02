#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cp2k-test
#SBATCH --constraint=ib
#SBATCH --output=cp2k.%j.out
#SBATCH --account=ucb357_asc3


cd $SLURM_SUBMIT_DIR

module purge

module load gcc/11.2.0
module load openmpi/4.1.1
module load cp2k/2023.1

export SLURM_EXPORT_ENV=ALL

NAME=CellOpt
NAME2=CellOpt

source_file=classical_relaxation-pos-1.xyz 

destination_file=coordinates.xyz

source_file2=QM_cellopt-POS-pos-1.xyz

destination_file2=coordinates.xyz


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1




#while true
#do
  




   cp optimized_cell.cell oldcell.cell  
  mpirun -np $SLURM_NTASKS cp2k.psmp -o $NAME.out -i $NAME.inp
# Find the most recent file starting with the given name

