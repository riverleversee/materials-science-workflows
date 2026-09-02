#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cp2k-test
#SBATCH --constraint=ib
#SBATCH --output=cp2k.%j.out
#SBATCH --account=ucb357_asc2


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
  




  mpirun -np $SLURM_NTASKS cp2k.psmp -o $NAME.out -i $NAME.inp
# Find the most recent file starting with the given name

# Check if the file exists
latest_file=$(ls -t QM_cellopt-optimized_cell.dat-1_* | head -n 1)
# Extract the values for vectors A, B, and C
a_values=$(grep "Vector a" "$latest_file" | awk '{print $5, $6, $7}' | tail -n 1)
b_values=$(grep "Vector b" "$latest_file" | awk '{print $5, $6, $7}' | tail -n 1)
c_values=$(grep "Vector c" "$latest_file" | awk '{print $5, $6, $7}' | tail -n 1)

# Create the output file and write the values in the desired format


output_file="optimized_cell.cell"
{
    echo "A $a_values"
    echo "B $b_values"
    echo "C $c_values"
} > "$output_file"

echo "Values for vectors A, B, and C have been written to $output_file"


  tail -n 8 $source_file2 > $destination_file2

filename="optimized_cell.cell"

# Read the values from the file
while read -r line; do
    case "$line" in
        A*) a=$(echo $line | awk '{print $2}') ;;
        B*) b=$(echo $line | awk '{print $3}') ;;
        C*) c=$(echo $line | awk '{print $4}') ;;
    esac
done < "$filename"

# Print the values to verify
echo "a=$a"
echo "b=$b"
echo "c=$c"

# Input and output files
input_file="coordinates.xyz"
output_file="scaled_coordinates.xyz"

# Function to convert Cartesian to scaled coordinates
convert_to_scaled() {
    x=$1
    y=$2
    z=$3

    sx=$(echo "scale=6; $x / $a" | bc -l)
    sy=$(echo "scale=6; $y / $b" | bc -l)
    sz=$(echo "scale=6; $z / $c" | bc -l)

    echo "$sx $sy $sz"
}

# Find the most negative z-coordinate
min_z=$(awk '{if ($4 < min || NR == 1) min = $4} END {print min}' "$input_file")

# Read input file and write to output file
{
    echo "SCALED T"
    while read -r element x y z; do
        # Adjust z-coordinate to be positive
        adjusted_z=$(echo "$z - $min_z + 15.7962" | bc -l)
        scaled_coords=$(convert_to_scaled $x $y $z)
        echo "$element $scaled_coords"
    done < "$input_file"
} > "$output_file"

echo "Conversion complete. Scaled coordinates written to $output_file"
