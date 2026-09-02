#!/bin/bash
# Find the most recent file starting with the given name

NAME=CellOpt
NAME2=CellOpt

source_file=classical_relaxation-pos-1.xyz

destination_file=coordinates.xyz

source_file2=QM_cellopt-POS-pos-1.xyz

destination_file2=coordinatesout.xyz

# Check if the file exists




  tail -n 23 $source_file2 > $destination_file2

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
input_file="coordinatesout.xyz"
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



#   rm QM_cellopt-*
 #  mpirun -np $SLURM_NTASKS cp2k.psmp -o $NAME.out -i $NAME2.inp
# Find the most recent file starting with the given name

# Check if the file exists


# Create the output file and write the values in the desired format


