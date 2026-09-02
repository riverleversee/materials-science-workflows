#!/bin/bash
# Find the most recent file starting with the given name

NAME=CellOpt
NAME2=CellOpt

source_file=classical_relaxation-pos-1.xyz

destination_file=coordinates.xyz

source_file2=QM_cellopt-POS-pos-1.xyz

destination_file2=coordinates.xyz

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




