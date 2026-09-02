#!/bin/bash

# List of directories to process
directories=("2per" "1per" "1p5per" "p7per" "p5per" "p2per" "negp5per")

# Loop through each directory
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        cd "$dir" || exit
        sbatch CPmultiscale.sh
        echo "Submitted CPmultiscale.sh in $dir"
        cd ..
    else
        echo "Directory $dir does not exist, skipping..."
    fi
done
