#!/bin/bash

#SBATCH --time=23:20:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cp2k-test
#SBATCH --constraint=ib
#SBATCH --output=cp2k.%j.out
#SBATCH --account=ucb357_asc2


#Bash Volcompress_matchpressure 
#Retreive scale factor 
#Scale XY
#Start loop
#SCF calc
#Pull stress tensor values 
#Evaluate if step 1 
#If 1 assume slope
#If not step 1 gradient descent 
#Scale coordinates in Z 
#Copy coordinate file 
#Return to 1 


cd $SLURM_SUBMIT_DIR

module purge

module load gcc/11.2.0
module load openmpi/4.1.1
module load cp2k/2023.1

export SLURM_EXPORT_ENV=ALL


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1


new_scale=0.995
SCFNAME=CellOpt
echo "startscale"
echo "$new_scale"

sourcecell=../optimized_cellin.cell
outcell=optimized_cell.cell
sourcecoord=../coordinatesin.xyz
outcoord1=coordinates.xyz

new_scale_shell=$new_scale
new_scale_core=$new_scale
new_scalexy=$new_scale





function scale_z_coordinates {
    input_file=$1
    output_file=$8
    scale_1=$2
    scale_2=$3
    z_min=$4
    z_max=$5
    z_lower=$6
    z_upper=$7
    scale_3=$9
    while IFS= read -r line; do
        if [[ $line == *"SCALED"* || $line == *"ATOM"* ]]; then
            echo "$line" > "$output_file"
        else
            elements=($line)
            atom=${elements[0]}
            x=${elements[1]}
            y=${elements[2]}
            z=${elements[3]}

            if (( $(echo "$z > $z_min && $z < $z_lower" | bc -l) )); then
                new_z=$(echo "($z - $z_min) * $scale_1 + $z_min*$scale_3" | bc -l)
            elif (( $(echo "$z >= $z_lower && $z < $z_upper" | bc -l) )); then
                new_z=$(echo "($z - $z_lower) * $scale_2 +  ($z_lower - $z_min) * $scale_1 + $z_min*$scale_3" | bc -l)
            elif (( $(echo "$z >= $z_upper && $z < $z_max" | bc -l) )); then
                new_z=$(echo "($z - $z_upper) * $scale_1 +  ($z_upper - $z_lower) * $scale_2 +  ($z_lower - $z_min) * $scale_1 + $z_min*$scale_3 " | bc -l)
            elif (( $(echo "$z >= $z_max" | bc -l) )); then
                    new_z=$(echo "($z - $z_max)*$scale_3 + ($z_max - $z_upper) * $scale_1 +  ($z_upper - $z_lower) * $scale_2  + ($z_lower - $z_min) * $scale_1 + $z_min*$scale_3" | bc -l)
            else
                new_z=$(echo "$z * $scale_3 " | bc -l)
            fi

            echo "$atom $x $y $new_z" >> "$output_file"
        fi
    done < "$input_file"
}

# Usage example
# scale_z_coordinates "input_file.txt" 1.1 1.2 0.1 0.9 0.3 0.7
function find_and_calculate_z_values {
    input_file=$1

    # Initialize arrays for z values
    Zn_z=()
    S_z=()
    Cd_z=()

    # Read the input file and extract z-values
    while IFS= read -r line; do
        elements=($line)
        atom=${elements[0]}
        z=${elements[3]}

        case $atom in
            Zn) Zn_z+=($z) ;;
            S)  S_z+=($z) ;;
            Cd) Cd_z+=($z) ;;
        esac
    done < "$input_file"

    # Sort the arrays and extract required values
    IFS=$'\n' Zn_z_sorted=($(sort -g <<<"${Zn_z[*]}"))
    IFS=$'\n' S_z_sorted=($(sort -g <<<"${S_z[*]}"))
    IFS=$'\n' Cd_z_sorted=($(sort -g <<<"${Cd_z[*]}"))

    Zn_largest=${Zn_z_sorted[-1]}
    Zn_smallest=${Zn_z_sorted[0]}

    S_largest=${S_z_sorted[-1]}
    S_smallest=${S_z_sorted[0]}
    S4_largest=${S_z_sorted[-4]}
    S4_smallest=${S_z_sorted[3]}

    Cd_largest=${Cd_z_sorted[-1]}
    Cd_smallest=${Cd_z_sorted[0]}

    # Calculate and return the required values
    result_1=$(echo "($Cd_largest + $S4_largest) * 0.5" | bc -l)
    result_2=$(echo "($Cd_smallest + $S4_smallest) * 0.5" | bc -l)
    result_3=$(echo "$Zn_largest " | bc -l)
    result_4=$(echo "$Zn_smallest" | bc -l)
    echo "$result_1 $result_2 $result_3 $result_4"
}




scale_values() {
    local input_file="$1"
    local scaling_factor="$2"
    local output_file="$3"

    awk -v scale="$scaling_factor" '
    NR == 1 { $2 *= scale }
    NR == 2 { $3 *= scale }
    NR == 3 { $4 *= 1.0 }
    { print $0 }
    ' "$input_file" > "$output_file"
}




scale_values $sourcecell "$new_scalexy" $outcell

z_valuesnpl=$(find_and_calculate_z_values $sourcecoord)
read coretop corebottom shelltop shellbottom <<< "$z_valuesnpl"
echo "core top core bottom shell top shell bottom"
echo "$coretop $corebottom $shelltop $shellbottom"


  echo "new loop"
set_scale3=1.0
  scale_z_coordinates $sourcecoord $new_scale_shell $new_scale_core $shellbottom $shelltop $corebottom $coretop $outcoord1 $set_scale3






mpirun -np $SLURM_NTASKS cp2k.psmp -o $SCFNAME.out -i $SCFNAME.inp




NAME=CellOpt
NAME2=CellOpt

source_file=classical_relaxation-pos-1.xyz

destination_file=coordinates.xyz

source_file2=QM_cellopt-POS-pos-1.xyz

destination_file2=coordinates.xyz

# Check if the file exists




  tail -n 39 $source_file2 > $destination_file2

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



