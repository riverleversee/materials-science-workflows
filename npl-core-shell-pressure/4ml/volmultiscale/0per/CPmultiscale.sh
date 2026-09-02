#!/bin/bash
# materials-science-workflows note:
# Slurm driver for core/shell multiscale compression (volmultiscale trees).
# Scales shell vs core independently (new_scale_shell / new_scale_core), runs
# JustEng1/2/3 single-points, matches stress. Paper EOS path for NPL platelets.
# (Git copy only; cluster source tree was not modified.)

#SBATCH --time=22:1:30
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cp2k-test
#SBATCH --constraint=ib
#SBATCH --output=cp2k.%j.out
#SBATCH --account=ucb357_asc2

new_scale=1.0


cd $SLURM_SUBMIT_DIR

module purge

module load gcc/11.2.0
module load openmpi/4.1.1
module load cp2k/2023.1

export SLURM_EXPORT_ENV=ALL


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1


sourcecell=optimized_cellin.cell
outcell=optimized_cell.cell
sourcecoord=coordinatesin.xyz
outcoord1=coordinates1.xyz
outcoord2=coordinates2.xyz
outcoord3=coordinates3.xyz


new_scale_shell=$(echo "$new_scale " | bc)
new_scale_core=$(echo "$new_scale " | bc)
firststep=0.001
new_scalexy=1.0

pressevalfrac=0.9999
stressname=QM_cellopt-stressfile-1_0.stress_tensor
SCFNAME1=JustEng1
SCFNAME2=JustEng2
SCFNAME3=JustEng3
step_number=1
echo "startscale"
echo "$new_scale"


start_step_size=0.01
verysmallstep=0.0003
smallstep=0.001
largestep=0.01

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
    result_3=$(echo "$Zn_largest + ($Zn_largest - $S_largest) * 0.5" | bc -l)
    result_4=$(echo "$Zn_smallest + ($Zn_smallest - $S_smallest) * 0.5" | bc -l)
    echo "$result_1 $result_2 $result_3 $result_4 $Zn_largest $Zn_smallest"
}



function load_stress_tensor {
    local file="$1"
    xx=$(grep -E "STRESS\| +x" "$file" | tail -n 2 | head -n 1 | awk '{print $3}')
    yy=$(grep -E "STRESS\| +y" "$file" | tail -n 2 | head -n 1 | awk '{print $4}')
    zz=$(grep -E "STRESS\| +z" "$file" | tail -n 2 | head -n 1 | awk '{print $5}')
}

function process_files {
    local scale_xy=$1
    local scale_core=$2
    local scale_shell=$3
    local scale_presseval=$4
    local vol0core=$5
    local vol0shell=$6
    local cellvol0=$7
    unit_scale_stress=1.0
    unit_scale_hartpera3togpa=4359.75
    local energy_file_1="JustEng1.out"
    local energy_file_2="JustEng2.out"
    local energy_file_3="JustEng3.out"

    # Extract the resulting energy from each file
    local energyvalue1=$(strings "$energy_file_1" | grep "ENERGY|" | tail -n 1 | awk '{print $9}')
    local energyvalue2=$(strings "$energy_file_2" | grep "ENERGY|" | tail -n 1 | awk '{print $9}')
    local energyvalue3=$(strings "$energy_file_3" | grep "ENERGY|" | tail -n 1 | awk '{print $9}')

    # Compute the stress in the core
    local dvcore=$(echo "$vol0core * $scale_xy * $scale_xy * $scale_core * (1 - $scale_presseval)" | bc -l)
    local dE_core=$(echo "($energyvalue2 - $energyvalue1)*$unit_scale_hartpera3togpa" | bc -l)
    local stress_core=$(echo "$dE_core / $dvcore" | bc -l)

    # Compute the stress in the shell
    local dvshell=$(echo "$vol0shell * $scale_xy * $scale_xy * $scale_shell * (1 - $scale_presseval)" | bc -l)
    local dE_shell=$(echo "($energyvalue3 - $energyvalue1)* $unit_scale_hartpera3togpa" | bc -l)
    local stress_shell=$(echo "$dE_shell / $dvshell" | bc -l)

    # Load the stress tensor from the first file and compute average x-y stress
    load_stress_tensor "$stressname"
    local xx_decimal=$(printf "%.15f" $xx)
    local yy_decimal=$(printf "%.15f" $yy)
    local zz_decimal=$(printf "%.15f" $zz)
    local avg_xyvalue=$(echo "0.5 * ($xx_decimal + $yy_decimal)" | bc -l)
    # Scale the average x-y stress to the correct units
    local newvol=$(echo "$vol0core * $scale_core + $vol0shell * $scale_shell" | bc -l)
    local scaledxy_pre_div=$(echo "$avg_xyvalue * $unit_scale_stress * $cellvol0" | bc -l)
    local scaledxyval=$(echo "$scaledxy_pre_div / $newvol" | bc -l)


    # Compute and return the final values
    local result_shell=$(echo "$scaledxyval - $stress_shell" | bc -l)
    local result_core=$(echo "$scaledxyval - $stress_core" | bc -l)
    echo "Debug: result_1 = $dE_shell , result_2 = $dE_core , result_3 = $dvshell, result_4 = $dvcore" >&2
    echo "Debug: result_1 = $energyvalue1 , result_2 = $energyvalue2 , result_3 = $energyvalue3" >&2
    echo "$result_shell $result_core $stress_shell $stress_core $scaledxyval"
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


function read_diagonal_elements {
    input_file=$1

    # Counters to identify rows
    row_counter=1

    while IFS= read -r line; do
        elements=($line)
        case $row_counter in
            1) cellA=${elements[1]} ;;  # Second entry of row 1
            2) cellB=${elements[2]} ;;  # Third entry of row 2
            3) cellC=${elements[3]} ;;  # Fourth entry of row 3
        esac
        ((row_counter++))
    done < "$input_file"
}


determine_scale_factor() {
    local step_number="$1"
    local prev_scale1="$2"
    local prev_scale2="$3"
    local prev_dep1="$4"
    local prev_dep2="$5"
    local calctype="$6"
# Calculate max step size based on step number




  # Determine the max step size
  if [ "$step_number" -le 3 ]; then
    max_step_size=$start_step_size
  else
    # Calculate the cycle step and cycle number
    local cycle_step=$(( (step_number - 4) % 6 + 1 ))
    local cycle_number=$(( (step_number - 4) / 6 ))
    if [ $calctype -eq 0 ]; then
      if [ "$cycle_step" -eq 1 ]; then
        max_step_size=$verysmallstep
      elif [ "$cycle_step" -le 3 ]; then
        max_step_size=$verysmallstep
      elif [ "$cycle_step" -eq 4 ]; then
         max_step_size=$smallstep
      else
        max_step_size=$largestep
      fi
    elif [ $calctype -eq 1 ]; then
      if [ "$cycle_step" -eq 1 ]; then
        max_step_size=$smallstep
      elif [ "$cycle_step" -le 3 ]; then
        max_step_size=$largestep
      elif [ "$cycle_step" -eq 4 ]; then
        max_step_size=$verysmallstep
     else
        max_step_size=$verysmallstep
      fi
    fi
  fi





    local new_scale

    if [ "$step_number" -eq 1 ]; then
        # Use the assumed slope to compute a new scale factor
        new_scale=$(echo "$prev_scale1 - $firststep" | bc -l)
    else
        # Compute the gradient descent to get a new scale factor
        local gradient=$(echo "($prev_dep2 - $prev_dep1) / ($prev_scale2 - $prev_scale1)" | bc -l)
        local proposed_new_scale=$(echo "$prev_scale2 - ($prev_dep2 / $gradient)" | bc -l)
        local step_size=$(echo "$proposed_new_scale - $prev_scale1" | bc -l)
        
        # Check if the step size exceeds the maximum allowed step size
        if (( $(echo "($step_size < 0) && ($step_size < -$max_step_size)" | bc -l) )); then
            new_scale=$(echo "$prev_scale1 - $max_step_size" | bc -l)
        elif (( $(echo "($step_size > 0) && ($step_size > $max_step_size)" | bc -l) )); then
            new_scale=$(echo "$prev_scale1 + $max_step_size" | bc -l)
        else
            new_scale=$proposed_new_scale
        fi
    fi

    echo "$new_scale"
}

function compute_initial_variables {
    matrix_file=$1
    z_values_file=$2

    # Read diagonal elements
    read_diagonal_elements "$matrix_file"
    
    # Find and calculate z-values
    z_values=$(find_and_calculate_z_values "$z_values_file")
    read result_1 result_2 result_3 result_4 ZnL ZnS<<< "$z_values"
    
    # Compute cell volume
    cellvol0=$(echo "$cellA * $cellB * $cellC" | bc -l)
    
    # Compute core and shell volumes
    vol0core=$(echo "($result_1 - $result_2) * $cellA * $cellB * $cellC" | bc -l)
    vol0shell=$(echo "($result_3 - $result_4) * $cellA * $cellB *$cellC - $vol0core" | bc -l)
    
    echo "cellvol0=$cellvol0"
    echo "vol0core=$vol0core"
    echo "vol0shell=$vol0shell"
}


output=$(compute_initial_variables $sourcecell $sourcecoord)
eval "$output"


scale_values $sourcecell "$new_scalexy" $outcell
prev_scale2_core=1.0
prev_scale1_core="$new_scale_core"
prev_dep2_core=0.0


prev_scale2_shell=1.0
prev_scale1_shell="$new_scale_shell"
prev_dep2_shell=0.0
z_valuesnpl=$(find_and_calculate_z_values $sourcecoord)
read coretop corebottom shelltop shellbottom ZnLarge ZnSmall<<< "$z_valuesnpl"
echo "core top core bottom shell top shell bottom"
echo "$coretop $corebottom $shelltop $shellbottom" 
echo "$ZnLarge $ZnSmall"
while true; do

  echo "new loop"
newscalecore1=$(echo "$new_scale_core * $pressevalfrac" | bc -l)
newscaleshell1=$(echo "$new_scale_shell * $pressevalfrac" | bc -l)
set_scale3=1.0  
  scale_z_coordinates $sourcecoord $new_scale_shell $new_scale_core $ZnSmall $ZnLarge $corebottom $coretop $outcoord1 $set_scale3
  
  scale_z_coordinates $sourcecoord $new_scale_shell $newscalecore1 $ZnSmall $ZnLarge $corebottom $coretop $outcoord2 $set_scale3

  scale_z_coordinates $sourcecoord $newscaleshell1 $new_scale_core $ZnSmall $ZnLarge $corebottom $coretop $outcoord3 $set_scale3
    mpirun -np $SLURM_NTASKS cp2k.psmp -o $SCFNAME1.out -i $SCFNAME1.inp
   mpirun -np $SLURM_NTASKS cp2k.psmp -o $SCFNAME2.out -i $SCFNAME2.inp
 mpirun -np $SLURM_NTASKS cp2k.psmp -o $SCFNAME3.out -i $SCFNAME3.inp
    read result_shell result_core stress_shell stress_core scaledxyval <<< $(process_files $new_scalexy $new_scale_core $new_scale_shell $pressevalfrac $vol0core $vol0shell $cellvol0)
 
    dep_value_core=$result_core
  dep_value_shell=$result_shell
  prev_dep1_core=$result_core
  prev_dep1_shell=$result_shell
  echo "xy scaled stress"
  echo "$scaledxyval"
  echo "core stress"
  echo "$stress_core" 
  echo "shell stress"
  echo "$stress_shell"
  echo "core and shell deviation"
  echo "$dep_value_core"
  echo "$dep_value_shell"
  echo "scale shell"
  echo  "$new_scale_shell"
  echo "scale core"
  echo  "$new_scale_core"
  energyoutfilenow="JustEng1.out"

  last_occurrence=$(strings "$energyoutfilenow" | grep "ENERGY|" | tail -n 1)
  new_scale_core=$(determine_scale_factor "$step_number" "$prev_scale1_core" "$prev_scale2_core" "$prev_dep1_core" "$prev_dep2_core" 0)
    new_scale_shell=$(determine_scale_factor "$step_number" "$prev_scale1_shell" "$prev_scale2_shell" "$prev_dep1_shell" "$prev_dep2_shell" 1)
   

    prev_scale2_core="$prev_scale1_core"

    prev_scale1_core="$new_scale_core"

    # Update variables for the next iteration
    prev_dep2_core="$prev_dep1_core"

    prev_scale2_shell="$prev_scale1_shell"

    prev_scale1_shell="$new_scale_shell"

    # Update variables for the next iteration
    prev_dep2_shell="$prev_dep1_shell"

    step_number=$((step_number + 1))

  
    if (( $(echo "$dep_value_core < 0.0005 && $dep_value_core > -0.0005 && $dep_value_shell < 0.0005 &&  $dep_value_shell > -0.0005" | bc -l) )); then
        break
    fi
done
