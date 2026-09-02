#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --job-name=cp2k-ramp
#SBATCH --constraint=ib
#SBATCH --output=ramp.%j.out
#SBATCH --account=your_slurm_account

# --- Workflow paths (see cluster.env.example) ---
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WORKFLOW_ROOT}/cp2k_env.sh"

# --- Load Environment ---
module purge
module load gcc/11.2.0
module load openmpi/4.1.1
module load cp2k/2023.1

export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1

# --- Configuration ---
START_GPA=1
END_GPA=40
SUBMIT_DIR="submitfiles"
BASE_DIR=$(pwd)

# --- Main Loop ---
for (( p=$START_GPA; p<=$END_GPA; p++ )); do
    FOLDER="${BASE_DIR}/${p}gpa"
    echo "Starting Pressure Step: $p GPa"
    
    mkdir -p "$FOLDER"
    cp "$SUBMIT_DIR/CellOpt.inp" "$FOLDER/"
    patch_cp2k_inp "$FOLDER/CellOpt.inp"

    # Setup inputs: First run uses templates, others use results from p-1
    if [ "$p" -eq "$START_GPA" ]; then
        cp "$SUBMIT_DIR/coordinates_init.xyz" "$FOLDER/coordinates_init.xyz"
        cp "$SUBMIT_DIR/init_cell.cell" "$FOLDER/init_cell.cell"
    else
        PREV_FOLDER="${BASE_DIR}/$((p-1))gpa"
        cp "$PREV_FOLDER/coordinates_final.xyz" "$FOLDER/coordinates_init.xyz"
        cp "$PREV_FOLDER/final_cell.cell" "$FOLDER/init_cell.cell"
    fi

    # Update pressure in .inp (1 GPa = 10000 bar)
    TARGET_BAR=$(( p * 10000 ))
    sed -i "s/EXTERNAL_PRESSURE.*/EXTERNAL_PRESSURE $TARGET_BAR/" "$FOLDER/CellOpt.inp"

    # Execute CP2K
    cd "$FOLDER" || exit
    mpirun -np $SLURM_NTASKS cp2k.psmp -o CellOpt.out -i CellOpt.inp

    # 1. Extract final coordinates (last 24 lines)
    if [ -f "QM_cellopt-POS-pos-1.xyz" ]; then
        tail -n 24 QM_cellopt-POS-pos-1.xyz > coordinates_final.xyz
    else
        echo "Error: XYZ trajectory not found at $p GPa"; exit 1
    fi

    # 2. Extract High-Precision Cell using bc
    LOG_FILE=$(ls QM_cellopt-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$LOG_FILE" ]; then
        # Parse magnitudes/angles
        A=$(grep "|a| =" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        B=$(grep "|b| =" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        C=$(grep "|c| =" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        ALPHA=$(grep "alpha \[degree\]:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        BETA=$(grep "beta  \[degree\]:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        GAMMA=$(grep "gamma \[degree\]:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')

        # Reconstruct vectors (A on X, B in XY)
        VECTORS=$(bc -l << EOF
scale=15
p=4*a(1)
ra=$ALPHA*p/180; rb=$BETA*p/180; rg=$GAMMA*p/180
ax=$A; ay=0; az=0
bx=$B*c(rg); by=$B*s(rg); bz=0
cx=$C*c(rb)
cy=$C*(c(ra)-(c(rb)*c(rg)))/s(rg)
cz=sqrt($C^2-cx^2-cy^2)
print "A ", ax, " ", ay, " ", az, "\n"
print "B ", bx, " ", by, " ", bz, "\n"
print "C ", cx, " ", cy, " ", cz, "\n"
EOF
)
        echo "$VECTORS" | awk '{printf "%-3s %15.10f %15.10f %15.10f\n", $1, $2, $3, $4}' > final_cell.cell
    else
        echo "Error: Log file not found at $p GPa"; exit 1
    fi

    echo "Completed $p GPa"
    cd "$BASE_DIR"
done
