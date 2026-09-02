#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --job-name=uniax-aaxis
#SBATCH --constraint=ib
#SBATCH --output=uniax.%j.out
#SBATCH --account=your_slurm_account

# --- Workflow paths (see cluster.env.example) ---
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WORKFLOW_ROOT}/cp2k_env.sh"

# --- Load Environment (OpenMPI for mpirun; CP2K runs via Apptainer) ---
module purge
module load gcc/11.2.0 openmpi/4.1.1

# CP2K via Apptainer (no module load cp2k)
CP2K_SIF="${CP2K_SIF:-/path/to/cp2k.sif}"
# Executable inside container (full path; directory may not be in PATH under mpirun)
CP2K_EXE="${CP2K_EXE:-/opt/cp2k/bin/cp2k.psmp}"

# NOW set the fixes so they stick:
export UCX_TLS=self,cma,posix,sysv,rc,ud,tcp
export UCX_MEMTYPE_CACHE=n

# Force OpenMPI to use UCX and ignore the broken legacy OpenIB
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib

export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run CP2K inside Apptainer (bind parent dir so container sees job + hydroopt)
run_cp2k() {
    local inp="$1" out="$2"
    patch_cp2k_inp "$inp"
    local bind_base
    bind_base="$(cd "$BASE_DIR/.." 2>/dev/null && pwd)" || bind_base="$BASE_DIR/.."
    srun -n "${SLURM_NTASKS}" apptainer exec \
        --sharens \
        --pwd "$(pwd)" \
        --bind "${bind_base}:${bind_base}" \
        --bind "${CP2K_SKF_PATH%/*}:${CP2K_SKF_PATH%/*}" \
        "$CP2K_SIF" "$CP2K_EXE" -o "$out" -i "$inp"
}

# --- Configuration ---
START_GPA=15
END_GPA=40
MAX_STAGE12_RETRIES=10
BASE_DIR=$(pwd)
SUBMIT_DIR="$(dirname "$BASE_DIR")/submitfiles"
HYDRO_DIR="${HYDRO_DIR:-${WORKFLOW_ROOT}/cp2k/hydroopt}"

# Array of deltas to run
DELTAS=(1 3 5 8 10)

# --- Shared extraction: parse last frame from XYZ, convert Cartesian to fractional ---
extract_coords_to_scaled() {
    local xyz_path="$1"
    local cell_path="$2"
    local out_path="$3"
    module purge
    module load python
    module load anaconda
    python3 - <<PY
import numpy as np
from pathlib import Path

xyz = Path(r"$xyz_path")
cell_file = Path(r"$cell_path")
out = Path(r"$out_path")

def read_last_frame_xyz(path):
    lines = path.read_text().splitlines()
    if not lines:
        return []
    i = len(lines) - 1
    n = 0
    while i >= 0:
        try:
            n = int(lines[i].split()[0])
            break
        except (ValueError, IndexError):
            i -= 1
    if n <= 0:
        return []
    start = i
    atoms = []
    for ln in lines[start + 2 : start + 2 + n]:
        parts = ln.split()
        if len(parts) >= 4:
            atoms.append((parts[0], np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
    return atoms

def read_cell_simple(path):
    rows = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s[0] in ("#", "!"):
            continue
        parts = s.split()
        nums = []
        for t in parts:
            try:
                nums.append(float(t))
            except ValueError:
                pass
        if len(nums) >= 3:
            rows.append(nums[-3:])
        if len(rows) == 3:
            break
    return np.array(rows, dtype=float)

cell = read_cell_simple(cell_file)
inv_cell = np.linalg.inv(cell.T)
atoms = read_last_frame_xyz(xyz)
if not atoms:
    raise SystemExit(f"Failed to read atoms from {xyz}")
out_lines = ["SCALED T"]
for el, r in atoms:
    f = inv_cell @ r
    out_lines.append(f"{el:2s} {f[0]: .12f} {f[1]: .12f} {f[2]: .12f}")
out.write_text("\n".join(out_lines) + "\n")
PY
    module purge
    module load gcc/11.2.0 openmpi/4.1.1
}

# --- Extract cell from CP2K Log (or .dat fallback) ---
extract_cell_from_log() {
    local log_file="$1"
    local out_cell="$2"
    module purge
    module load python
    module load anaconda
    python3 - "$log_file" "$out_cell" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

def extract_vec_from_line(line):
    """Extract 3 floats (vector) from a line. Format: 'Vector a [angstrom]: x y z |a|=...' uses first 3; '(x y z)' uses last 3."""
    nums = re.findall(r'-?\d+\.?\d*(?:[Ee][-+]?\d+)?', line)
    if len(nums) < 3:
        return None
    if '[angstrom]' in line or 'Vector' in line:
        return [float(x) for x in nums[:3]]
    return [float(x) for x in nums[-3:]]

def try_log(log_p):
    vectors = {}
    text = log_p.read_text(errors='replace')
    for label, key in [('a', 'a'), ('b', 'b'), ('c', 'c')]:
        for pat in [
            rf'CELL\s*[\|\s]+\s*Vector\s+{key}\s',
            rf'Vector\s+{key}\s+\[',
            rf'Vector\s+{key}\s',
            rf'Lattice\s+vector\s+{key}\s',
            rf'\b{key}\s+vector\s',
        ]:
            last_v = None
            for m in re.finditer(pat, text, re.I):
                start = text.rfind('\n', 0, m.start()) + 1
                end = text.find('\n', m.end())
                line = text[start:end] if end >= 0 else text[start:m.end()+120]
                v = extract_vec_from_line(line)
                if v:
                    last_v = v
            if last_v:
                vectors[label] = last_v
                break
    if len(vectors) == 3:
        return [vectors['a'], vectors['b'], vectors['c']]
    return None

def try_dat(log_p):
    """Fallback: parse .dat file (same base name, no .Log)."""
    base = str(log_p).rstrip('.Log').rstrip('.log')
    dat_p = Path(base)
    if not dat_p.exists():
        return None
    rows = []
    for line in dat_p.read_text(errors='replace').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) >= 4 and parts[0].upper() in ('A', 'B', 'C'):
            try:
                nums = [float(x) for x in parts[1:4]]
                rows.append((parts[0].upper(), nums))
            except ValueError:
                pass
        elif len(parts) >= 3:
            try:
                nums = [float(x) for x in parts[-3:]]
                if len(nums) == 3:
                    rows.append((None, nums))
            except ValueError:
                pass
    if len(rows) >= 3:
        last_a = last_b = last_c = None
        for lab, n in rows:
            if lab == 'A': last_a = n
            elif lab == 'B': last_b = n
            elif lab == 'C': last_c = n
        if last_a and last_b and last_c:
            return [last_a, last_b, last_c]
        generic = [r[1] for r in rows if r[0] is None]
        if len(generic) >= 3:
            return generic[-3:]
    return None

cell = try_log(log_path)
if not cell:
    cell = try_dat(log_path)
if not cell:
    print(f"Error: could not parse cell from {log_path}", file=sys.stderr)
    lines = log_path.read_text(errors='replace').splitlines()
    relevant = [(i+1, ln) for i, ln in enumerate(lines) if 'vector' in ln.lower() or 'cell' in ln.lower()]
    print("Sample lines with 'vector' or 'cell' (for format debug):", file=sys.stderr)
    for num, ln in relevant[-15:]:
        print(f"  {num}: {ln[:140]}", file=sys.stderr)
    sys.exit(1)

with open(out_path, 'w') as f:
    for name, vec in [('A', cell[0]), ('B', cell[1]), ('C', cell[2])]:
        f.write(f"{name:<3} {vec[0]:18.15f} {vec[1]:18.15f} {vec[2]:18.15f}\n")
PY
    module purge
    module load gcc/11.2.0 openmpi/4.1.1
}

# --- Check if stage 12 (full cell opt) converged by parsing CellOpt.out ---
# Supports both old CP2K format ("Convergence in step size" etc.) and new format ("Maximum step size is converged" etc.)
check_stage12_converged() {
    local out_file="${1:-CellOpt.out}"
    [ ! -f "$out_file" ] && return 1
    grep -q "GEOMETRY OPTIMIZATION COMPLETED" "$out_file" || return 1
    local before
    before=$(sed '/GEOMETRY OPTIMIZATION COMPLETED/,$d' "$out_file")
    # Each criterion: try old-format string first, then new-format string
    local crits=(
        "Convergence in step size|Maximum step size is converged"
        "Convergence in RMS step|RMS step size is converged"
        "Conv. in gradients|Maximum gradient is converged"
        "Conv. in RMS gradients|RMS gradient is converged"
        "Conv. for  PRESSURE|Pressure is converged"
    )
    for crit in "${crits[@]}"; do
        local line
        line=$(echo "$before" | grep -E "$crit" | tail -1)
        [[ -z "$line" || "$line" != *"YES"* ]] && return 1
    done
    return 0
}

# --- Check stall and apply isotropic scaling if fixed-atom opt stalled (stages 7+) ---
check_stall_and_scale() {
    local run_out="$1" cell_file="$2" target_p_bar="$3" p_gpa="$4" stage_num="$5"
    [ ! -f "$run_out" ] || [ ! -f "$cell_file" ] && return 0
    module purge
    module load python
    module load anaconda
    if python3 "$(dirname "$BASE_DIR")/scripts/check_stall_and_scale.py" "$run_out" "$cell_file" "$target_p_bar" "$p_gpa" "$HYDRO_DIR" "$stage_num"; then
        :
    else
        cp "$cell_file" init_cell.cell
    fi
    module purge
    module load gcc/11.2.0 openmpi/4.1.1
}

# --- Remove all but last restart file and all but last optimized_cell file for given project ---
cleanup_stage_outputs() {
    local proj="$1"
    local files i num
    files=($(ls ${proj}*[Rr]estart* 2>/dev/null | grep -v '[Bb]ak' | sort -V))
    num=${#files[@]}
    if [ "$num" -gt 1 ]; then
        for ((i=0; i<num-1; i++)); do rm -f "${files[$i]}"; done
    fi
    files=($(ls ${proj}-optimized_cell.dat* 2>/dev/null | sort -V))
    num=${#files[@]}
    if [ "$num" -gt 1 ]; then
        for ((i=0; i<num-1; i++)); do rm -f "${files[$i]}"; done
    fi
}

# --- Extract cell from last restart file (high precision, stage 8 only) ---
extract_cell_from_restart() {
    local proj="$1"
    local out_cell="$2"
    local rst
    rst=$(ls ${proj}*[Rr]estart* 2>/dev/null | grep -v '[Bb]ak' | sort -V | tail -n 1)
    [ -z "$rst" ] && return 1
    module purge
    module load python
    module load anaconda
    python3 - "$rst" "$out_cell" <<'PY'
import re
import sys
from pathlib import Path

rst_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

text = rst_path.read_text(errors='replace')
blocks = []
rows = []
in_cell = False
for line in text.splitlines():
    s = line.strip()
    if '&CELL' in s and '&END' not in s:
        if rows:
            blocks.append(rows)
        rows = []
        in_cell = True
        continue
    if in_cell and '&END' in s:
        if rows:
            blocks.append(rows)
        rows = []
        in_cell = False
        continue
    if in_cell and len(s) >= 3:
        m = re.match(r'^([ABC])\s+(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s+(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s+(-?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s*', s, re.I)
        if m:
            rows.append((m.group(1).upper(), [float(m.group(2)), float(m.group(3)), float(m.group(4))]))
if rows:
    blocks.append(rows)
rows = blocks[-1] if blocks else []
if len(rows) < 3:
    sys.exit(1)
vec = {r[0]: r[1] for r in rows}
with open(out_path, 'w') as f:
    for name in ('A', 'B', 'C'):
        v = vec.get(name)
        if v:
            f.write(f"{name:<3} {v[0]:18.15f} {v[1]:18.15f} {v[2]:18.15f}\n")
PY
    module purge
    module load gcc/11.2.0 openmpi/4.1.1
}

# --- Main Loop: iterate P first, then D (delta) for easier restart ---
for (( p=$START_GPA; p<=$END_GPA; p++ )); do
    PREV_D=""
    for D in "${DELTAS[@]}"; do
        DELTA_DIR="${BASE_DIR}/${D}gpadelta"
        mkdir -p "$DELTA_DIR"
        FOLDER="${DELTA_DIR}/${p}gpa"
        echo "Processing Delta: $D GPa | Target P: $p GPa"

        mkdir -p "$FOLDER"
        if [ -z "$PREV_D" ]; then
            SRC="${HYDRO_DIR}/${p}gpa"
        else
            SRC="${BASE_DIR}/${PREV_D}gpadelta/${p}gpa"
        fi
        if [ ! -f "$SRC/final_cell.cell" ]; then
            echo "Error: Source $SRC/final_cell.cell not found!"; exit 1
        fi

        # --- Initial setup: rotate cell so A along X (first delta) or copy from previous delta ---
        if [ -z "$PREV_D" ]; then
            module purge
            module load python
            module load anaconda
            python3 "$(dirname "$BASE_DIR")/scripts/rotate_cell_and_convert.py" aaxis \
                "$SRC/final_cell.cell" "$SRC/coordinates_final.xyz" \
                "$FOLDER/init_cell.cell" "$FOLDER/coordinates_init.xyz"
        else
            cp "$SRC/final_cell.cell" "$FOLDER/init_cell.cell"
            extract_coords_to_scaled "$SRC/coordinates_final.xyz" "$SRC/final_cell.cell" "$FOLDER/coordinates_init.xyz"
        fi
        module purge
        module load gcc/11.2.0 openmpi/4.1.1

        # --- Stress tensor (a-axis: high stress along X) ---
        SXX=$(bc -l <<< "($p + (2/3 * $D)) * 10000")
        SYYZZ=$(bc -l <<< "($p - (1/3 * $D)) * 10000")
        SXX_INT=$(printf "%.0f" "$SXX")
        SYYZZ_INT=$(printf "%.0f" "$SYYZZ")
        SED_EXTP="s/^[[:space:]]*EXTERNAL_PRESSURE.*/    EXTERNAL_PRESSURE $SXX_INT 0 0 0 $SYYZZ_INT 0 0 0 $SYYZZ_INT/"

        cd "$FOLDER" || exit

        # ========== STAGE 1: Cell opt (atoms fixed) ==========
        echo "  Stage 1: Cell optimization (atoms fixed)"
        cp init_cell.cell init1_cell.cell
        cp coordinates_init.xyz init1_coords.xyz
        cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run1.inp
        sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_s1/" run1.inp
        sed -i -E "$SED_EXTP" run1.inp
        sed -i 's/\r$//' run1.inp
        run_cp2k run1.inp run1.out
        LOG1=$(ls QM_s1-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LOG1" ]; then
            echo "Error: Stage 1 did not produce optimized_cell log"; exit 1
        fi
        extract_cell_from_log "$LOG1" cell_s1.cell
        cp cell_s1.cell init_cell.cell
        cleanup_stage_outputs "QM_s1"
        TARGET_P_BAR=$(printf "%.0f" "$(bc -l <<< "$p * 10000")")

        # ========== STAGE 2: Geo opt (positions only) ==========
        echo "  Stage 2: Geometry optimization (positions only)"
        cp init_cell.cell init2_cell.cell
        cp coordinates_init.xyz init2_coords.xyz
        cp "$SUBMIT_DIR/GeoOpt.inp" ./run2.inp
        sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_s2/" run2.inp
        sed -i 's/\r$//' run2.inp
        run_cp2k run2.inp run2.out
        if [ ! -f "QM_s2-POS-pos-1.xyz" ]; then
            echo "Error: Stage 2 did not produce trajectory"; exit 1
        fi
        extract_coords_to_scaled "QM_s2-POS-pos-1.xyz" "cell_s1.cell" "coords_s2.xyz"
        cleanup_stage_outputs "QM_s2"

        # ========== STAGE 3: Cell opt (atoms fixed) ==========
        echo "  Stage 3: Cell optimization (atoms fixed)"
        cp cell_s1.cell init_cell.cell
        cp coords_s2.xyz coordinates_init.xyz
        cp init_cell.cell init3_cell.cell
        cp coordinates_init.xyz init3_coords.xyz
        cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run3.inp
        sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_s3/" run3.inp
        sed -i -E "$SED_EXTP" run3.inp
        sed -i 's/\r$//' run3.inp
        run_cp2k run3.inp run3.out
        LOG3=$(ls QM_s3-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LOG3" ]; then
            echo "Error: Stage 3 did not produce optimized_cell log"; exit 1
        fi
        extract_cell_from_log "$LOG3" cell_s3.cell
        cleanup_stage_outputs "QM_s3"

        # ========== STAGE 4: Geo opt (positions only) ==========
        echo "  Stage 4: Geometry optimization (positions only)"
        cp cell_s3.cell init_cell.cell
        cp coords_s2.xyz coordinates_init.xyz
        cp init_cell.cell init4_cell.cell
        cp coordinates_init.xyz init4_coords.xyz
        cp "$SUBMIT_DIR/GeoOpt.inp" ./run4.inp
        sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_s4/" run4.inp
        sed -i 's/\r$//' run4.inp
        run_cp2k run4.inp run4.out
        if [ ! -f "QM_s4-POS-pos-1.xyz" ]; then
            echo "Error: Stage 4 did not produce trajectory"; exit 1
        fi
        extract_coords_to_scaled "QM_s4-POS-pos-1.xyz" "cell_s3.cell" "coords_s4.xyz"
        cleanup_stage_outputs "QM_s4"

        # ========== STAGE 5: Cell opt (atoms fixed) ==========
        echo "  Stage 5: Cell optimization (atoms fixed)"
        cp cell_s3.cell init_cell.cell
        cp coords_s4.xyz coordinates_init.xyz
        cp init_cell.cell init5_cell.cell
        cp coordinates_init.xyz init5_coords.xyz
        cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run5.inp
        sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_s5/" run5.inp
        sed -i -E "$SED_EXTP" run5.inp
        sed -i 's/\r$//' run5.inp
        run_cp2k run5.inp run5.out
        LOG5=$(ls QM_s5-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LOG5" ]; then
            echo "Error: Stage 5 did not produce optimized_cell log"; exit 1
        fi
        extract_cell_from_log "$LOG5" cell_s5.cell
        cleanup_stage_outputs "QM_s5"

        # ========== STAGE 6: Geo opt (positions only) ==========
        echo "  Stage 6: Geometry optimization (positions only)"
        cp cell_s5.cell init_cell.cell
        cp coords_s4.xyz coordinates_init.xyz
        cp init_cell.cell init6_cell.cell
        cp coordinates_init.xyz init6_coords.xyz
        cp "$SUBMIT_DIR/GeoOpt.inp" ./run6.inp
        sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_s6/" run6.inp
        sed -i 's/\r$//' run6.inp
        run_cp2k run6.inp run6.out
        if [ ! -f "QM_s6-POS-pos-1.xyz" ]; then
            echo "Error: Stage 6 did not produce trajectory"; exit 1
        fi
        extract_coords_to_scaled "QM_s6-POS-pos-1.xyz" "cell_s5.cell" "coords_s6.xyz"
        cleanup_stage_outputs "QM_s6"

        # ========== STAGE 7: Cell opt (atoms fixed) ==========
        echo "  Stage 7: Cell optimization (atoms fixed)"
        cp cell_s5.cell init_cell.cell
        cp coords_s6.xyz coordinates_init.xyz
        cp init_cell.cell init7_cell.cell
        cp coordinates_init.xyz init7_coords.xyz
        cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run7.inp
        sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_s7/" run7.inp
        sed -i -E "$SED_EXTP" run7.inp
        sed -i 's/\r$//' run7.inp
        run_cp2k run7.inp run7.out
        LOG7=$(ls QM_s7-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LOG7" ]; then
            echo "Error: Stage 7 did not produce optimized_cell log"; exit 1
        fi
        extract_cell_from_log "$LOG7" cell_s7.cell
        cleanup_stage_outputs "QM_s7"
        check_stall_and_scale run7.out cell_s7.cell "$TARGET_P_BAR" "$p" 7

        # ========== STAGE 8: Geo opt (positions only) ==========
        echo "  Stage 8: Geometry optimization (positions only)"
        cp cell_s7.cell init_cell.cell
        cp coords_s6.xyz coordinates_init.xyz
        cp init_cell.cell init8_cell.cell
        cp coordinates_init.xyz init8_coords.xyz
        cp "$SUBMIT_DIR/GeoOpt.inp" ./run8.inp
        sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_s8/" run8.inp
        sed -i 's/\r$//' run8.inp
        run_cp2k run8.inp run8.out
        if [ ! -f "QM_s8-POS-pos-1.xyz" ]; then
            echo "Error: Stage 8 did not produce trajectory"; exit 1
        fi
        extract_coords_to_scaled "QM_s8-POS-pos-1.xyz" "cell_s7.cell" "coords_s8.xyz"
        cleanup_stage_outputs "QM_s8"

        # ========== STAGE 9: Cell opt (atoms fixed) ==========
        echo "  Stage 9: Cell optimization (atoms fixed)"
        cp cell_s7.cell init_cell.cell
        cp coords_s8.xyz coordinates_init.xyz
        cp init_cell.cell init9_cell.cell
        cp coordinates_init.xyz init9_coords.xyz
        cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run9.inp
        sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_s9/" run9.inp
        sed -i -E "$SED_EXTP" run9.inp
        sed -i 's/\r$//' run9.inp
        run_cp2k run9.inp run9.out
        LOG9=$(ls QM_s9-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LOG9" ]; then
            echo "Error: Stage 9 did not produce optimized_cell log"; exit 1
        fi
        extract_cell_from_log "$LOG9" cell_s9.cell
        cleanup_stage_outputs "QM_s9"
        check_stall_and_scale run9.out cell_s9.cell "$TARGET_P_BAR" "$p" 9

        # ========== STAGE 10: Geo opt (positions only) ==========
        echo "  Stage 10: Geometry optimization (positions only)"
        cp cell_s9.cell init_cell.cell
        cp coords_s8.xyz coordinates_init.xyz
        cp init_cell.cell init10_cell.cell
        cp coordinates_init.xyz init10_coords.xyz
        cp "$SUBMIT_DIR/GeoOpt.inp" ./run10.inp
        sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_s10/" run10.inp
        sed -i 's/\r$//' run10.inp
        run_cp2k run10.inp run10.out
        if [ ! -f "QM_s10-POS-pos-1.xyz" ]; then
            echo "Error: Stage 10 did not produce trajectory"; exit 1
        fi
        extract_coords_to_scaled "QM_s10-POS-pos-1.xyz" "cell_s9.cell" "coords_s10.xyz"
        cleanup_stage_outputs "QM_s10"

        # ========== STAGE 11: Cell opt (atoms fixed) ==========
        echo "  Stage 11: Cell optimization (atoms fixed)"
        cp cell_s9.cell init_cell.cell
        cp coords_s10.xyz coordinates_init.xyz
        cp init_cell.cell init11_cell.cell
        cp coordinates_init.xyz init11_coords.xyz
        cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run11.inp
        sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_s11/" run11.inp
        sed -i -E "$SED_EXTP" run11.inp
        sed -i 's/\r$//' run11.inp
        run_cp2k run11.inp run11.out
        LOG11=$(ls QM_s11-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LOG11" ]; then
            echo "Error: Stage 11 did not produce optimized_cell log"; exit 1
        fi
        extract_cell_from_log "$LOG11" cell_s11.cell
        cleanup_stage_outputs "QM_s11"
        check_stall_and_scale run11.out cell_s11.cell "$TARGET_P_BAR" "$p" 11

        # ========== STAGE 12: Full unconstrained cell opt (max 30 steps, retry on non-convergence) ==========
        cp cell_s11.cell init_cell.cell
        cp coords_s10.xyz coordinates_init.xyz
        stage12_retry=0
        while true; do
            echo "  Stage 12: Full cell optimization (unconstrained)$([ "$stage12_retry" -gt 0 ] && echo " [attempt $stage12_retry]")"
            cp init_cell.cell init12_cell.cell
            cp coordinates_init.xyz init12_coords.xyz
            cp "$SUBMIT_DIR/CellOpt.inp" ./run12.inp
            sed -i -E 's/MAX_ITER 10000/MAX_ITER 30/' run12.inp
            sed -i -E "s/PROJECT QM_cellopt/PROJECT QM_s12/" run12.inp
            sed -i -E "$SED_EXTP" run12.inp
            sed -i 's/\r$//' run12.inp
            run_cp2k run12.inp CellOpt.out

            # Extract current state (last frame of trajectory)
            if [ -f "QM_s12-POS-pos-1.xyz" ]; then
                n_atoms=$(tac QM_s12-POS-pos-1.xyz | awk 'NF>=1 && $1+0==$1 {print int($1); exit}'); [ -z "$n_atoms" ] && n_atoms=24
                tail -n $((n_atoms + 2)) QM_s12-POS-pos-1.xyz > coordinates_final.xyz
            fi
            if ! extract_cell_from_restart "QM_s12" final_cell.cell; then
                LOG12=$(ls QM_s12-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
                if [ -n "$LOG12" ]; then
                    extract_cell_from_log "$LOG12" final_cell.cell
                else
                    echo "Error: Stage 12 did not produce optimized cell or restart"; exit 1
                fi
            fi

            if check_stage12_converged CellOpt.out; then
                cleanup_stage_outputs "QM_s12"
                # --- Single-point at final structure, extract stress tensor ---
                echo "  Single-point calculation at final structure"
                extract_coords_to_scaled "coordinates_final.xyz" "final_cell.cell" "coords_sp.xyz"
                cp "$SUBMIT_DIR/SinglePoint.inp" ./sp.inp
                sed -i "s/init_cell.cell/final_cell.cell/g" sp.inp
                sed -i "s/coordinates_init.xyz/coords_sp.xyz/g" sp.inp
                sed -i 's/\r$//' sp.inp
                run_cp2k sp.inp sp.out
                python3 "$(dirname "$BASE_DIR")/scripts/extract_stress.py" sp.out stress_tensor.txt
                break
            fi

            stage12_retry=$((stage12_retry + 1))
            if [ "$stage12_retry" -ge "$MAX_STAGE12_RETRIES" ]; then
                echo "Error: Stage 12 did not converge after $MAX_STAGE12_RETRIES retries"; exit 1
            fi

            cp CellOpt.out "CellOpt_attempt${stage12_retry}.out"
            n_cycles=$((2 ** stage12_retry))
            [ "$n_cycles" -gt 16 ] && n_cycles=16
            echo "  Stage 12 not converged; discarding result, using input coords/cell. Running $n_cycles cycles of Geo+Fixed..."
            # init_cell and coordinates_init are unchanged (stage 12 input)

            for ((rc=1; rc<=n_cycles; rc++)); do
                # Geo opt (positions only) - uses init_cell.cell, coordinates_init.xyz
                echo "    Recovery cycle $rc: Geo opt"
                cp "$SUBMIT_DIR/GeoOpt.inp" ./run_rc${rc}a.inp
                sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_rc${rc}a/" run_rc${rc}a.inp
                sed -i 's/\r$//' run_rc${rc}a.inp
                run_cp2k run_rc${rc}a.inp run_rc${rc}a.out
                if [ ! -f "QM_rc${rc}a-POS-pos-1.xyz" ]; then
                    echo "Error: Recovery geo opt did not produce trajectory"; exit 1
                fi
                extract_coords_to_scaled "QM_rc${rc}a-POS-pos-1.xyz" "init_cell.cell" "coords_rc${rc}a.xyz"
                cleanup_stage_outputs "QM_rc${rc}a"

                # Cell opt (atoms fixed) - uses init_cell.cell, coordinates_init.xyz
                echo "    Recovery cycle $rc: Cell opt (fixed)"
                cp coords_rc${rc}a.xyz coordinates_init.xyz
                cp "$SUBMIT_DIR/CellOpt_fixed.inp" ./run_rc${rc}b.inp
                sed -i -E "s/PROJECT QM_stage_cell/PROJECT QM_rc${rc}b/" run_rc${rc}b.inp
                sed -i -E "$SED_EXTP" run_rc${rc}b.inp
                sed -i 's/\r$//' run_rc${rc}b.inp
                run_cp2k run_rc${rc}b.inp run_rc${rc}b.out
                LOG_RC=$(ls QM_rc${rc}b-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)
                if [ -z "$LOG_RC" ]; then
                    echo "Error: Recovery cell opt did not produce optimized_cell log"; exit 1
                fi
                extract_cell_from_log "$LOG_RC" cell_rc${rc}b.cell
                cleanup_stage_outputs "QM_rc${rc}b"
                check_stall_and_scale run_rc${rc}b.out cell_rc${rc}b.cell "$TARGET_P_BAR" "$p" 12

                cp cell_rc${rc}b.cell init_cell.cell
                cp coords_rc${rc}a.xyz coordinates_init.xyz
            done
        done

        PREV_D="$D"
        cd "$BASE_DIR"
    done
done
