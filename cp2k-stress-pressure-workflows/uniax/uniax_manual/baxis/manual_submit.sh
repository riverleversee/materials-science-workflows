#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --job-name=uniax-manual-b
#SBATCH --constraint=ib
#SBATCH --output=uniax_b.%j.out
#SBATCH --account=your_slurm_account

# --- Workflow paths (see cluster.env.example) ---
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${WORKFLOW_ROOT}/cp2k_env.sh"

# --- Load Environment ---
module purge
module load gcc/11.2.0 openmpi/4.1.1
module load apptainer 2>/dev/null || true

CP2K_SIF="${CP2K_SIF:-/path/to/cp2k.sif}"
CP2K_EXE="${CP2K_EXE:-/opt/cp2k/bin/cp2k.psmp}"
APPTAINER_CMD="${APPTAINER_CMD:-$(command -v apptainer 2>/dev/null || echo apptainer)}"

export UCX_TLS=self,cma,posix,sysv,rc,ud,tcp
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- Configuration (b-axis: run from uniax_manual/baxis) ---
# Use SLURM_SUBMIT_DIR when running under SLURM (script is copied to spool, $0 is wrong)
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}/baxis"
else
  BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
UNIAX_MANUAL="$(dirname "$BASE_DIR")"
SUBMIT_DIR="${UNIAX_MANUAL}/submitfiles"
SCRIPTS_DIR="${UNIAX_MANUAL}/scripts"
# hydroopt is a subfolder of CP2Kbenz
CP2KBENZ_DIR="${CP2KBENZ_DIR:-${CP2K_ROOT:-${WORKFLOW_ROOT}/cp2k}}"
HYDRO_DIR="${CP2KBENZ_DIR}/hydroopt"

source "${UNIAX_MANUAL}/config.sh"

# Ensure we run from the submit directory (SLURM default; needed if paths are relative)
[ -n "${SLURM_SUBMIT_DIR}" ] && cd "${SLURM_SUBMIT_DIR}" || true

START_GPA=15
END_GPA=40
DELTAS=(1 3 5 8 10)

# Run CP2K inside Apptainer (bind_base = parent of uniax_manual, same level as aaxis/baxis/cbaxis)
run_cp2k() {
    local inp="$1" out="$2"
    patch_cp2k_inp "$inp"
    local bind_base
    bind_base="$(cd "$(dirname "$UNIAX_MANUAL")" 2>/dev/null && pwd)" || bind_base="$(dirname "$UNIAX_MANUAL")"
    srun -n "${SLURM_NTASKS}" "$APPTAINER_CMD" exec \
        --sharens \
        --pwd "$(pwd)" \
        --bind "${bind_base}:${bind_base}" \
        "$CP2K_SIF" "$CP2K_EXE" -o "$out" -i "$inp"
}

# Create run_sp.sh and run_fd_cmd.txt for single-point runs
# run_fd_cmd.txt: srun command template for manual_cell_opt --prepare (run from bash, not Python)
create_run_sp_script() {
    local folder="$1"
    local bind_base
    bind_base="$(cd "$(dirname "$UNIAX_MANUAL")" 2>/dev/null && pwd)" || bind_base="$(dirname "$UNIAX_MANUAL")"
    local ntasks="${SLURM_NTASKS:-8}"
    local omp_threads="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
    cat > "$folder/run_sp.sh" << RUNSP
#!/bin/bash
set -e
inp="\$1"
out="\$2"
export OMP_NUM_THREADS=${omp_threads}
cd "$folder" || exit 1
srun -n ${ntasks} "$APPTAINER_CMD" exec \
    --sharens \
    --pwd "$folder" \
    --bind "${bind_base}:${bind_base}" \
    "$CP2K_SIF" "$CP2K_EXE" -o "\$out" -i "\$inp"
RUNSP
    chmod +x "$folder/run_sp.sh"
    cat > "$folder/run_fd_cmd.txt" << RUNFD
srun -n ${ntasks} "$APPTAINER_CMD" exec --sharens --pwd "$folder" --bind "${bind_base}:${bind_base}" "$CP2K_SIF" "$CP2K_EXE"
RUNFD
}

# --- Extract coords: last frame XYZ -> scaled fractional ---
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
    sys.exit(1)

with open(out_path, 'w') as f:
    for name, vec in [('A', cell[0]), ('B', cell[1]), ('C', cell[2])]:
        f.write(f"{name:<3} {vec[0]:18.15f} {vec[1]:18.15f} {vec[2]:18.15f}\n")
PY
    module purge
    module load gcc/11.2.0 openmpi/4.1.1
}

# --- Main Loop (b-axis: high stress along Y) ---
for (( p=$START_GPA; p<=$END_GPA; p++ )); do
    PREV_D=""
    for D in "${DELTAS[@]}"; do
        DELTA_DIR="${BASE_DIR}/${D}gpadelta"
        mkdir -p "$DELTA_DIR"
        FOLDER="${DELTA_DIR}/${p}gpa"
        echo "Processing b-axis Delta: $D GPa | Target P: $p GPa"

        mkdir -p "$FOLDER"
        if [ -z "$PREV_D" ]; then
            SRC="${HYDRO_DIR}/${p}gpa"
        else
            SRC="${BASE_DIR}/${PREV_D}gpadelta/${p}gpa"
        fi
        if [ ! -f "$SRC/final_cell.cell" ]; then
            echo "Error: Source $SRC/final_cell.cell not found!"; exit 1
        fi

        if [ -z "$PREV_D" ]; then
            # --- First delta: rotate cell so B along Y; fractional coords from hydro direct conversion (no extra rotation) ---
            module purge
            module load python
            module load anaconda
            python3 "$SCRIPTS_DIR/rotate_cell_and_convert.py" baxis \
                "$SRC/final_cell.cell" "$SRC/coordinates_final.xyz" \
                "$FOLDER/init_cell.cell" "$FOLDER/coordinates_init.xyz"
            module purge
            module load gcc/11.2.0 openmpi/4.1.1
        else
            # --- Subsequent deltas: use previous delta's final cell and coords ---
            cp "$SRC/final_cell.cell" "$FOLDER/init_cell.cell"
            extract_coords_to_scaled "$SRC/coordinates_final.xyz" "$SRC/final_cell.cell" "$FOLDER/coordinates_init.xyz"
        fi


        # --- Target stress (b-axis: Sxx=Szz, Syy high) ---
        SXXZ=$(bc -l <<< "($p - (1/3 * $D)) * 10000")
        SYY=$(bc -l <<< "($p + (2/3 * $D)) * 10000")
        SXXZ_I=$(printf "%.0f" "$SXXZ")
        SYY_I=$(printf "%.0f" "$SYY")
        TARGET_STRESS="$SXXZ_I $SYY_I $SXXZ_I 0 0 0"
        SED_EXTP="s/^[[:space:]]*EXTERNAL_PRESSURE.*/    EXTERNAL_PRESSURE $SXXZ_I 0 0 0 $SYY_I 0 0 0 $SXXZ_I/"
        TARGET_P_BAR=$(printf "%.0f" "$(bc -l <<< "$p * 10000")")

        create_run_sp_script "$FOLDER"
        cd "$FOLDER" || exit 1
        : > geoopt_progress.txt
        {
            echo "Initial cell:"
            cat init_cell.cell
            echo "==========="
        } >> geoopt_progress.txt

        cycle=0
        while true; do
            cycle=$((cycle + 1))
            echo "  Cycle $cycle: Manual cell step"
            cp init_cell.cell cell_before_manual.cell

            # Phase 1: Python prepares inputs + run_fd_sp.sh (no CP2K from Python)
            module load python
            module load anaconda
            python3 "$SCRIPTS_DIR/manual_cell_opt.py" prepare \
                --work-dir "$FOLDER" \
                --cell "$FOLDER/init_cell.cell" \
                --coords "$FOLDER/coordinates_init.xyz" \
                --target-stress $TARGET_STRESS \
                --delta-length-ang "$FD_DELTA_LENGTH_ANG" \
                --delta-angle "$FD_DELTA_ANGLE" \
                --axis baxis \
                --submit-dir "$SUBMIT_DIR" \
                --inp-template "GeoOpt_fd.inp" \
                --cycle "$cycle"
            # Phase 2: Bash runs srun directly (avoids 100x SCF slowdown from Python subprocess)
            ./run_fd_sp.sh
            # Phase 3: Python postprocesses outputs, writes new cell
            python3 "$SCRIPTS_DIR/manual_cell_opt.py" postprocess \
                --work-dir "$FOLDER" \
                --cell "$FOLDER/init_cell.cell" \
                --coords "$FOLDER/coordinates_init.xyz" \
                --target-stress $TARGET_STRESS \
                --step-fraction "$STEP_FRACTION" \
                --delta-length-ang "$FD_DELTA_LENGTH_ANG" \
                --delta-angle "$FD_DELTA_ANGLE" \
                --submit-dir "$SUBMIT_DIR" \
                --max-delta-length-ang "$MAX_DELTA_LENGTH_ANG" \
                --max-delta-angle-deg "$MAX_DELTA_ANGLE_DEG" \
                --cycle "$cycle" \
                --progress-log "geoopt_progress.txt"
            python3 "$SCRIPTS_DIR/rotate_cell_preserve_frac.py" baxis \
                cell_before_manual.cell coordinates_init.xyz \
                init_cell.cell init_cell.cell coordinates_init.xyz
            module purge
            module load gcc/11.2.0 openmpi/4.1.1

            echo "  Cycle $cycle: Geo opt"
            cp "$SUBMIT_DIR/GeoOpt.inp" ./run_geo.inp
            sed -i -E "s/PROJECT QM_stage_geo/PROJECT QM_geo/" run_geo.inp
            sed -i 's/\r$//' run_geo.inp
            run_cp2k run_geo.inp run_geo.out
            if [ ! -f "QM_geo-POS-pos-1.xyz" ]; then
                echo "Error: Geo opt did not produce trajectory"; exit 1
            fi
            extract_coords_to_scaled "QM_geo-POS-pos-1.xyz" "init_cell.cell" "coordinates_init.xyz"

            module load python
            module load anaconda
            python3 "$SCRIPTS_DIR/log_geoopt_progress.py" --step "$cycle" --geo-out run_geo.out \
                --cell init_cell.cell --coords QM_geo-POS-pos-1.xyz --output geoopt_progress.txt

            echo "  Cycle $cycle: Checking stress convergence"
            if python3 "$SCRIPTS_DIR/check_stress_converged.py" run_geo.out $TARGET_STRESS --tolerance "$STRESS_TOL_BAR" 2>/dev/null; then
                echo "  Stress converged."
                module purge
                module load gcc/11.2.0 openmpi/4.1.1
                cp init_cell.cell final_cell.cell
                cp coordinates_init.xyz coordinates_final.xyz
                extract_coords_to_scaled "coordinates_final.xyz" "final_cell.cell" "coords_sp.xyz"
                cp "$SUBMIT_DIR/SinglePoint.inp" ./sp.inp
                sed -i "s/init_cell.cell/final_cell.cell/g" sp.inp
                sed -i "s/coordinates_init.xyz/coords_sp.xyz/g" sp.inp
                sed -i 's/\r$//' sp.inp
                run_cp2k sp.inp sp.out
                python3 "$SCRIPTS_DIR/extract_stress.py" sp.out stress_tensor.txt
                break
            fi

            if [ "$cycle" -ge "$MAX_MANUAL_CYCLES" ]; then
                echo "Error: Max manual cycles ($MAX_MANUAL_CYCLES) reached without stress convergence"; exit 1
            fi
        done

        PREV_D="$D"
        cd "$BASE_DIR" || exit 1
    done
done
