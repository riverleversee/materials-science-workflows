#!/bin/bash
# CP2K centered finite-difference parameterization at an arbitrary cell.
#
# Required env:
#   CENTER_CELL   — path to center final_cell.cell
#   CENTER_COORDS — path to coordinates_final.xyz
#   OUT_DIR       — output directory (parameter_trend_matrices.txt written here)
#
# Optional env:
#   PRESSURE_GPA  — nominal isotropic pressure [GPa] for H_base (default: 15)
#   WORKFLOW_ROOT — repo root (auto-detected if unset)
#   CP2K_AXIS     — axis frame for manual_cell_opt (default: cbaxis)
set -euo pipefail

: "${CENTER_CELL:?CENTER_CELL required}"
: "${CENTER_COORDS:?CENTER_COORDS required}"
: "${OUT_DIR:?OUT_DIR required}"

PRESSURE_GPA="${PRESSURE_GPA:-15}"
WORKFLOW_ROOT="${WORKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CP2K_AXIS="${CP2K_AXIS:-cbaxis}"

# shellcheck source=/dev/null
source "${WORKFLOW_ROOT}/cp2k_env.sh"

UNIAX_MANUAL="${WORKFLOW_ROOT}/uniax/uniax_manual"
SCRIPTS_DIR="${UNIAX_MANUAL}/scripts"
SUBMIT_DIR="${UNIAX_MANUAL}/submitfiles"

# shellcheck source=/dev/null
source "${UNIAX_MANUAL}/config.sh"

module purge
module load gcc/11.2.0 openmpi/4.1.1 2>/dev/null || true
module load apptainer 2>/dev/null || true
module load python 2>/dev/null || true
module load anaconda 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
export SLURM_EXPORT_ENV=ALL

mkdir -p "${OUT_DIR}"
WORK_DIR="${OUT_DIR}/center_fd"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

cp "${CENTER_CELL}" init_cell.cell
cp "${CENTER_COORDS}" coordinates_init.xyz

P_ISO_BAR="$(python3 -c "print(int(round(float('${PRESSURE_GPA}') * 10000)))")"
TARGET_STRESS="${P_ISO_BAR} ${P_ISO_BAR} ${P_ISO_BAR} 0 0 0"

# srun / apptainer command template for run_fd_sp.sh
BIND_BASE="$(cd "$(dirname "${UNIAX_MANUAL}")" 2>/dev/null && pwd)" || BIND_BASE="$(dirname "${UNIAX_MANUAL}")"
NTASKS="${SLURM_NTASKS:-8}"
OMP_THREADS="${OMP_NUM_THREADS}"
cat > run_fd_cmd.txt <<EOF
srun -n ${NTASKS} "${APPTAINER_CMD}" exec --sharens --pwd "${WORK_DIR}" --bind "${BIND_BASE}:${BIND_BASE}" "${CP2K_SIF}" "${CP2K_EXE}"
EOF

echo "=== CP2K centered FD: prepare ==="
python3 "${SCRIPTS_DIR}/manual_cell_opt.py" prepare \
    --work-dir "${WORK_DIR}" \
    --cell "${WORK_DIR}/init_cell.cell" \
    --coords "${WORK_DIR}/coordinates_init.xyz" \
    --target-stress ${TARGET_STRESS} \
    --delta-length-ang "${FD_DELTA_LENGTH_ANG}" \
    --delta-angle "${FD_DELTA_ANGLE}" \
    --axis "${CP2K_AXIS}" \
    --submit-dir "${SUBMIT_DIR}" \
    --inp-template "GeoOpt_fd.inp" \
    --cycle 1 \
    --no-full-coupling-first-step

echo "=== CP2K centered FD: run_fd_sp.sh ==="
bash ./run_fd_sp.sh

echo "=== CP2K centered FD: postprocess ==="
python3 "${SCRIPTS_DIR}/manual_cell_opt.py" postprocess \
    --work-dir "${WORK_DIR}" \
    --cell "${WORK_DIR}/init_cell.cell" \
    --coords "${WORK_DIR}/coordinates_init.xyz" \
    --target-stress ${TARGET_STRESS} \
    --step-fraction 0.0 \
    --delta-length-ang "${FD_DELTA_LENGTH_ANG}" \
    --delta-angle "${FD_DELTA_ANGLE}" \
    --submit-dir "${SUBMIT_DIR}" \
    --max-delta-length-ang 0.0 \
    --max-delta-angle-deg 0.0 \
    --cycle 1 \
    --no-full-coupling-first-step \
    --trend-log "${OUT_DIR}/parameter_trend_matrices.txt"

if [ ! -f "${OUT_DIR}/parameter_trend_matrices.txt" ]; then
    echo "ERROR: parameter_trend_matrices.txt not written to ${OUT_DIR}" >&2
    exit 1
fi

echo "Wrote ${OUT_DIR}/parameter_trend_matrices.txt"
