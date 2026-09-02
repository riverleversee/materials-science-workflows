# Cluster runbook — CP2K ΔP ramp (CURC Alpine)

**Target system:** [CURC](https://curc.readthedocs.io/) **Alpine** cluster (CU Research Computing).  
Login: `login.rc.colorado.edu` · Load scheduler: `module load slurm/alpine` before any `sbatch` / `squeue`.

Run commands **in order** on a CURC login node unless noted. Edit the **Configuration** block once, then copy/paste each section.

**Do not** upload `uniax/deltaPramp/pIso_*` from your laptop/WSL — those folders contain **local absolute paths**. Regenerate them on Alpine (steps 3–4).

---

## Configuration (edit once)

Run on your **local/WSL** machine when transferring files:

```bash
# --- Local (WSL) ---
LOCAL_REPO="/mnt/d/Github candidates/materials-science-workflows/cp2k-stress-pressure-workflows"

# --- CURC Alpine login ---
CLUSTER_USER="rile5166"                    # your RC username
CLUSTER_HOST="login.rc.colorado.edu"
CLUSTER="${CLUSTER_USER}@${CLUSTER_HOST}"

# --- Alpine scratch (workflow root) ---
SCRATCH="/scratch/alpine/${CLUSTER_USER}/cp2k-stress-pressure-workflows"

# --- Alpine projects (conda, Apptainer image, long-lived software) ---
PROJECTS="/projects/${CLUSTER_USER}"

# --- Physics / workflow ---
PRESSURE_GPA=15
P_ISO_BAR=150000
CP2K_AXIS=cbaxis   # cbaxis | aaxis | baxis — match your uniax FD run

# --- Slurm on Alpine (match your existing CP2Kbenz / dftbbenz jobs) ---
SLURM_ACCOUNT="ucb-general"    # Trailhead auto-allocation; was ucb357_asc3 (may be expired)
SLURM_PARTITION="acpu"         # was amilan (renamed Aug 2026)
SLURM_QOS="cpu-normal"         # was normal
SLURM_CONSTRAINT=""            # leave empty unless you need a feature
CONDA_ENV="${PROJECTS}/software/dftb_22"   # existing env with numpy; or your CP2K env
CP2K_SIF="${PROJECTS}/software/cp2kapptainer/cp2k_latest.sif"
```

On **Alpine** (after `ssh login.rc.colorado.edu`):

```bash
module load slurm/alpine

export SCRATCH="/scratch/alpine/${USER}/cp2k-stress-pressure-workflows"
export PROJECTS="/projects/${USER}"
export WORKFLOW_ROOT="${SCRATCH}"
export PRESSURE_GPA=15
cd "${WORKFLOW_ROOT}"
```

---

## Phase A — Transfer to cluster (run on WSL / local)

### A1. Sync workflow scripts (exclude run outputs)

```bash
rsync -avz --progress \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'uniax/deltaPramp/pIso_*' \
  --exclude 'uniax/deltaPramp/axes/' \
  --exclude 'uniax/deltaPramp/axes_manifest.json' \
  --exclude 'scfhel/minima_scale_*.json' \
  --exclude 'cluster.env' \
  "${LOCAL_REPO}/" \
  "${CLUSTER}:${SCRATCH}/"
```

### A2. Copy `cluster.env` (secrets / site paths — not in git)

```bash
scp "${LOCAL_REPO}/cluster.env" "${CLUSTER}:${SCRATCH}/cluster.env"
```

If you do not have `cluster.env` yet, SCP the example and edit on cluster:

```bash
scp "${LOCAL_REPO}/cluster.env.example" "${CLUSTER}:${SCRATCH}/cluster.env"
```

### A3. Copy 3ob SKF files (if not already on cluster)

```bash
# Example — point LOCAL_SKF at your SKF directory
LOCAL_SKF="/path/to/skfdatafiles"
scp -r "${LOCAL_SKF}/" "${CLUSTER}:${SCRATCH}/cp2k/skfdatafiles/"
```

### A4. Copy base hydrostatic structures at P_iso = 15 GPa

Required for `make_axes_manifest.py` (`HYDRO_COORDS`). Use your validated hydroopt output:

```bash
# Example from D: CursorCoding CP2Kbenz hydroopt
LOCAL_HYDRO="/mnt/d/CursorCoding/CP2Kbenz/hydroopt/15gpa"
scp "${LOCAL_HYDRO}/coordinates_final.xyz" \
    "${CLUSTER}:${SCRATCH}/cp2k/hydroopt/15gpa/coordinates_final.xyz"
scp "${LOCAL_HYDRO}/final_cell.cell" \
    "${CLUSTER}:${SCRATCH}/cp2k/hydroopt/15gpa/final_cell.cell"
```

Create the directory on cluster first if needed:

```bash
ssh "${CLUSTER}" "mkdir -p ${SCRATCH}/cp2k/hydroopt/15gpa"
```

### A5. Copy validated base `parameter_trend_matrices.txt`

Required for axis identification (`CP2K_PARAM_TREND_PATH`). From your existing CP2K FD run:

```bash
LOCAL_TREND="/mnt/d/CursorCoding/CP2Kbenz/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt"
scp "${LOCAL_TREND}" \
    "${CLUSTER}:${SCRATCH}/uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt"
```

```bash
ssh "${CLUSTER}" "mkdir -p ${SCRATCH}/uniax/uniax_manual/cbaxis/1gpadelta/15gpa"
```

*(Alternatively: run hydroopt + uniax_manual FD on the cluster first — see [Prerequisites](#prerequisites-if-not-done-yet).)*

---

## Phase B — One-time setup on CURC Alpine

```bash
ssh "${CLUSTER}"
module load slurm/alpine

export SCRATCH="/scratch/alpine/${USER}/cp2k-stress-pressure-workflows"
export PROJECTS="/projects/${USER}"
export WORKFLOW_ROOT="${SCRATCH}"
cd "${WORKFLOW_ROOT}"
```

### B1. Create `cluster.env` (or run setup script)

```bash
bash setup_alpine.sh
```

That copies `cluster.env.example` → `cluster.env` and checks SKF / hydro / trend / SIF.
If you already hand-edited `cluster.env`, skip the copy and run checks only:

```bash
source cp2k_env.sh
test -f "${CP2K_PARAM_TREND_PATH}" && echo "trend OK"
test -f "${HYDRO_DIR}/15gpa/coordinates_final.xyz" && echo "hydro OK"
test -f "${CP2K_SIF}" && echo "SIF OK"
```

Example `cluster.env` (points at existing `/projects/$USER/CP2Kbenz` data):

```bash
export CP2K_SIF=/projects/rile5166/software/cp2kapptainer/cp2k_latest.sif
export CP2K_EXE=/opt/cp2k/bin/cp2k.psmp
export APPTAINER_CMD=/usr/bin/apptainer
export CP2K_SKF_PATH=/projects/rile5166/CP2Kbenz/skfdatafiles
export HYDRO_DIR=/projects/rile5166/CP2Kbenz/hydroopt
export CP2K_PARAM_TREND_PATH=/projects/rile5166/CP2Kbenz/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt
```

Apptainer is **not** on Alpine login nodes; CP2K runs via `/usr/bin/apptainer` on **compute nodes** inside Slurm jobs.

### B2. Python environment

```bash
module purge
module load slurm/alpine
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-/projects/${USER}/software/dftb_22}"
python3 -c "import numpy; print('numpy OK')"
chmod +x "${WORKFLOW_ROOT}/uniax/deltaPramp/uniax_param_centered_fd.sh"
```

---

## Phase C — Workflow commands (in order on Alpine)

**Before every Slurm command:** `module load slurm/alpine`

### Step 1 — Axis identification (surrogate only, no CP2K)

Produces `scfhel/minima_scale_1.0.json`. CPU + memory; ~tens of minutes.

```bash
module load slurm/alpine
cd "${WORKFLOW_ROOT}"
sbatch scfhel/run_axis_id.sbatch
# wait for job; then:
test -f scfhel/minima_scale_1.0.json && echo "Step 1 OK"
```

*(The sbatch file is in the repo — no heredoc needed.)*

---

### Step 2–3 — Build axes + generate Slurm jobs

```bash
bash uniax/deltaPramp/run_prep_on_alpine.sh
```

Or run the two Python commands manually (see git history). Then continue with Step 4.

---

### Step 4 — Verify Slurm headers (usually no edit needed)

Generated jobs default to **`ucb-general` / `acpu` / `cpu-normal`**. Confirm:

```bash
cd "${WORKFLOW_ROOT}/uniax/deltaPramp/pIso_150000bar"

grep SBATCH axis_0/run_axis.sbatch
grep WORKFLOW_ROOT axis_0/run_axis.sh | head -1
# WORKFLOW_ROOT must be /scratch/alpine/$USER/..., not /mnt/d/...
```

If your allocation differs, patch once:

```bash
find . -name 'run_axis.sbatch' -print0 | xargs -0 sed -i \
  -e 's|#SBATCH --account=ucb-general|#SBATCH --account=YOUR_ALLOC|g'
```

---

### Step 5 — Submit all axis ramp jobs (CP2K — long)

Each axis: Phase A (≤2 FD) + Phase B (FD at ΔP = 1,2,3,4,5 GPa). **24 h** walltime (Alpine `acpu` / `cpu-normal` max).

```bash
module load slurm/alpine
cd "${WORKFLOW_ROOT}/uniax/deltaPramp/pIso_150000bar"

bash copypastesubmit.txt | tee submit.log

squeue -u "${USER}" | grep cp2k-ramp
```

Submit a **single test axis** first (recommended):

```bash
sbatch axis_0/run_axis.sbatch
# watch: tail -f axis_0/cp2k-ramp-g0.*.out  (or slurm output in axis_0/)
# confirm deltaP_1p00/ ... deltaP_5p00/ appear before submitting all
```

---

### Step 6 — Monitor

```bash
cd "${WORKFLOW_ROOT}/uniax/deltaPramp/pIso_150000bar"

# Job queue
squeue -u "${USER}"

# One axis progress
AXIS=axis_0
ls -la "${AXIS}"/min_refine/ 2>/dev/null
ls -d "${AXIS}"/deltaP_* 2>/dev/null
cat "${AXIS}/deltaP_5p00/state.json" 2>/dev/null
```

Expected per-axis tree when complete:

```
axis_<g>/
  min_refine/iter_1/ ...          # Phase A FD + GD
  phase_a_end/                    # handoff snapshot
  deltaP_1p00/ ... deltaP_5p00/ # each with state.json, final_cell.cell
```

---

### Step 7 — Summarize results on cluster

```bash
cd "${WORKFLOW_ROOT}/uniax/deltaPramp/pIso_150000bar"

python3 - << 'PY'
import json
from pathlib import Path
root = Path(".")
rows = []
for d in sorted(root.glob("axis_*")):
    st = d / "deltaP_5p00" / "state.json"
    if not st.is_file():
        rows.append((d.name, "INCOMPLETE", "", ""))
        continue
    s = json.loads(st.read_text())
    rows.append((d.name, "OK", s.get("Hlike_total_Ha"), s.get("deltaP_GPa")))
print(f"{'axis':12} {'status':12} {'Hlike_total_Ha':>18} {'deltaP_GPa'}")
for r in rows:
    print(f"{r[0]:12} {r[1]:12} {str(r[2]):>18} {r[3]}")
PY
```

---

## Phase D — Pull results back to local (run on WSL)

```bash
LOCAL_REPO="/mnt/d/Github candidates/materials-science-workflows/cp2k-stress-pressure-workflows"
CLUSTER="rile5166@login.rc.colorado.edu"
SCRATCH="/scratch/alpine/rile5166/cp2k-stress-pressure-workflows"

mkdir -p "${LOCAL_REPO}/results/deltaPramp_pIso_150000bar"

rsync -avz --progress \
  "${CLUSTER}:${SCRATCH}/uniax/deltaPramp/pIso_150000bar/" \
  "${LOCAL_REPO}/results/deltaPramp_pIso_150000bar/"

# Optional: minima + manifest + logs
scp "${CLUSTER}:${SCRATCH}/scfhel/minima_scale_1.0.json" \
    "${LOCAL_REPO}/scfhel/"
scp "${CLUSTER}:${SCRATCH}/uniax/deltaPramp/axes_manifest.json" \
    "${LOCAL_REPO}/uniax/deltaPramp/"
```

---

## Prerequisites (if not done yet)

Run **before** Step 1 if you do not have hydrostatic 15 GPa structures or base FD trend file on cluster:

| Prerequisite | Location on cluster | How |
|--------------|---------------------|-----|
| Zero-P cell | `cp2k/initopt/` | `cp2k/initopt` submit scripts |
| Hydrostatic ramp | `cp2k/hydroopt/15gpa/` | `cp2k/hydroopt/CP2Kpressureall.sh` |
| Base FD trend | `uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt` | `uniax/uniax_manual/cbaxis/manual_submit.sh` |

See repo [`README.md`](../../README.md) and [`uniax/README.md`](../README.md).

---

## Environment reference (ramp jobs)

Set in `run_axis.sh` (auto-generated) or override before `sbatch`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKFLOW_ROOT` | `${SCRATCH}` | Repo root |
| `PRESSURE_GPA` | `15` | Nominal P_iso for FD enthalpy reference |
| `RAMP_FULL_FD_GPAS` | `1,2,3,4,5` | GPa checkpoints with full CP2K FD |
| `CP2K_AXIS` | `cbaxis` | Passed to centered FD wrapper |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `sbatch: command not found` | Run `module load slurm/alpine` |
| Paths contain `/mnt/d/` | Re-run **Steps 2–3 on Alpine**; do not use WSL-generated `pIso_*` |
| `Invalid account` | Check allocation: `rc --account=` or CURC support; patch Step 4 |
| `parameter_trend_matrices.txt not found` | Complete prerequisite FD; set `CP2K_PARAM_TREND_PATH` in `cluster.env` |
| `coordinates_final.xyz not found` | SCP hydro 15 GPa coords (Phase A4) or run hydroopt |
| CP2K fails in Apptainer | Check `CP2K_SIF` under `/projects/$USER/software/`; `module load apptainer` |
| Job timeout | One axis ≈ 7 FD × ~30 CP2K runs; request `long` QOS if >24 h needed on Alpine |
| SCF slow inside Python | Ramp uses bash FD wrapper — do **not** call CP2K from nested Python |

---

## Quick checklist

- [ ] A1–A5: rsync repo + `cluster.env` + SKF + hydro coords + base trend
- [ ] B1–B2: `cluster.env` verified, `uniax_param_centered_fd.sh` executable
- [ ] Step 1: `scfhel/minima_scale_1.0.json`
- [ ] Step 2: `uniax/deltaPramp/axes_manifest.json` + `axes/axis_*`
- [ ] Step 3: `pIso_150000bar/copypastesubmit.txt` with **cluster** paths
- [ ] Step 4: Slurm account/partition patched
- [ ] Step 5: Test `axis_0`, then full `copypastesubmit.txt`
- [ ] Step 6–7: All `deltaP_5p00/state.json` present
- [ ] Phase D: rsync results home
