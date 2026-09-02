# CP2K ΔP Uniaxial Ramp (to 5 GPa)

Ramp each eigenvector-group axis from the hydrostatic structure up to **ΔP = 5 GPa** at constant mean pressure, minimizing enthalpy on a CP2K finite-difference surrogate at each step.

## Assumptions

- Base hydrostatic pressure **P_iso = 15 GPa** (default), with a validated base parameterization at  
  `uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt`.
- Hydrostatic coordinates come from `cp2k/hydroopt/<P>gpa/reopt/coordinates_final.xyz`.
- One lowest-enthalpy representative per 20° eigenvector group (~27 axes).

## Environment variables

| Variable | Role |
|----------|------|
| `CP2K_PARAM_TREND_PATH` | Base `parameter_trend_matrices.txt` for axis identification |
| `CENTER_CELL` / `CENTER_COORDS` | Center cell + coords for centered FD (`uniax_param_centered_fd.sh`) |
| `OUT_DIR` | Output directory for centered FD |
| `PRESSURE_GPA` | Nominal isotropic pressure [GPa] (default **15**) |
| `WORKFLOW_ROOT` | Repo root; sources `cp2k_env.sh` |
| `AXIS_OUT_DIR` | Per-axis ramp output root (`run_ramp_workflow.py`) |
| `START_CELL` / `START_COORDS` | Axis starting structure |
| `OPT_SCRIPT` | Surrogate optimizer (default: `scfhel/uniax_surrogate_optimizer_test.py`) |
| `RAMP_FULL_FD_GPAS` | Comma-separated ΔP values [GPa] that trigger full CP2K FD (default: `1,2,3,4,5`) |
| `AXES_MANIFEST` | Path to `axes_manifest.json` (job generator) |

Configure cluster paths in `cluster.env` (copy from `cluster.env.example`) before submitting.

See **[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md)** for SCP commands and ordered cluster steps.

## Run order (cluster)

1. **Axis identification** — set `CP2K_PARAM_TREND_PATH` to the base parameterization, then run the optimizer:
   ```bash
   export CP2K_PARAM_TREND_PATH=uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt
   python3 scfhel/uniax_surrogate_optimizer_test.py
   ```
   Produces `scfhel/minima_scale_1.0.json`.

2. **Build axis manifest** — one folder per eigenvector group:
   ```bash
   export CP2K_PARAM_TREND_PATH=...   # same as step 1
   export PRESSURE_GPA=15
   python3 scfhel/make_axes_manifest.py
   ```
   Writes `uniax/deltaPramp/axes_manifest.json` and `uniax/deltaPramp/axes/axis_<g>/`.

3. **Generate per-axis jobs**:
   ```bash
   export PRESSURE_GPA=15
   python3 uniax/deltaPramp/generate_ramp_jobs.py
   ```
   Creates `uniax/deltaPramp/pIso_150000bar/axis_<g>/run_axis.{sh,sbatch}` and `copypastesubmit.txt`.

4. **Submit** (~27 axis jobs):
   ```bash
   cd uniax/deltaPramp/pIso_150000bar
   bash copypastesubmit.txt
   ```

5. **Collect results** — per axis, read `deltaP_5p00/state.json` and `final_cell.cell` for the finished ΔP = 5 GPa enthalpy comparison.

## Ramp mechanics

- **Phase A** (≤2 cycles): CP2K centered FD + gradient descent to a self-consistent ΔP ≈ 0 minimum; convergence 0.01 Å / 0.05° on cell parameters.
- **Phase B** (1–5 GPa): **full CP2K FD re-fit at every integer GPa** (1, 2, 3, 4, 5), then GD to the uniaxial target on the fresh surrogate. Override with `RAMP_FULL_FD_GPAS=1,3,5` if you need fewer FD parameterizations.
- Enthalpy accumulation uses `enthalpy_like(x_opt, 0)` after each FD re-center (frame-consistent segments).

Each centered FD runs ~30 CP2K geo-opt jobs (base + 6 singles + strong pairs). Per axis expect up to **7** full parameterizations (2 Phase A + 5 Phase B) unless Phase A converges in one cycle.

## Files

| File | Purpose |
|------|---------|
| `scfhel/make_axes_manifest.py` | Group minima → axis folders + manifest |
| `uniax_param_centered_fd.sh` | `manual_cell_opt.py` prepare → `run_fd_sp.sh` → postprocess |
| `run_ramp_workflow.py` | Per-axis Phase A + ΔP ramp driver |
| `generate_ramp_jobs.py` | Slurm/bash job generation |
