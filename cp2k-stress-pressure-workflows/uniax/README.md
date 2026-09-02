# Uniaxial CP2K workflows (Apptainer + Slurm)

Part of [`materials-science-workflows`](../../).

## Setup

1. Copy `../cluster.env.example` to `../cluster.env` and set:
   - `CP2K_SIF` — Apptainer image path on your cluster
   - `CP2K_SKF_PATH` — directory with 3ob SKF files (not in git)
2. Edit `#SBATCH --account=your_slurm_account` in submit scripts (or pass `--account=` to `sbatch`).
3. Run hydrostatic ramp first (`../cp2k/hydroopt`) so `cp2k/hydroopt/<P>gpa/final_cell.cell` exists.

Submit scripts source `../cp2k_env.sh` and patch `__CP2K_SKF_PATH__` in inputs at runtime.

## Layout

| Path | Role |
|------|------|
| `uniax_manual/scripts/manual_cell_opt.py` | FD stress matching → `parameter_trend_matrices.txt` |
| `uniax_manual/{a,b,cb}axis/` | Manual FD workflows per uniaxial axis |
| `{a,b,cb}axis/*submit.sh` | Legacy multi-stage uniax batch submits |
| `scripts/check_stall_and_scale.py` | Isotropic rescale when fixed-cell opt stalls |

## Verify Apptainer CP2K

```bash
apptainer exec "$CP2K_SIF" which cp2k.psmp
```

## Hydro path

Axis submits read hydrostatic structures from `${WORKFLOW_ROOT}/cp2k/hydroopt` (override with `HYDRO_DIR` in `cluster.env`).
