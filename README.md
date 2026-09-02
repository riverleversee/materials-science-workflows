# Materials Science Workflows

Non-perovskite computational materials workflows: **pressure-dependent normal-mode analysis** on energetic crystals and **CP2K-based stress optimization** (hydrostatic → anisotropic → surrogate search).

Perovskite and standalone DFTB+ benzene tracks are intentionally excluded.

## Layout

| Folder | Description |
|--------|-------------|
| [`energetic-normal-mode-analysis/`](energetic-normal-mode-analysis/) | BNFF/DNTF normal-mode coupling and functional-group decomposition |
| [`cp2k-stress-pressure-workflows/`](cp2k-stress-pressure-workflows/) | CP2K internal DFTB: hydrostatic ramp, Apptainer uniax FD, scfhel surrogate search |

## First-time cluster setup (CP2K stack)

```bash
cd cp2k-stress-pressure-workflows
cp cluster.env.example cluster.env
# Edit cluster.env: CP2K_SIF, CP2K_SKF_PATH, CP2K_PARAM_TREND_PATH
```

Place 3ob SKF files under `cp2k/skfdatafiles/` on the cluster (or set `CP2K_SKF_PATH` elsewhere). SKF data is **not** in git.

Edit `#SBATCH --account=your_slurm_account` in submit scripts before running.

## Pipeline order

```
cp2k/initopt → cp2k/hydroopt (1–40 GPa)
    → uniax/uniax_manual (FD parameterization)
    → scfhel/ (surrogate optimizer on parameter_trend_matrices.txt)
```

## Requirements (summary)

| Component | Needs |
|-----------|--------|
| Normal-mode analysis | Python 3.10+, `numpy`, `matplotlib`; **plots in `results/minpress/`**; trajectories local-only to rerun |
| CP2K workflows | CP2K (module or Apptainer), Python 3.10+, `numpy`, Slurm on HPC |

## Not in git

Mode animation trajectories (`anime_*.xyz`), per-pressure CP2K run trees, restart/output files, SKF parameter files, and full `parameter_trend_matrices.txt` logs. Energetic NMA **result figures** are under `energetic-normal-mode-analysis/results/minpress/`.

## Provenance

Staged from local `CursorCoding` trees (2026-09). Original paths are listed in each subfolder README.
