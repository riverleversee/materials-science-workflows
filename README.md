# Materials Science Workflows (private)

**CP2K stress/pressure workflows** — hydrostatic ramp, uniaxial FD parameterization, and scfhel surrogate search.

> **Note:** Energetic normal-mode analysis lives in the public repo [energetic-normal-mode-analysis](https://github.com/riverleversee/energetic-normal-mode-analysis). This private repo holds CP2K tooling until it is ready to publish.

Perovskite and standalone DFTB+ benzene tracks are intentionally excluded.

## Layout

| Folder | Description |
|--------|-------------|
| [`cp2k-stress-pressure-workflows/`](cp2k-stress-pressure-workflows/) | CP2K internal DFTB: hydrostatic ramp, Apptainer uniax FD, scfhel surrogate search |

## First-time cluster setup

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

## Requirements

CP2K (module or Apptainer), Python 3.10+, `numpy`, Slurm on HPC.

## Not in git

Per-pressure CP2K run trees, restart/output files, SKF parameter files, and full `parameter_trend_matrices.txt` logs. `uniax/deltaPramp/` is WIP and gitignored.

## Provenance

Staged from local `CursorCoding` trees (2026-09). Original paths are listed in each subfolder README.
