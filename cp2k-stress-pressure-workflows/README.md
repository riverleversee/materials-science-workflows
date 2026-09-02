# CP2K Stress & Pressure Workflows

Part of the [`materials-science-workflows`](../../) repo.

**Isotropic hydrostatic pressure optimization**, **anisotropic (uniaxial) CP2K cell optimization**, and **scfhel** surrogate search — CP2K internal DFTB on benzene under pressure.

DFTB+ standalone workflows are intentionally omitted.

## Configuration

See [`cluster.env.example`](cluster.env.example) and [`cp2k_env.sh`](cp2k_env.sh). All submit scripts source `cp2k_env.sh` and substitute `__CP2K_SKF_PATH__` in CP2K inputs at run time.

Set `CP2K_PARAM_TREND_PATH` for scfhel optimizers (path to a `parameter_trend_matrices.txt` from a uniax FD run).

## Pipeline

```
cp2k/initopt          → zero-pressure CELL_OPT
cp2k/hydroopt         → isotropic EXTERNAL_PRESSURE ramp (1–40 GPa)
uniax/uniax_manual    → finite-difference stress matching
uniax/{a,b,cb}axis    → staged uniaxial CP2K submits (Apptainer)
scfhel/               → surrogate build/search on CP2K FD outputs
```

## Requirements

- CP2K (cluster module or Apptainer)
- Python 3.10+, `numpy`
- Slurm + Apptainer on HPC
- 3ob SKF parameter files (local/cluster only)

## Not in git

Per-pressure run directories, CP2K restart/output files, SKF data, and full parameter trend logs.
