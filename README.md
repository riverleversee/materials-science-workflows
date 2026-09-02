# materials-science-workflows

**Status: project in progress (WIP)** — refers to the CP2K ΔP uniaxial ramp on CURC Alpine. Energetic NMA, QE benzene, and NPL core/shell workflows here are finished work packaged for reuse.

## Layout

| Folder | Description |
|--------|-------------|
| [`energetic-normal-mode-analysis/`](energetic-normal-mode-analysis/) | BNFF/DNTF normal-mode coupling and functional-group decomposition under pressure |
| [`cp2k-stress-pressure-workflows/`](cp2k-stress-pressure-workflows/) | CP2K internal DFTB hydrostatic + uniax FD + `scfhel` surrogate + ΔP ramp (`uniax/deltaPramp`); **active WIP** |
| [`qe-benzene-pressure/`](qe-benzene-pressure/) | Small QE benzene hydro/uniax/conver showcase + modified `cell_base.f90` |
| [`npl-core-shell-pressure/`](npl-core-shell-pressure/) | Finished CP2K CdSe/CdZnS nanoplatelet pressure workflows (JACS production tree + ZB bulk bands); [DOI 10.1021/jacs.5c14939](https://doi.org/10.1021/jacs.5c14939) |
| [`perovskite-workflows/`](perovskite-workflows/) | Mixed-halide perovskite classical MD + IR scripting (**heavily redacted** subset) |

A standalone copy of the NMA tree may still exist at [energetic-normal-mode-analysis](https://github.com/riverleversee/energetic-normal-mode-analysis); **this monorepo is the working home**.

## CP2K ΔP ramp (current focus)

Cluster runbook: [`cp2k-stress-pressure-workflows/uniax/deltaPramp/CLUSTER_RUNBOOK.md`](cp2k-stress-pressure-workflows/uniax/deltaPramp/CLUSTER_RUNBOOK.md).

**In progress / not done yet**

- Full Alpine production run across all axes to `deltaP_5p00`
- Result packaging / figures for publication

**Do not commit:** `cluster.env`, generated CP2K `pIso_*` / `axes/` / `minima_scale_*.json`, or large NMA trajectory trees.
