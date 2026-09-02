# materials-science-workflows

**Status: project in progress (WIP)** — CP2K ΔP uniaxial ramp is under active cluster development on CURC Alpine; energetic NMA scripts are included and usable.

## Layout

| Folder | Description |
|--------|-------------|
| [`energetic-normal-mode-analysis/`](energetic-normal-mode-analysis/) | BNFF/DNTF normal-mode coupling and functional-group decomposition under pressure |
| [`cp2k-stress-pressure-workflows/`](cp2k-stress-pressure-workflows/) | CP2K internal DFTB hydrostatic + uniax FD + `scfhel` surrogate + ΔP ramp (`uniax/deltaPramp`) |
| [`perovskite-workflows/`](perovskite-workflows/) | Mixed-halide perovskite classical MD + IR scripting (**heavily redacted** subset) |

A standalone public copy of the NMA tree may still exist at [energetic-normal-mode-analysis](https://github.com/riverleversee/energetic-normal-mode-analysis); **this monorepo is the working home**.

## CP2K ΔP ramp (current focus)

Cluster runbook: [`cp2k-stress-pressure-workflows/uniax/deltaPramp/CLUSTER_RUNBOOK.md`](cp2k-stress-pressure-workflows/uniax/deltaPramp/CLUSTER_RUNBOOK.md).

**In progress / not done yet**

- Full Alpine production run across all axes to `deltaP_5p00`
- Result packaging / figures for publication

**Do not commit:** `cluster.env`, generated CP2K `pIso_*` / `axes/` / `minima_scale_*.json`, or large NMA trajectory trees.
