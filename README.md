# materials-science-workflows

**Status: project in progress (WIP)** — CP2K ΔP uniaxial ramp is under active cluster development on CURC Alpine; not a finished publishable result set yet.

## Layout

| Folder | Description |
|--------|-------------|
| [`cp2k-stress-pressure-workflows/`](cp2k-stress-pressure-workflows/) | CP2K internal DFTB hydrostatic + uniax FD + `scfhel` surrogate + ΔP ramp (`uniax/deltaPramp`) |

Published separately:

- **Energetic normal-mode analysis (public):** https://github.com/riverleversee/energetic-normal-mode-analysis

## CP2K ΔP ramp (current focus)

Workflow scripts for a ~25-axis ΔP = 1–5 GPa uniaxial ramp at fixed mean pressure. Cluster runbook: [`cp2k-stress-pressure-workflows/uniax/deltaPramp/CLUSTER_RUNBOOK.md`](cp2k-stress-pressure-workflows/uniax/deltaPramp/CLUSTER_RUNBOOK.md).

**In progress / not done yet**

- Full Alpine production run across all axes to `deltaP_5p00`
- Result packaging / figures for publication

**Do not commit:** `cluster.env`, generated `pIso_*`, `axes/`, or `minima_scale_*.json` (regenerate on the cluster).
