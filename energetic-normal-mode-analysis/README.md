# Energetic Normal Mode Analysis

Part of [`materials-science-workflows`](../) (this monorepo is the working home).

Python tooling to quantify **normal-mode displacement patterns** on compressed energetic crystals (BNFF / DNTF family), including functional-group decomposition under hydrostatic pressure.

## Workflow overview

```
MD / CP2K vibrational trajectories
        │
        ▼
  anime_{mode}.xyz  (multi-frame, per pressure)
  optimized_cell{P}GPa.cell
        │
        ▼
  run_group_analysis.py  ──►  JSON metrics per mode × pressure
        │
        ▼
  make_plots() / plottingmodes.py  ──►  publication PNG figures
```

### What each stage computes

1. **Supercell build** — wrap fractional coordinates, replicate 3×3×3, assign molecular connectivity (bond cutoff).
2. **Group tagging** — nitro (`NO₂`), furazan, furoxano (ring + exocyclic O) via connectivity rules.
3. **Mode metrics** — radial vs COM projection, intermolecular coupling (distance-threshold and r⁻² scaled), spread, axis projections, nearest-neighbor distances.
4. **Pressure sweep** — repeat at 0 / 4 / 10 GPa (configurable), compare mode localization vs pressure.

## Quick start

```bash
cd energetic-normal-mode-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Unit tests (no trajectory data needed)
python -m unittest discover -s tests -v
```

To **rerun** the pipeline, copy a local study tree with `anime_{mode}.xyz` trajectories and set `NMA_DATA_DIR` (see below). Precomputed figures are in [`results/minpress/`](results/minpress/).

## Data layout (local only — not in git)

Point `NMA_DATA_DIR` (or `--data-dir`) at a study folder on your machine:

```
minpress/
  optimized_cell0GPa.cell
  optimized_cell4GPa.cell
  optimized_cell10GPa.cell
  0GPa/AnimationFiles/anime_120.xyz
  4GPa/AnimationFiles/anime_120.xyz
  10GPa/AnimationFiles/anime_120.xyz
  ...
```

`Ambient_DNTF_UnitCell.xyz` is a sample ambient unit cell for I/O tests only.

## Results in git

| Path | Contents |
|------|----------|
| [`results/minpress/`](results/minpress/) | Min-pressure cohort PNG figures (~3 MB) |

No `minamb` figures or mode trajectories are included.

## Script map

| Path | Role |
|------|------|
| `run_group_analysis.py` | **CLI entry point** — analysis + optional plots |
| `mode_analysis/` | Shared I/O and path helpers |
| `BNFFanalysis/minpress/group_mode_analysis.py` | **Canonical driver** (metrics + `make_plots`) |
| `BNFFanalysis/minpress/plottingmodes.py` | Additional figure helpers from saved JSON |
| `BNFFanalysis/minpress/test_group_overlap.py` | Connectivity / group-overlap sanity check |
| `tests/` | Unit tests (no trajectory data required) |
| `_history/` | Archived earlier script versions (not maintained) |

## CLI options

```
python run_group_analysis.py --help

  --data-dir PATH          Study root (default: NMA_DATA_DIR or BNFFanalysis/minpress/)
  --pressures 0,4,10       GPa values to compare
  --modes 120,166          Mode indices (default: common anime_*.xyz at all pressures)
  --analysis-only          Skip plot generation
  --plots-only RESULTS.json  Regenerate figures from saved JSON
```

## Requirements

- Python 3.10+
- `numpy`, `matplotlib` (see `requirements.txt`)

## Not in git

- Mode animation trajectories (`anime_*.xyz`) — keep on private/cluster storage
- `minamb/` study and figures
- Analysis JSON scratch files (`*_results.json`)

## Provenance

Staged from `D:\CursorCoding\MetricAnalysis\` (2026-09). Original paths are unchanged on the D: drive.
