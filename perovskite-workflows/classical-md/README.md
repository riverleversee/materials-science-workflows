# Classical MD (NPT sample)

Minimal Slurm + LAMMPS input for an isothermal–isobaric run at elevated pressure (~500 MPa as atm in metal units).

## Files

- `npt_500mpa_sample/lammpssubmit.sh` — Slurm wrapper
- `npt_500mpa_sample/perovin.in` — LAMMPS input

## Setup

1. Place a LAMMPS data file named `lammps_data_full_supercell.data` in the run directory (or edit `read_data` in `perovin.in`).
2. Set `LAMMPS_BIN` (and Slurm account) in `lammpssubmit.sh` for your cluster.
3. Submit: `sbatch lammpssubmit.sh`

Force-field coefficients must already be present in your data file (or added via LAMMPS commands you maintain separately).
