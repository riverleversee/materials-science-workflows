# QE benzene pressure workflows

Part of [`materials-science-workflows`](../).

Small showcase of Quantum ESPRESSO benzene hydrostatic / uniaxial / convergence work, copied verbatim from a CURC `/projects` dump (`QEme/BenzeneFinal`). Source trees were not modified.

## Layout

| Path | Role |
|------|------|
| `conver/` | ecut / ecutrho / mesh / smearing / optimize job scripts |
| `hydro/` | hydrostatic `vc-relax` submit scripts |
| `examples/uniax/` | representative uniaxial inputs (`lockA`, `lockB`, `BCaxis`) |
| `patches/cell_base.f90` | modified QE 7.0 `Modules/cell_base.f90` with custom `cell_dofree` cases |
| `makecopy.py` | helper from the original BenzeneFinal tree |

## Notes

- Typical settings in the example `.in` files: PBE + Grimme-D3, `ecutwfc=80`, `ecutrho=640`, `6×6×6`.
- Custom `cell_dofree` values used in uniax examples require the patched `cell_base.f90` (see `patches/`).
- Cluster absolute paths (e.g. `pseudo_dir`) are left as in the source copies.
