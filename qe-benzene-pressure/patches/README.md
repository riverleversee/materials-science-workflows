# Patched QE `cell_base.f90`

This is a **verbatim copy** of the modified Quantum ESPRESSO 7.0
`Modules/cell_base.f90` used with the benzene uniaxial workflows.

Compared with stock QE 7.0, this file adds extra `cell_dofree` `CASE`s
that control which cell matrix elements (`iforceh`) may relax, for example:

- `lockA`, `lockB`, `lockC`, `lockAC` — freeze one lattice direction while
  allowing the others (used for a/b-axis uniaxial jobs)
- `BCaxis` — relax B/C-related degrees of freedom (cb-axis jobs)
- `shapeAaxis`, `shapeBaxis`, `shapeBCaxis` — shape-constrained variants

See example inputs under `../examples/uniax/` (`lockA` / `lockB` / `BCaxis`).

Do not replace this with an unmodified upstream `cell_base.f90` if you need
those custom `cell_dofree` strings to work.
