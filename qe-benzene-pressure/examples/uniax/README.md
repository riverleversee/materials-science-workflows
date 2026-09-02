# Representative uniaxial example inputs (QE `pw.x` `vc-relax`).

Require the patched `Modules/cell_base.f90` under `../../patches/` for custom
`cell_dofree` values: `lockA` (a-axis), `lockB` (b-axis), `BCaxis` (cb-axis).

Typical settings: PBE + Grimme-D3, `ecutwfc=80`, `ecutrho=640`, `6×6×6`; `press` in kbar.
