# Structure

`gensuper.py` builds a mixed-halide perovskite supercell geometry and topology
(bonds/angles/dihedrals). Force-field masses, charges, and pair/bond/angle/dihedral
coefficients are **not** embedded — insert your own before running LAMMPS.

```bash
# Optional: BR_FRACTION=0.5  (or run from a directory named like 50perbr)
python3 gensuper.py
```
