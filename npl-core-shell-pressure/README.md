# NPL core/shell pressure (CP2K)

Part of [`materials-science-workflows`](../).

CP2K workflows for CdSe/CdZnS nanoplatelets under pressure, copied **verbatim** from CURC `QEme/NPL/FullPlatelet/CP2K`. Source trees were not modified. **No Quantum ESPRESSO** content is included here.

**Paper:** Optical Properties of CdSe/CdZnS Core/Shell Nanoplatelets at High Pressure  
**DOI:** https://doi.org/10.1021/jacs.5c14939

## Layout (production-oriented)

| Path | Role |
|------|------|
| `extraCdSlayershell4mcore4cmlshell/` | 4 ML core + CdZnS shell (`FreeZhighcost`, `ligandopt`, `volmultiscale`) |
| `extraCdSlayershell6mcore4cmlshell/` | 6 ML core + CdZnS shell |
| `4ml/`, `6ml/` | ZnS-only shell variants (`ligandopt`, `volmultiscale`) |
| `coreonly/` | Unshelled core (`ligandopt`, `volscalemultinew`) |
| `BULK/ZincBlende/` | Bulk CdSe/CdS/ZnS EOS + `bgap` band structures (`bands.bs`) for effective-mass analysis |
| `BASISPOT/` | Shared TZV2P / GTH basis and potential files |

## Methods (as in the paper / these inputs)

- Code: CP2K QS/GPW
- Functional / basis: PBE / TZV2P-MOLOPT-PBE-GTH
- Cutoff: 2400 Ry (rel 240)
- k-points: NPL 12×12×2; bulk ZB 12×12×12
- DFT ligands: acetate (experiment uses PVP)
- Compression: linear scale ≤2% with ligand handling + core/shell z-scale (`JustEng*`)

## Staging notes

- Run outputs (`*.out`, restarts, trajectory POS-xyz) were omitted at copy time.
- Git-only trim: removed `*.Hessian` and `*.Log` (optimizer/restart junk) from this copy.
- Older / non-production trees (`*old*`, `GasPressure`, `QEeffmass`, `BULK/Wurtz`) were not copied.
- Cluster source under `E:\River\Projects` was not modified.
- A few representative scripts have short header comments in this git copy only.
