# Perovskite workflows

Scripts and templates for mixed-halide perovskite classical MD and IR analysis.

**Note: this tree has been heavily redacted.** It is a sanitized subset of internal workflows—scripts and templates only. Force-field parameters, production data, trajectories, experimental spectra, and related unpublished analysis are not included. Treat paths and inputs as placeholders you must supply locally.

| Folder | Contents |
|--------|----------|
| [`classical-md/`](classical-md/) | Minimal NPT submit example (user supplies LAMMPS data file) |
| [`structure/`](structure/) | Supercell geometry generator (force-field parameters supplied by user) |
| [`ir-from-md/`](ir-from-md/) | IR from MD via `makeir.py` (+ `ir_plot_results.py`) |
| [`ftir-plotting/`](ftir-plotting/) | FTIR plotting helpers (point at your own spectra directory) |

No trajectory or experimental data files are bundled.
