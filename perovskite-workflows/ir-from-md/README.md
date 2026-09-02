# IR from MD

Canonical workflow: **`makeir.py`** — ACF of dipole-derivative vectors → IR spectrum  
(supports single-file and batch `random*/NNperbr/` averaging).

Plotting: **`ir_plot_results.py`**.

Dipole time series and spectra are **not** bundled; supply your own inputs.

```bash
# Single trajectory
python3 makeir.py --help

# Batch (average ACFs across random* for each composition folder)
python3 makeir.py --batch-root /path/to/batch --input-name total_dipole_derivative_vectors.txt

# Plot result CSVs
python3 ir_plot_results.py --results-dir /path/to/results
```
