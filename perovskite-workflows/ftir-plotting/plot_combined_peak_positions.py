from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTAL_SCRIPT = HERE / "perovftir_plots.py"
# Optional overlay CSV. Set CALCULATED_PEAKS_CSV or place a file under data/.
CALCULATED_PEAKS = Path(
    os.environ["CALCULATED_PEAKS_CSV"]
    if "CALCULATED_PEAKS_CSV" in os.environ
    else HERE / "data" / "calculated_peak_positions.csv"
)


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("perovftir_plots", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.1,
            "lines.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _fmt_n_sigfigs(value: float, n: int) -> str:
    """Format to exactly *n* significant figures, keeping trailing zeros (e.g. ``13.0`` not ``13``).

    General ``g`` formatting drops insignificant zeros; this rounds then uses fixed decimals.
    """
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0." + "0" * n
    sign = "-" if value < 0 else ""
    x = abs(float(value))
    e = int(np.floor(np.log10(x)))
    ndec = max(0, (n - 1) - e)
    rounded = round(x, ndec)
    if rounded == 0.0:
        return sign + ("0." + "0" * n)
    e2 = int(np.floor(np.log10(rounded)))
    ndec2 = max(0, (n - 1) - e2)
    return sign + f"{rounded:.{ndec2}f}"


def fmt_slope_3sf(value: float) -> str:
    """Three significant figures for slope (fraction axis)."""
    return _fmt_n_sigfigs(value, 3)


def fmt_offset_4sf(value: float) -> str:
    """Four significant figures for offset / intercept (cm⁻¹)."""
    return _fmt_n_sigfigs(value, 4)


def format_forced_intercept_fit(x0: float, slope_wrt_fraction: float) -> str:
    return f"{fmt_offset_4sf(x0)} + {fmt_slope_3sf(slope_wrt_fraction)} x"


def format_linear_fit(slope_wrt_fraction: float, intercept: float) -> str:
    sign = "+" if intercept >= 0 else "-"
    return f"{fmt_slope_3sf(slope_wrt_fraction)} x {sign} {fmt_offset_4sf(abs(intercept))}"


def load_experimental_peaks():
    module = load_module(EXPERIMENTAL_SCRIPT)
    spectra = module.build_spectra()
    spectra = sorted(spectra, key=lambda s: s.br_pct)
    comps = np.array([s.br_pct for s in spectra], dtype=float)
    peak1 = np.array([s.peak1_x for s in spectra], dtype=float)
    peak2 = np.array([s.peak2_x for s in spectra], dtype=float)

    x0_p1 = float(peak1[0])
    x0_p2 = float(peak2[0])
    m1 = module.fit_slope_through_intercept(comps, peak1, x0_p1)
    m2 = module.fit_slope_through_intercept(comps, peak2, x0_p2)
    fit1 = x0_p1 + m1 * comps
    fit2 = x0_p2 + m2 * comps
    return comps, peak1, peak2, fit1, fit2, x0_p1, x0_p2, m1, m2


def load_calculated_peaks():
    data = np.loadtxt(CALCULATED_PEAKS, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    comps = data[:, 0]
    peaks = data[:, 1]
    coeffs = np.polyfit(comps, peaks, 1)
    fit = np.polyval(coeffs, comps)
    return comps, peaks, fit, float(coeffs[0]), float(coeffs[1])


def style_axes(ax):
    ax.tick_params(axis="both", direction="in", top=True, right=True, length=6, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)


def main() -> int:
    publication_style()

    exp_comp, exp_peak1, exp_peak2, exp_fit1, exp_fit2, x0_p1, x0_p2, m1, m2 = load_experimental_peaks()
    calc_comp, calc_peak, calc_fit, calc_slope, calc_intercept = load_calculated_peaks()

    # Composition as bromide fraction x in [0, 1]; fits were trained on percent.
    exp_x = exp_comp / 100.0
    calc_x = calc_comp / 100.0
    m1_frac = m1 * 100.0
    m2_frac = m2 * 100.0
    calc_slope_frac = calc_slope * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True, sharey=False)

    exp_color = "#c9252d"
    calc_color = "#2b2b2b"

    # peak1: lower wavenumber → low frequency symmetric; peak2 → high frequency asymmetric
    axes[0].scatter(exp_x, exp_peak1, s=42, color=exp_color, marker="o", label="Low frequency symmetric (exp.)")
    axes[0].scatter(exp_x, exp_peak2, s=48, color=exp_color, marker="x", label="High frequency asymmetric (exp.)")
    axes[0].plot(
        exp_x,
        exp_fit1,
        color=exp_color,
        linestyle="--",
        linewidth=1.2,
        label=f"Low frequency symmetric: {format_forced_intercept_fit(x0_p1, m1_frac)}",
    )
    axes[0].plot(
        exp_x,
        exp_fit2,
        color=exp_color,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"High frequency asymmetric: {format_forced_intercept_fit(x0_p2, m2_frac)}",
    )
    axes[0].set_title("Experimental Peak Positions")
    axes[0].set_xlabel("Bromide fraction, $x$")
    axes[0].set_ylabel("Peak position (cm$^{-1}$)")
    axes[0].legend(frameon=False, loc="best")

    axes[1].scatter(calc_x, calc_peak, s=46, color=calc_color, marker="o", label="Empirical Peak")
    axes[1].plot(
        calc_x,
        calc_fit,
        color=calc_color,
        linestyle="--",
        linewidth=1.3,
        label=f"Linear fit: {format_forced_intercept_fit(calc_intercept, calc_slope_frac)}",
    )
    axes[1].set_title("Calculated Peak Positions")
    axes[1].set_xlabel("Bromide fraction, $x$")
    axes[1].legend(frameon=False, loc="best")

    xmax = max(float(np.max(exp_comp)), float(np.max(calc_comp))) / 100.0
    xmin = min(float(np.min(exp_comp)), float(np.min(calc_comp))) / 100.0

    for ax in axes:
        ax.set_xlim(xmin - 0.03, xmax + 0.03)
        style_axes(ax)

    exp_ymin = min(float(np.min(exp_peak1)), float(np.min(exp_peak2))) - 4.0
    exp_ymax = max(float(np.max(exp_peak1)), float(np.max(exp_peak2))) + 4.0
    calc_ymin = float(np.min(calc_peak)) - 4.0
    calc_ymax = float(np.max(calc_peak)) + 4.0

    common_span = max(exp_ymax - exp_ymin, calc_ymax - calc_ymin)
    exp_center = 0.5 * (exp_ymin + exp_ymax)
    calc_center = 0.5 * (calc_ymin + calc_ymax)

    axes[0].set_ylim(exp_center - 0.5 * common_span, exp_center + 0.5 * common_span)
    axes[1].set_ylim(calc_center - 0.5 * common_span, calc_center + 0.5 * common_span)

    out_png = HERE / "combined_peak_positions_exp_vs_calculated.png"
    out_pdf = HERE / "combined_peak_positions_exp_vs_calculated.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
