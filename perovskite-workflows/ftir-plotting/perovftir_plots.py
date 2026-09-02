import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SpectrumSpec:
    br_pct: float
    filename: str


@dataclass(frozen=True)
class SpectrumData:
    br_pct: float
    x: np.ndarray
    y_raw: np.ndarray
    y_norm: np.ndarray
    peak1_x: float
    peak2_x: float


HERE = os.path.dirname(os.path.abspath(__file__))
# Point FTIR_DATA_DIR at your local spectra folder (CSV files not bundled).
DATA_DIR = os.path.abspath(os.environ.get("FTIR_DATA_DIR", os.path.join(HERE, "data")))

# Example composition labels — rename filenames to match your local CSVs.
SPECTRA = [
    SpectrumSpec(0.0, "spectrum_0perbr.CSV"),
    SpectrumSpec(15.0, "spectrum_15perbr.CSV"),
    SpectrumSpec(30.0, "spectrum_30perbr.CSV"),
    SpectrumSpec(50.0, "spectrum_50perbr.CSV"),
    SpectrumSpec(70.0, "spectrum_70perbr.CSV"),
    SpectrumSpec(100.0, "spectrum_100perbr.CSV"),
]

# Two requested publication waterfall ranges
PUB_RANGE_1 = (3000.0, 3300.0)
PUB_RANGE_2 = (2700.0, 3400.0)

# Requested normalization factor:
# factor = local_max(3000-3250) - local_min(3150-3300)
NORM_MAX_WINDOW = (3000.0, 3250.0)
NORM_MIN_WINDOW = (3150.0, 3300.0)

# Peak tracking (interpolated guess + max search window)
PEAK1_GUESSES = (3125.0, 3140.0)  # (0% Br, 100% Br)
PEAK2_GUESSES = (3175.0, 3180.0)  # (0% Br, 100% Br)
PEAK_SEARCH_WIDTH = 30.0

# Waterfall styling
OFFSET_SPAN_0_TO_100 = 4.0  # vertical separation from 0% to 100% (composition-proportional)
LINEWIDTH = 1.35

# Label placement (far right, slightly above trace)
LABEL_DY = 0.06

# Extra outputs (previous workflow)
PLOT_FULL_RANGE = True
PLOT_START_TO_MAX_SERIES = True
START_TO_MAX_STARTS = list(range(2600, 1600, -100))  # 2600, 2500, ..., 1700

# Publication-style defaults
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "savefig.dpi": 300,
    }
)


def load_two_column_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2-column numeric CSV, got shape {getattr(data, 'shape', None)} for {path}")
    x = data[:, 0]
    y = data[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def slice_range(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]


def interp_guess(guesses: tuple[float, float], br_pct: float) -> float:
    g0, g100 = guesses
    return g0 + (g100 - g0) * (br_pct / 100.0)


def find_peak_near_guess(x: np.ndarray, y: np.ndarray, guess_pos: float, width: float) -> float:
    x0 = guess_pos - width
    x1 = guess_pos + width
    xw, yw = slice_range(x, y, x0, x1)
    if xw.size == 0:
        raise ValueError(f"No points found near guess={guess_pos} in [{x0}, {x1}]")
    i = int(np.argmax(yw))
    return float(xw[i])


def normalize_by_local_minmax(
    x: np.ndarray,
    y: np.ndarray,
    max_window: tuple[float, float],
    min_window: tuple[float, float],
) -> np.ndarray:
    _, y_in_max = slice_range(x, y, max_window[0], max_window[1])
    _, y_in_min = slice_range(x, y, min_window[0], min_window[1])
    if y_in_max.size == 0:
        raise ValueError(f"No points found in max_window={max_window}")
    if y_in_min.size == 0:
        raise ValueError(f"No points found in min_window={min_window}")

    y_max = float(np.max(y_in_max))
    y_min = float(np.min(y_in_min))
    factor = y_max - y_min
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"Bad normalization factor (y_max - y_min)={factor} from y_max={y_max}, y_min={y_min}")
    return (y - y_min) / factor


def y_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    return float(np.interp(x0, x, y))


def offset_for_br(br_pct: float) -> float:
    return (br_pct / 100.0) * OFFSET_SPAN_0_TO_100


def fit_slope_through_intercept(comps: np.ndarray, xs: np.ndarray, x0_at_0: float) -> float:
    """
    Forced-intercept linear regression:
        x(comp) = x0_at_0 + m * comp
    """
    dx = xs - x0_at_0
    denom = float(np.sum(comps**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(comps * dx) / denom)


def build_spectra() -> list[SpectrumData]:
    out: list[SpectrumData] = []
    for spec in sorted(SPECTRA, key=lambda s: s.br_pct):
        path = os.path.join(DATA_DIR, spec.filename)
        x, y = load_two_column_csv(path)
        y_norm = normalize_by_local_minmax(x, y, NORM_MAX_WINDOW, NORM_MIN_WINDOW)

        p1_guess = interp_guess(PEAK1_GUESSES, spec.br_pct)
        p2_guess = interp_guess(PEAK2_GUESSES, spec.br_pct)
        peak1_x = find_peak_near_guess(x, y, p1_guess, PEAK_SEARCH_WIDTH)
        peak2_x = find_peak_near_guess(x, y, p2_guess, PEAK_SEARCH_WIDTH)

        out.append(SpectrumData(spec.br_pct, x, y, y_norm, peak1_x, peak2_x))
    return out


def plot_pub_waterfall_with_guides(spectra: list[SpectrumData], x_min: float, x_max: float, outpath: str) -> None:
    spectra = sorted(spectra, key=lambda s: s.br_pct)
    comps = np.array([s.br_pct for s in spectra], dtype=float)
    p1 = np.array([s.peak1_x for s in spectra], dtype=float)
    p2 = np.array([s.peak2_x for s in spectra], dtype=float)

    # Forced-intercept fits anchored at 0% Br peak positions.
    x0_p1 = float(p1[0])
    x0_p2 = float(p2[0])
    m1 = fit_slope_through_intercept(comps, p1, x0_p1)
    m2 = fit_slope_through_intercept(comps, p2, x0_p2)
    x_fit1 = x0_p1 + m1 * comps
    x_fit2 = x0_p2 + m2 * comps

    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    # Plot traces (black) and labels (far right, slightly above).
    for spec in spectra:
        off = offset_for_br(spec.br_pct)
        x2, y2 = slice_range(spec.x, spec.y_norm, x_min, x_max)
        ax.plot(x2, y2 + off, color="black", lw=LINEWIDTH)

        x_text = x_max - 2.0
        y_text = y_at(spec.x, spec.y_norm, x_text) + off + LABEL_DY
        ax.text(
            x_text,
            y_text,
            f"{int(round(spec.br_pct))}% Br",
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="black",
        )

    # Dashed red guide lines. We force them to cross the peaks by evaluating y on each trace at the fitted x.
    y_fit1 = []
    y_fit2 = []
    for spec, xf1, xf2 in zip(spectra, x_fit1, x_fit2):
        off = offset_for_br(spec.br_pct)
        y_fit1.append(y_at(spec.x, spec.y_norm, float(xf1)) + off)
        y_fit2.append(y_at(spec.x, spec.y_norm, float(xf2)) + off)
    y_fit1 = np.array(y_fit1, dtype=float)
    y_fit2 = np.array(y_fit2, dtype=float)

    in1 = (x_fit1 >= x_min) & (x_fit1 <= x_max)
    in2 = (x_fit2 >= x_min) & (x_fit2 <= x_max)
    if np.any(in1):
        ax.plot(x_fit1[in1], y_fit1[in1], color="red", lw=1.0, ls="--", alpha=0.85)
    if np.any(in2):
        ax.plot(x_fit2[in2], y_fit2[in2], color="red", lw=1.0, ls="--", alpha=0.85)

    # Requested direction: low frequency -> high frequency
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity + offset (a.u.)")
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_peak_positions_fitted(spectra: list[SpectrumData], outpath: str) -> None:
    """
    Peak-position overview with fitted (forced-intercept) dashed lines.
    """
    spectra = sorted(spectra, key=lambda s: s.br_pct)
    comps = np.array([s.br_pct for s in spectra], dtype=float)
    p1 = np.array([s.peak1_x for s in spectra], dtype=float)
    p2 = np.array([s.peak2_x for s in spectra], dtype=float)

    x0_p1 = float(p1[0])
    x0_p2 = float(p2[0])
    m1 = fit_slope_through_intercept(comps, p1, x0_p1)
    m2 = fit_slope_through_intercept(comps, p2, x0_p2)
    fit1 = x0_p1 + m1 * comps
    fit2 = x0_p2 + m2 * comps

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(comps, p1, color="red", marker="o", s=30, label="Peak 1")
    ax.scatter(comps, p2, color="red", marker="x", s=40, label="Peak 2")
    ax.plot(comps, fit1, color="red", ls="--", lw=1.2, alpha=0.75)
    ax.plot(comps, fit2, color="red", ls="--", lw=1.2, alpha=0.75)
    ax.set_xlabel("% Br")
    ax.set_ylabel("Peak position (cm$^{-1}$)")
    ax.set_title("Fitted peak maxima vs composition")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_simple_waterfall(spectra: list[SpectrumData], x_min: float, x_max: float, outpath: str) -> None:
    """
    Simple normalized waterfall using the requested normalization but no guides/labels.
    (Kept for the earlier 'full range' and 'start->max' series outputs.)
    """
    spectra = sorted(spectra, key=lambda s: s.br_pct)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, spec in enumerate(spectra):
        x2, y2 = slice_range(spec.x, spec.y_norm, x_min, x_max)
        ax.plot(x2, y2 + i * 0.6, color="black", lw=1.0, alpha=0.9)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity + offset")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    spectra = build_spectra()

    # Publication-quality requested outputs
    out_pub1 = os.path.join(HERE, "waterfall_norm_pub_3000_3300.png")
    plot_pub_waterfall_with_guides(spectra, PUB_RANGE_1[0], PUB_RANGE_1[1], out_pub1)
    print(out_pub1)

    out_pub2 = os.path.join(HERE, "waterfall_norm_pub_2700_3400.png")
    plot_pub_waterfall_with_guides(spectra, PUB_RANGE_2[0], PUB_RANGE_2[1], out_pub2)
    print(out_pub2)

    # Fitted peak maxima plot (red points)
    out_peaks = os.path.join(HERE, "peaksposperov_fitted.png")
    plot_peak_positions_fitted(spectra, out_peaks)
    print(out_peaks)

    # Optional: older convenience outputs
    if PLOT_FULL_RANGE:
        x_min = float(min(np.min(s.x) for s in spectra))
        x_max = float(max(np.max(s.x) for s in spectra))
        out_full = os.path.join(HERE, "waterfall_6spectra_full.png")
        plot_simple_waterfall(spectra, x_min, x_max, out_full)
        print(out_full)

    if PLOT_START_TO_MAX_SERIES:
        x_max = float(max(np.max(s.x) for s in spectra))
        for start in START_TO_MAX_STARTS:
            outp = os.path.join(HERE, f"waterfall_6spectra_{start}_to_max.png")
            plot_simple_waterfall(spectra, float(start), x_max, outp)
            print(outp)


if __name__ == "__main__":
    main()

