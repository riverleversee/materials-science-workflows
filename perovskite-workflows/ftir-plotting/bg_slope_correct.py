import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SpectrumSpec:
    br_pct: float
    filename: str


HERE = os.path.dirname(os.path.abspath(__file__))
# Spectra and background live under FTIR_DATA_DIR (not bundled).
DATA_DIR = os.path.abspath(os.environ.get("FTIR_DATA_DIR", os.path.join(HERE, "data")))
BACKGROUND_CSV = os.environ.get(
    "FTIR_BACKGROUND_CSV", os.path.join(DATA_DIR, "background.CSV")
)

SPECTRA = [
    SpectrumSpec(0.0, "spectrum_0perbr.CSV"),
    SpectrumSpec(15.0, "spectrum_15perbr.CSV"),
    SpectrumSpec(30.0, "spectrum_30perbr.CSV"),
    SpectrumSpec(50.0, "spectrum_50perbr.CSV"),
    SpectrumSpec(70.0, "spectrum_70perbr.CSV"),
    SpectrumSpec(100.0, "spectrum_100perbr.CSV"),
]

# Window used to determine the slope (cm^-1)
SLOPE_WINDOW = (1990.0, 2000.0)

OUT_DIR = os.path.join(HERE, "bg_slope_corrected")
OUT_SUMMARY = os.path.join(OUT_DIR, "scales_summary.csv")

# Data model assumptions (per your note):
# - Sample spectra y-values are absorbance A (base-10):  A = -log10(T)
# - Background y-values are transmission (typically %T).
# We find a per-spectrum scale s applied to the background transmission Tb such that the
# corrected absorbance spectrum has near-zero slope in SLOPE_WINDOW.
ASSUME_BG_IS_PERCENT_TRANSMISSION = True
SEARCH_POINTS = 401  # coarse scan points for scale search
NEGATIVE_SCALE_LIMIT = 2.0  # allow adding back up to 2× background (scale down to -2)
NEGATIVE_SCALE_LIMIT_MAX = 50.0  # cap for adaptive expansion


def load_two_column_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2-column numeric CSV, got shape {getattr(data, 'shape', None)} for {path}")
    x = data[:, 0]
    y = data[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def slice_window(x: np.ndarray, y: np.ndarray, x0: float, x1: float) -> tuple[np.ndarray, np.ndarray]:
    m = (x >= x0) & (x <= x1)
    return x[m], y[m]


def slope_linear_fit(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        raise ValueError("Need at least 2 points to compute slope.")
    # y = m x + b
    m = float(np.polyfit(x, y, 1)[0])
    return m


def to_transmission_from_absorbance(a: np.ndarray) -> np.ndarray:
    # A = -log10(T)  ->  T = 10**(-A)
    return np.power(10.0, -a)


def bg_to_transmission(y_bg: np.ndarray) -> np.ndarray:
    if ASSUME_BG_IS_PERCENT_TRANSMISSION:
        return y_bg / 100.0
    return y_bg


def corrected_absorbance(
    x_s: np.ndarray,
    a_s: np.ndarray,
    t_bg_interp: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Apply scaled background subtraction in transmission space and convert back to absorbance:
        T_sample = 10**(-A_sample)
        T_corr = T_sample - scale * T_bg
        A_corr = -log10(T_corr)
    """
    t_s = to_transmission_from_absorbance(a_s)
    t_corr = t_s - scale * t_bg_interp
    # avoid invalid log
    t_corr = np.clip(t_corr, 1e-8, None)
    return -np.log10(t_corr)


def best_scale_for_zero_slope(
    x_s: np.ndarray,
    a_s: np.ndarray,
    t_bg_interp: np.ndarray,
    window: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Find scale that minimizes slope(A_corr) in the given window.
    Returns (scale_best, slope_before, slope_after).
    """
    w0, w1 = window
    xw, aw = slice_window(x_s, a_s, w0, w1)
    if xw.size < 2:
        raise ValueError("Insufficient sample points in slope window.")
    slope_before = slope_linear_fit(xw, aw)

    # Determine a safe upper bound for positive scale in the window so T_corr stays positive.
    # T_corr = T_s - s*T_bg > 0  ->  s < min(T_s/T_bg)
    tw_s, tw_bg = slice_window(x_s, to_transmission_from_absorbance(a_s), w0, w1)[1], slice_window(x_s, t_bg_interp, w0, w1)[1]
    if tw_bg.size < 2:
        raise ValueError("Insufficient background points in slope window.")
    # avoid divide by zero
    denom = np.clip(tw_bg, 1e-12, None)
    s_max = float(np.min(tw_s / denom)) * 0.999
    if not np.isfinite(s_max) or s_max <= 0:
        s_max = 0.0

    # Allow negative scale (meaning background was over-subtracted; adding some background back in T-space).
    if not np.isfinite(s_max):
        s_max = 0.0

    neg_limit = float(NEGATIVE_SCALE_LIMIT)
    best_s = 0.0
    best_m = None
    best_val = float("inf")

    while True:
        s_min = -neg_limit
        if s_max < s_min:
            s_max_scan = s_min
        else:
            s_max_scan = s_max

        scales = np.linspace(s_min, s_max_scan, SEARCH_POINTS)
        best_i = 0
        best_val_scan = float("inf")
        for i, s in enumerate(scales):
            a_corr = corrected_absorbance(x_s, a_s, t_bg_interp, float(s))
            xcw, acw = slice_window(x_s, a_corr, w0, w1)
            m = slope_linear_fit(xcw, acw)
            val = m * m
            if val < best_val_scan:
                best_val_scan = val
                best_i = i

        # Local refinement with a smaller bracket around best_i using a denser scan
        lo = max(0, best_i - 3)
        hi = min(scales.size - 1, best_i + 3)
        s_lo = float(scales[lo])
        s_hi = float(scales[hi])
        fine = np.linspace(s_lo, s_hi, 401)
        best_s_local = float(scales[best_i])
        best_val_local = float("inf")
        best_m_local = None
        for s in fine:
            a_corr = corrected_absorbance(x_s, a_s, t_bg_interp, float(s))
            xcw, acw = slice_window(x_s, a_corr, w0, w1)
            m = slope_linear_fit(xcw, acw)
            val = m * m
            if val < best_val_local:
                best_val_local = val
                best_s_local = float(s)
                best_m_local = float(m)

        # Track best across expansions
        if best_val_local < best_val:
            best_val = best_val_local
            best_s = best_s_local
            best_m = best_m_local

        # If optimum is not on the negative boundary, or we hit cap, stop expanding.
        if best_s_local > s_min + 1e-12:
            break
        if neg_limit >= float(NEGATIVE_SCALE_LIMIT_MAX):
            break
        neg_limit = min(float(NEGATIVE_SCALE_LIMIT_MAX), neg_limit * 2.0)

    slope_after = float(best_m if best_m is not None else slope_before)
    return best_s, float(slope_before), slope_after


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_dir(OUT_DIR)

    x_bg, y_bg = load_two_column_csv(BACKGROUND_CSV)
    t_bg = bg_to_transmission(y_bg)

    w0, w1 = SLOPE_WINDOW
    xb_win, tb_win = slice_window(x_bg, t_bg, w0, w1)
    if xb_win.size < 2 or tb_win.size < 2:
        raise ValueError(f"Background has insufficient points in window {SLOPE_WINDOW}")

    # We will always compute background slope on the same x grid as each sample (via interpolation),
    # so the only thing we need here is the raw background for interpolation.

    rows = ["br_pct,filename,scale_T,abs_slope_before,abs_slope_after"]

    for spec in SPECTRA:
        in_path = os.path.join(DATA_DIR, spec.filename)
        x_s, a_s = load_two_column_csv(in_path)

        # Interpolate background onto sample x grid
        t_bg_i = np.interp(x_s, x_bg, t_bg)

        scale, slope_before, slope_after = best_scale_for_zero_slope(
            x_s=x_s,
            a_s=a_s,
            t_bg_interp=t_bg_i,
            window=SLOPE_WINDOW,
        )

        a_corr = corrected_absorbance(x_s, a_s, t_bg_i, scale)

        # Save corrected CSV
        base, _ = os.path.splitext(spec.filename)
        out_csv = os.path.join(OUT_DIR, f"{base}_bgSlopeCorr_abs.csv")
        np.savetxt(out_csv, np.column_stack([x_s, a_corr]), delimiter=",", fmt="%.10g")

        # QC plot around the window
        qc_png = os.path.join(OUT_DIR, f"{base}_qc_{int(w0)}_{int(w1)}.png")
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(x_s, a_s, color="black", lw=1.0, alpha=0.30, label="original A (full)")
        ax.plot(x_s, a_corr, color="black", lw=1.2, alpha=0.85, label="corrected A (full)")
        xw, aw = slice_window(x_s, a_s, w0, w1)
        xw_c, acw = slice_window(x_s, a_corr, w0, w1)
        ax.plot(xw, aw, color="tab:blue", lw=2.0, label=f"orig slope={slope_before:.3g}")
        ax.plot(xw_c, acw, color="tab:green", lw=2.0, label=f"corr slope={slope_after:.3g}")
        ax.set_xlim(w0 - 30, w1 + 30)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Absorbance (a.u.)")
        ax.set_title(f"{int(round(spec.br_pct))}% Br | scale_T={scale:.4g}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(qc_png, dpi=250)
        plt.close(fig)

        rows.append(f"{spec.br_pct},{spec.filename},{scale:.10g},{slope_before:.10g},{slope_after:.10g}")

        print(
            f"{spec.br_pct:>6.1f}% Br  scale_T={scale: .6g}  slope(orig)={slope_before: .3g}  slope(corr)={slope_after: .3g}"
        )
        print(f"  wrote: {out_csv}")
        print(f"  wrote: {qc_png}")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()

