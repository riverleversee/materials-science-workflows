import os

import numpy as np
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.environ.get("FTIR_DATA_DIR", os.path.join(HERE, "data")))

# Background CSV (not bundled). Override with FTIR_BACKGROUND_CSV.
BACKGROUND_CSV = os.environ.get(
    "FTIR_BACKGROUND_CSV", os.path.join(DATA_DIR, "background.CSV")
)

X_MIN = 1700.0
OUTPUT_PNG = os.path.join(HERE, "background_plot.png")
OUTPUT_PNG_RESIDUAL_1P1 = os.path.join(HERE, "background_scaled1p1_minus_bg.png")
OUTPUT_PNG_RESIDUAL_0P9 = os.path.join(HERE, "background_scaled0p9_minus_bg.png")
SCALE_UP = 1.1
SCALE_DOWN = 0.9
ASSUME_PERCENT_TRANSMISSION = True


def load_two_column_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2-column numeric CSV, got shape {getattr(data, 'shape', None)} for {path}")
    x = data[:, 0]
    y = data[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def to_absorbance(y: np.ndarray) -> np.ndarray:
    """
    Convert transmission to absorbance.

    If ASSUME_PERCENT_TRANSMISSION=True, interprets y as %T and uses:
        A = -log10(T/100)
    Otherwise interprets y as already-absorbance and returns y.
    """
    if not ASSUME_PERCENT_TRANSMISSION:
        return y

    t = y / 100.0
    # Avoid log10(<=0) from noise/offsets
    t = np.clip(t, 1e-6, 1.0)
    return -np.log10(t)


def main() -> None:
    x, y = load_two_column_csv(BACKGROUND_CSV)
    a = to_absorbance(y)
    residual_up = (SCALE_UP * a) - a
    residual_down = (SCALE_DOWN * a) - a

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
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, a, color="black", lw=1.2)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Background absorbance: bkg_n2purged_blankslide")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.minorticks_on()

    ax.set_xlim(X_MIN, float(np.max(x)))

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG)
    plt.close(fig)
    print(OUTPUT_PNG)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, residual_up, color="black", lw=1.2)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance residual (a.u.)")
    ax.set_title(f"Absorbance residual: ({SCALE_UP}×background) − background")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.minorticks_on()

    ax.set_xlim(X_MIN, float(np.max(x)))

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_RESIDUAL_1P1)
    plt.close(fig)
    print(OUTPUT_PNG_RESIDUAL_1P1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, residual_down, color="black", lw=1.2)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance residual (a.u.)")
    ax.set_title(f"Absorbance residual: ({SCALE_DOWN}×background) − background")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.minorticks_on()

    ax.set_xlim(X_MIN, float(np.max(x)))

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_RESIDUAL_0P9)
    plt.close(fig)
    print(OUTPUT_PNG_RESIDUAL_0P9)


if __name__ == "__main__":
    main()

