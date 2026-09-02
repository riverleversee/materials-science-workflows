#!/usr/bin/env python3
"""
Generate an IR spectrum from dipole-derivative vectors time series.

Input format (as in your file):
  # comments...
  TimeStep  c_dMuDxTotal  c_dMuDyTotal  c_dMuDzTotal
  0  162.031  -1191.28  247.034
  1  ...

Method (tail-noise controlled):
  - Compute the vector autocorrelation function (ACF):
      C(t) = <v(0) · v(t)>, v = (dx, dy, dz) with mean removed per component
  - Truncate the ACF at a max lag (fs) so the noisy tail does not feed the spectrum
  - Apply a taper-to-zero window (Hann) on the truncated ACF to reduce truncation ringing
  - rFFT the truncated, tapered ACF to obtain a spectrum (arbitrary units)
  - Convert frequency (Hz) to wavenumber (cm^-1)

Outputs:
  - <prefix>.csv  (comma-separated)
  - <prefix>.txt  (comma-separated; CSV-style in a .txt as requested)
  - <prefix>.png  (if matplotlib available, unless --no-plot)

Batch (multi-trajectory) mode:
  If you have multiple trajectories laid out like:

    <batch-root>/
      random1/10perbr/<input-name>
      random1/20perbr/<input-name>
      random2/10perbr/<input-name>
      ...

  then run with:

    makeir.py --batch-root <batch-root> --input-name <your_filename>

  The script will compute one ACF per trajectory, average the ACFs across
  random* for each percent-Br folder, then compute one IR spectrum per percent.
  Results are written to <batch-root>/results/ by default.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable
import math

import numpy as np


def read_vectors(path: Path) -> np.ndarray:
    """
    Read vectors from the input file.
    Returns ndarray shape (N, 3): columns are dx, dy, dz.
    Skips comment lines and malformed trailing/incomplete lines.
    """
    data: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                data.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue

    if not data:
        raise ValueError(f"No data rows found in {path}")

    return np.asarray(data, dtype=np.float64)


def count_vector_rows(path: Path) -> int:
    """
    Fast row count for trajectory length without parsing floats.
    Counts non-comment lines that have at least 4 whitespace-separated fields.
    """
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if len(s.split()) >= 4:
                n += 1
    return n


def filter_trajectory_paths(
    paths: Iterable[Path],
    dt_fs: float,
    acf_max_lag_fs: float,
) -> tuple[list[tuple[Path, int]], int, int, int]:
    """
    Filter trajectories for a single %perbr group.

    Rules:
    - Must be at least half the length of the longest trajectory in the group.
    - Must be long enough to support the desired ACF cutoff length.

    Returns:
      (usable_paths_with_lengths, required_pts, max_len, half_len_threshold)
    """
    paths = list(paths)
    if not paths:
        return ([], 0, 0, 0)

    required_pts = int(round(acf_max_lag_fs / dt_fs))
    required_pts = max(16, required_pts)

    lengths = {p: count_vector_rows(p) for p in paths}
    max_len = max(lengths.values()) if lengths else 0
    half_len_threshold = int(math.ceil(0.5 * max_len))

    usable: list[tuple[Path, int]] = []
    for p in paths:
        n_samples = int(lengths.get(p, 0))
        if n_samples < half_len_threshold:
            print(
                f"WARNING: skipping trajectory (too short vs longest): {p} "
                f"(N={n_samples}; need >= {half_len_threshold} which is half of max N={max_len})",
                file=sys.stderr,
            )
            continue
        if n_samples < required_pts:
            print(
                f"WARNING: skipping trajectory (too short for ACF cutoff): {p} "
                f"(N={n_samples}; need >= {required_pts} for "
                f"--acf-max-lag-fs {acf_max_lag_fs} at --dt-fs {dt_fs})",
                file=sys.stderr,
            )
            continue
        usable.append((p, n_samples))

    return usable, required_pts, max_len, half_len_threshold


def autocorr_unbiased_fft(x: np.ndarray) -> np.ndarray:
    """
    Unbiased autocorrelation via FFT for a 1D signal.
    Returns C[0..N-1] where C[k] = <x[t] x[t+k]> with mean removed.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    nfft = 1 << (2 * n - 1).bit_length()  # >= 2N, pow2 for speed
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    ac = ac / np.arange(n, 0, -1, dtype=np.float64)  # unbiased
    return ac


def hanning_taper_acf(n: int) -> np.ndarray:
    """
    Traditional one-sided Hanning/Hann taper for an ACF defined on [0, T].

    We want w(0)=1 (do NOT damp C(0)) and w(T)=0 to smoothly bring the truncated
    tail to zero:

      w[k] = 0.5 * (1 + cos(pi * k / (n-1)))   for k=0..n-1

    This is a "raised cosine" taper (half Hann). It avoids the mistake of using
    the symmetric Hann window (0 at both ends), which would force C(0)=0.
    """
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 + np.cos(np.pi * k / (n - 1)))


def vector_acf(vecs: np.ndarray) -> np.ndarray:
    """
    Vector ACF C(t) = <v(0)·v(t)> for v=(x,y,z), with mean removed per component.

    Returns C[0..N-1] normalized by C(0) (if C(0) != 0), matching the previous
    methodology that fed a normalized ACF into the truncation+taper+FFT stages.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.ndim != 2 or vecs.shape[1] != 3:
        raise ValueError("vecs must have shape (N, 3)")

    c = autocorr_unbiased_fft(vecs[:, 0]) + autocorr_unbiased_fft(vecs[:, 1]) + autocorr_unbiased_fft(vecs[:, 2])
    if c.shape[0] > 0 and c[0] != 0:
        c = c / c[0]
    return c


def truncate_acf(c: np.ndarray, dt_fs: float, acf_max_lag_fs: float) -> np.ndarray:
    if dt_fs <= 0:
        raise ValueError("--dt-fs must be > 0")
    if acf_max_lag_fs <= 0:
        raise ValueError("--acf-max-lag-fs must be > 0")

    c = np.asarray(c, dtype=np.float64)
    max_pts = int(round(acf_max_lag_fs / dt_fs))
    max_pts = max(16, min(max_pts, c.shape[0]))
    return c[:max_pts]


def acf_to_spectrum_from_acf(
    c: np.ndarray,
    dt_fs: float,
    taper: str,
    temperature_k: float | None,
    max_cm1: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spectrum from an already-prepared (optionally averaged) ACF.
    Methodology is identical to the original ACF->spectrum portion.
    """
    c = np.asarray(c, dtype=np.float64)

    # Taper
    taper = taper.lower().strip()
    if taper == "hann":
        w = hanning_taper_acf(c.shape[0])
    elif taper == "none":
        w = np.ones_like(c)
    else:
        raise ValueError("--taper must be 'hann' or 'none'")

    c_win = c * w

    # FFT
    dt_s = dt_fs * 1e-15
    freqs_hz = np.fft.rfftfreq(c_win.shape[0], d=dt_s)
    spec = np.real(np.fft.rfft(c_win))
    spec = np.maximum(spec, 0.0)
    if spec.shape[0] > 0:
        spec[0] = 0.0

    # Quantum correction (optional)
    if temperature_k is not None:
        h = 6.62607015e-34
        kB = 1.380649e-23
        x = (h * freqs_hz) / (kB * temperature_k)
        factor = 1.0 - np.exp(-x, dtype=np.float64)
        factor[0] = 1.0
        spec = spec * factor

    # Hz -> cm^-1
    c_cm_s = 2.99792458e10
    wn_cm1 = freqs_hz / c_cm_s

    if max_cm1 is not None:
        mask = wn_cm1 <= max_cm1
        wn_cm1 = wn_cm1[mask]
        spec = spec[mask]

    return wn_cm1, spec


def acf_to_spectrum(
    vecs: np.ndarray,
    dt_fs: float,
    acf_max_lag_fs: float,
    taper: str,
    temperature_k: float | None,
    max_cm1: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spectrum from truncated+tapered vector ACF.
    """
    c = vector_acf(vecs)
    c = truncate_acf(c, dt_fs=dt_fs, acf_max_lag_fs=acf_max_lag_fs)
    return acf_to_spectrum_from_acf(c, dt_fs=dt_fs, taper=taper, temperature_k=temperature_k, max_cm1=max_cm1)


def write_csv_like(path: Path, wn_cm1: np.ndarray, intensity: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("wavenumber_cm^-1,intensity_arb\n")
        for w, i in zip(wn_cm1, intensity):
            f.write(f"{w:.6f},{i:.10e}\n")


def maybe_plot(path: Path, wn_cm1: np.ndarray, intensity: np.ndarray, title: str = "IR spectrum (ACF truncation)") -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return

    plt.figure(figsize=(11, 5))
    plt.plot(wn_cm1, intensity, linewidth=1.0)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb.)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def find_local_maxima(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return indices of simple 1D local maxima: y[i-1] < y[i] >= y[i+1].
    (No SciPy dependency; intended for quick peak picking.)
    """
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] < 3:
        return np.array([], dtype=np.int64)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def write_peak_report(
    path: Path,
    wn_cm1: np.ndarray,
    intensity: np.ndarray,
    lo_cm1: float,
    hi_cm1: float,
) -> None:
    wn_cm1 = np.asarray(wn_cm1, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if wn_cm1.shape != intensity.shape:
        raise ValueError("wn_cm1 and intensity must have the same shape")

    # Restrict to desired window
    mask = (wn_cm1 >= lo_cm1) & (wn_cm1 <= hi_cm1)
    wn_w = wn_cm1[mask]
    it_w = intensity[mask]

    peaks_local = find_local_maxima(wn_w, it_w)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Peaks (local maxima) in {lo_cm1:.1f}–{hi_cm1:.1f} cm^-1\n")
        f.write("# wavenumber_cm^-1  intensity_arb\n")

        if wn_w.size == 0:
            f.write("# No data in the requested range.\n")
            return

        if peaks_local.size == 0:
            # Still report the absolute maximum in the window for convenience
            imax = int(np.argmax(it_w))
            f.write("# No strict local maxima found; reporting window maximum.\n")
            f.write(f"{wn_w[imax]:.6f}  {it_w[imax]:.10e}\n")
            return

        # Sort peaks by descending intensity (most relevant first)
        peaks_sorted = peaks_local[np.argsort(it_w[peaks_local])[::-1]]
        for idx in peaks_sorted:
            f.write(f"{wn_w[idx]:.6f}  {it_w[idx]:.10e}\n")


def _extract_percent_br(name: str) -> tuple[float, str] | None:
    """
    Parse a directory name like '10perbr' or '12.5perbr' (case-insensitive).
    Returns (value_as_float, normalized_label) or None if not matched.
    """
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*perbr\s*", name, flags=re.IGNORECASE)
    if not m:
        return None
    v = float(m.group(1))
    raw = m.group(1)
    label = raw.replace(".", "p") + "perbr"
    return v, label


def discover_trajectories(
    root: Path,
    random_glob: str,
    input_name: str,
) -> dict[str, list[Path]]:
    """
    Discover trajectory files in:
      root/random*/<percent>perbr/<input_name>

    Returns dict: percent_label -> list of file Paths across random folders.
    """
    root = root.resolve()
    out: dict[str, list[Path]] = {}

    for rnd in sorted(root.glob(random_glob)):
        if not rnd.is_dir():
            continue
        for per_dir in sorted(rnd.iterdir()):
            if not per_dir.is_dir():
                continue
            parsed = _extract_percent_br(per_dir.name)
            if parsed is None:
                continue
            _v, label = parsed
            p = per_dir / input_name
            if p.is_file():
                out.setdefault(label, []).append(p)
    return out


def average_acf_over_trajectories(
    paths: Iterable[Path],
    dt_fs: float,
    acf_max_lag_fs: float,
) -> tuple[np.ndarray, int]:
    """
    Compute one normalized vector ACF per trajectory, truncate to common length,
    then compute a length-weighted average across trajectories so shorter
    trajectories contribute less.

    Returns (acf_avg, n_traj_used).
    """
    acfs: list[np.ndarray] = []
    weights: list[float] = []
    n_used = 0
    usable, required_pts, max_len, half_len_threshold = filter_trajectory_paths(
        paths,
        dt_fs=dt_fs,
        acf_max_lag_fs=acf_max_lag_fs,
    )
    for p, n_samples in usable:
        vecs = read_vectors(p)
        # If parsing skipped malformed lines, update weight to actual samples used,
        # and re-check cutoff requirement.
        n_samples_parsed = int(vecs.shape[0])
        if n_samples_parsed != n_samples:
            n_samples = n_samples_parsed
            if n_samples < required_pts:
                print(
                    f"WARNING: skipping trajectory (too short after parsing): {p} "
                    f"(N={n_samples}; need >= {required_pts})",
                    file=sys.stderr,
                )
                continue
        c = vector_acf(vecs)
        if c.shape[0] < required_pts:
            # Extremely defensive: should not happen if N check above passes.
            print(
                f"WARNING: skipping trajectory (ACF shorter than expected): {p} "
                f"(ACF points={c.shape[0]}; need >= {required_pts})",
                file=sys.stderr,
            )
            continue
        c = c[:required_pts]
        acfs.append(c)
        # Weight by trajectory length (number of samples) so short trajectories contribute less.
        weights.append(float(n_samples))
        n_used += 1

    if not acfs:
        raise ValueError(
            "No usable trajectories found for averaging after filtering.\n"
            f"Required for ACF cutoff: N >= {required_pts}\n"
            f"Required vs longest:    N >= {half_len_threshold} (half of max N={max_len})"
        )

    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("Invalid trajectory weights encountered while averaging ACF.")
    c_stack = np.stack(acfs, axis=0)
    c_avg = np.average(c_stack, axis=0, weights=w)
    return c_avg, n_used


def plot_overlay(
    out_png: Path,
    spectra: list[tuple[str, np.ndarray, np.ndarray]],
    lo_cm1: float,
    hi_cm1: float,
) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return

    plt.figure(figsize=(11, 5))
    for label, wn, it in spectra:
        mask = (wn >= lo_cm1) & (wn <= hi_cm1)
        plt.plot(wn[mask], it[mask], linewidth=1.2, label=label)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb.)")
    plt.title(f"IR spectra overlay ({lo_cm1:.0f}–{hi_cm1:.0f} cm$^{{-1}}$)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncols=2, fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)


def plot_spectrum_range(
    out_png: Path,
    wn_cm1: np.ndarray,
    intensity: np.ndarray,
    lo_cm1: float,
    hi_cm1: float,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return

    wn_cm1 = np.asarray(wn_cm1, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    mask = (wn_cm1 >= lo_cm1) & (wn_cm1 <= hi_cm1)

    plt.figure(figsize=(11, 5))
    plt.plot(wn_cm1[mask], intensity[mask], linewidth=1.2)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb.)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)


def average_spectra_over_trajectories(
    usable: list[tuple[Path, int]],
    dt_fs: float,
    acf_max_lag_fs: float,
    taper: str,
    temperature_k: float | None,
    max_cm1: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute one spectrum per trajectory and then take a length-weighted average
    over spectra.

    Note: this is intentionally different from averaging ACFs prior to FFT;
    it is provided for comparison only (same discard rules are expected to be
    applied upstream).
    """
    wn_common: np.ndarray | None = None
    specs: list[np.ndarray] = []
    weights: list[float] = []
    n_used = 0

    for p, n_counted in usable:
        vecs = read_vectors(p)
        n_samples = int(vecs.shape[0]) or int(n_counted)
        wn, it = acf_to_spectrum(
            vecs=vecs,
            dt_fs=dt_fs,
            acf_max_lag_fs=acf_max_lag_fs,
            taper=taper,
            temperature_k=temperature_k,
            max_cm1=max_cm1,
        )
        if wn_common is None:
            wn_common = wn
        else:
            if wn.shape != wn_common.shape or not np.allclose(wn, wn_common, rtol=0.0, atol=1e-9):
                raise ValueError(f"Inconsistent wavenumber grid across trajectories; cannot average spectra. Offender: {p}")
        specs.append(it)
        weights.append(float(n_samples))
        n_used += 1

    if wn_common is None or not specs:
        raise ValueError("No spectra available to average.")

    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("Invalid trajectory weights encountered while averaging spectra.")
    spec_avg = np.average(np.stack(specs, axis=0), axis=0, weights=w)
    return wn_common, spec_avg, n_used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help=(
            "Batch root directory to scan. If omitted, defaults to the current working directory. "
            "The script always runs in batch mode: scan <root>/random*/<percent>perbr/<input-name>, "
            "average ACF over random* per percent, then compute one spectrum per percent."
        ),
    )
    ap.add_argument(
        "--random-glob",
        default="random*",
        help="Glob under --batch-root to identify random folders (default: random*).",
    )
    ap.add_argument(
        "--input-name",
        default="total_dipole_derivative_vectors.txt",
        help="Trajectory file name inside each <percent>perbr folder (default keeps old single-trajectory name).",
    )
    ap.add_argument(
        "--results-dirname",
        default="results",
        help="Folder name (under --batch-root) to write per-percent spectra and overlay plot.",
    )
    ap.add_argument(
        "--dt-fs",
        type=float,
        default=0.2,
        help="Time step between rows in femtoseconds (fs).",
    )
    ap.add_argument(
        "--acf-max-lag-fs",
        type=float,
        default=20000.0,
        help="Truncate ACF at this lag (fs). 20000 fs = 20 ps.",
    )
    ap.add_argument(
        "--taper",
        choices=("hann", "none"),
        default="hann",
        help="One-sided Hanning taper applied to truncated ACF (keeps C(0)=1, goes to 0 at cutoff).",
    )
    ap.add_argument(
        "--temperature-k",
        type=float,
        default=None,
        help="If set, apply a simple quantum correction factor.",
    )
    ap.add_argument(
        "--max-cm1",
        type=float,
        default=4000.0,
        help="Max wavenumber (cm^-1) to output/plot.",
    )
    ap.add_argument(
        "--output-prefix",
        default="ir_spectrum",
        help="(Ignored in batch mode) Kept for backward compatibility.",
    )
    ap.add_argument(
        "--peak-range-cm1",
        type=float,
        nargs=2,
        default=(2700.0, 3500.0),
        metavar=("LO", "HI"),
        help="Find and report peak locations within this wavenumber range (cm^-1).",
    )
    ap.add_argument("--no-plot", action="store_true", help="Do not write PNG.")
    ap.add_argument(
        "--overlay-range-cm1",
        type=float,
        nargs=2,
        default=(2900.0, 3200.0),
        metavar=("LO", "HI"),
        help="Overlay plot range (cm^-1) for batch mode.",
    )
    args = ap.parse_args()

    # Always batch mode.
    batch_root: Path = (Path(args.batch_root) if args.batch_root is not None else Path.cwd()).resolve()
    trajs_by_percent = discover_trajectories(
        root=batch_root,
        random_glob=args.random_glob,
        input_name=args.input_name,
    )
    if not trajs_by_percent:
        raise SystemExit(
            f"No trajectories found under {batch_root} with pattern "
            f"{args.random_glob}/<percent>perbr/{args.input_name}"
        )

    # Sort by numeric percent if possible (fallback: label sort)
    def _sort_key(label: str) -> tuple[int, float, str]:
        parsed = _extract_percent_br(label)
        if parsed is None:
            return (1, 0.0, label)
        v, _ = parsed
        return (0, v, label)

    results_dir = batch_root / args.results_dirname
    spectra_for_overlay: list[tuple[str, np.ndarray, np.ndarray]] = []
    spectra_for_overlay_specavg: list[tuple[str, np.ndarray, np.ndarray]] = []

    for label in sorted(trajs_by_percent.keys(), key=_sort_key):
        paths = trajs_by_percent[label]
        usable, required_pts, max_len, half_len_threshold = filter_trajectory_paths(
            paths,
            dt_fs=args.dt_fs,
            acf_max_lag_fs=args.acf_max_lag_fs,
        )
        usable_paths = [p for p, _n in usable]
        if not usable_paths:
            print(
                f"WARNING: {label}: no usable trajectories after filtering "
                f"(need N >= {required_pts} for cutoff; and N >= {half_len_threshold} which is half of max N={max_len}).",
                file=sys.stderr,
            )
            continue

        c_avg, ntraj = average_acf_over_trajectories(usable_paths, dt_fs=args.dt_fs, acf_max_lag_fs=args.acf_max_lag_fs)
        wn_cm1, intensity = acf_to_spectrum_from_acf(
            c_avg,
            dt_fs=args.dt_fs,
            taper=args.taper,
            temperature_k=args.temperature_k,
            max_cm1=args.max_cm1,
        )
        spectra_for_overlay.append((label, wn_cm1, intensity))

        # Also compute (for comparison) a length-weighted average of the spectra themselves.
        wn_specavg, intensity_specavg, ntraj_specavg = average_spectra_over_trajectories(
            usable,
            dt_fs=args.dt_fs,
            acf_max_lag_fs=args.acf_max_lag_fs,
            taper=args.taper,
            temperature_k=args.temperature_k,
            max_cm1=args.max_cm1,
        )
        spectra_for_overlay_specavg.append((label, wn_specavg, intensity_specavg))

        # Per-trajectory spectra plots for comparison
        if not args.no_plot:
            lo_plot, hi_plot = float(args.overlay_range_cm1[0]), float(args.overlay_range_cm1[1])
            traj_plot_dir = results_dir / "trajectory_spectra" / label
            for p in usable_paths:
                # Expect layout: <batch-root>/random#/label/<input-name>
                rnd = p.parent.parent.name
                if not re.fullmatch(r"random\d+", rnd, flags=re.IGNORECASE):
                    rnd = p.parent.parent.name  # fallback to whatever is there

                vecs = read_vectors(p)
                wn_t, it_t = acf_to_spectrum(
                    vecs=vecs,
                    dt_fs=args.dt_fs,
                    acf_max_lag_fs=args.acf_max_lag_fs,
                    taper=args.taper,
                    temperature_k=args.temperature_k,
                    max_cm1=args.max_cm1,
                )
                out_traj_png = traj_plot_dir / f"ir_{label}_{rnd}.png"
                plot_spectrum_range(
                    out_traj_png,
                    wn_t,
                    it_t,
                    lo_cm1=lo_plot,
                    hi_cm1=hi_plot,
                    title=f"IR spectrum ({label}, {rnd})",
                )

        # Outputs per percent Br
        prefix = results_dir / f"ir_{label}"
        out_csv = prefix.with_suffix(".csv")
        out_txt = prefix.with_suffix(".txt")
        out_png = prefix.with_suffix(".png")
        out_peaks = prefix.with_name(prefix.name + "_peaks_2700_3500").with_suffix(".txt")

        write_csv_like(out_csv, wn_cm1, intensity)
        write_csv_like(out_txt, wn_cm1, intensity)
        if not args.no_plot:
            maybe_plot(out_png, wn_cm1, intensity, title=f"IR spectrum (ACF-averaged, {label})")
        lo, hi = float(args.peak_range_cm1[0]), float(args.peak_range_cm1[1])
        write_peak_report(out_peaks, wn_cm1, intensity, lo_cm1=lo, hi_cm1=hi)

        # Outputs for spectrum-averaged (comparison)
        prefix_s = results_dir / f"ir_specavg_{label}"
        out_csv_s = prefix_s.with_suffix(".csv")
        out_txt_s = prefix_s.with_suffix(".txt")
        out_png_s = prefix_s.with_suffix(".png")
        out_peaks_s = prefix_s.with_name(prefix_s.name + "_peaks_2700_3500").with_suffix(".txt")

        write_csv_like(out_csv_s, wn_specavg, intensity_specavg)
        write_csv_like(out_txt_s, wn_specavg, intensity_specavg)
        if not args.no_plot:
            maybe_plot(out_png_s, wn_specavg, intensity_specavg, title=f"IR spectrum (spectrum-averaged, {label})")
        write_peak_report(out_peaks_s, wn_specavg, intensity_specavg, lo_cm1=lo, hi_cm1=hi)

        print(
            f"{label}: ACF-avg used {ntraj} trajectories -> {out_csv} | "
            f"spec-avg used {ntraj_specavg} trajectories -> {out_csv_s}"
        )

    # One overlay plot in the requested range
    if not args.no_plot:
        lo, hi = float(args.overlay_range_cm1[0]), float(args.overlay_range_cm1[1])
        overlay_png = results_dir / f"ir_overlay_{int(lo)}_{int(hi)}.png"
        plot_overlay(overlay_png, spectra_for_overlay, lo_cm1=lo, hi_cm1=hi)
        if overlay_png.exists():
            print(f"Overlay plot: {overlay_png}")
        else:
            print("Overlay plot skipped (matplotlib not available).")

        overlay_png_s = results_dir / f"ir_specavg_overlay_{int(lo)}_{int(hi)}.png"
        plot_overlay(overlay_png_s, spectra_for_overlay_specavg, lo_cm1=lo, hi_cm1=hi)
        if overlay_png_s.exists():
            print(f"Spectrum-avg overlay plot: {overlay_png_s}")
        else:
            print("Spectrum-avg overlay plot skipped (matplotlib not available).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


