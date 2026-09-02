#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def label_value(path: Path) -> float:
    m = re.match(r"ir_(\d+(?:p\d+)?)perbr\.csv$", path.name)
    if not m:
        return float("inf")
    return float(m.group(1).replace("p", "."))


def plot_single(csv_path: Path) -> Path:
    wn, intensity = load_csv(csv_path)
    out_png = csv_path.with_suffix(".png")
    plt.figure(figsize=(11, 5))
    plt.plot(wn, intensity, linewidth=1.0)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb.)")
    plt.title(f"IR spectrum ({csv_path.stem})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def plot_overlay(csv_paths: list[Path], out_png: Path, lo: float, hi: float) -> None:
    plt.figure(figsize=(11, 5))
    for csv_path in sorted(csv_paths, key=label_value):
        wn, intensity = load_csv(csv_path)
        mask = (wn >= lo) & (wn <= hi)
        plt.plot(wn[mask], intensity[mask], linewidth=1.2, label=csv_path.stem.removeprefix("ir_"))
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb.)")
    plt.title(f"IR spectra overlay ({int(lo)}-{int(hi)} cm$^{{-1}}$)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--overlay-lo", type=float, default=2900.0)
    ap.add_argument("--overlay-hi", type=float, default=3200.0)
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    csv_paths = [
        p for p in results_dir.glob("ir_*perbr.csv")
        if not p.name.startswith("ir_specavg_")
    ]
    if not csv_paths:
        raise SystemExit(f"No primary IR CSVs found in {results_dir}")

    for csv_path in csv_paths:
        out_png = plot_single(csv_path)
        print(f"Wrote {out_png}")

    overlay_png = results_dir / f"ir_overlay_{int(args.overlay_lo)}_{int(args.overlay_hi)}.png"
    plot_overlay(csv_paths, overlay_png, lo=args.overlay_lo, hi=args.overlay_hi)
    print(f"Wrote {overlay_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
