#!/usr/bin/env python3
"""
CLI entry point for group-resolved normal-mode analysis (canonical workflow).

Analyzes BNFF/DNTF trajectories at multiple pressures and writes JSON + publication plots.
Point --data-dir at a study tree containing optimized_cell*GPa.cell and */AnimationFiles/anime_*.xyz.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def _load_driver_module():
    """Import the canonical driver without requiring a package install."""
    here = os.path.dirname(os.path.abspath(__file__))
    driver_path = os.path.join(
        here,
        "BNFFanalysis",
        "minpress",
        "group_mode_analysis.py",
    )
    spec = importlib.util.spec_from_file_location("nma_groups_driver", driver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load driver from {driver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_modes(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_pressures(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    from mode_analysis.paths import build_study_paths, resolve_data_dir

    parser = argparse.ArgumentParser(
        description="Run group-resolved energetic-crystal normal-mode analysis."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Study root (optimized_cell*GPa.cell + {P}GPa/AnimationFiles/). "
        "Defaults to NMA_DATA_DIR or BNFFanalysis/minpress/.",
    )
    parser.add_argument(
        "--pressures",
        default="0,4,10",
        help="Comma-separated hydrostatic pressures in GPa (default: 0,4,10).",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated mode indices. Default: intersection of anime_*.xyz at all pressures.",
    )
    parser.add_argument(
        "--bond-cutoff",
        type=float,
        default=1.6,
        help="Bond cutoff for connectivity (Angstrom).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=3.2,
        help="Intermolecular coupling distance threshold (Angstrom).",
    )
    parser.add_argument(
        "--output-prefix",
        default="final_analysismodes_groups",
        help="Prefix for JSON results and intermediate displacement files.",
    )
    parser.add_argument(
        "--plots-prefix",
        default="modesout_groups",
        help="Prefix for generated PNG figures.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis and write JSON; skip publication plot pass.",
    )
    parser.add_argument(
        "--plots-only",
        metavar="RESULTS_JSON",
        help="Regenerate plots from an existing results JSON (skip trajectory processing).",
    )
    args = parser.parse_args(argv)

    driver = _load_driver_module()
    data_dir = resolve_data_dir(args.data_dir)

    if args.plots_only:
        import json

        with open(args.plots_only, "r", encoding="utf-8") as handle:
            results_dict = json.load(handle)
        driver.make_plots(results_dict, output_prefix=args.plots_prefix)
        print(f"Plots written with prefix '{args.plots_prefix}'")
        return 0

    study = build_study_paths(
        data_dir=data_dir,
        pressures_gpa=_parse_pressures(args.pressures),
        mode_indices=_parse_modes(args.modes),
    )
    if not study["modes"]:
        print(
            "ERROR: No common anime_*.xyz modes found under the pressure folders.\n"
            f"  data_dir = {study['data_dir']}\n"
            f"  pressures = {study['pressures']}\n"
            "Copy your BNFFanalysis tree locally or set NMA_DATA_DIR.",
            file=sys.stderr,
        )
        return 1

    missing = [
        path
        for grid in study["mode_trajectory_grid"]
        for path in grid
        if not os.path.exists(path)
    ]
    if missing:
        print("ERROR: Missing trajectory files (first few):", file=sys.stderr)
        for path in missing[:5]:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(f"data_dir={study['data_dir']}")
    print(f"pressures_GPa={study['pressures']}")
    print(f"modes={study['modes']}")

    results_dict, results_path = driver.run_full_analysis(
        study["pressures"],
        study["position_paths"],
        study["unit_cells"],
        study["mode_trajectory_grid"],
        study["modes"],
        args.bond_cutoff,
        args.distance_threshold,
        output_prefix=args.output_prefix,
    )
    print(f"Wrote {results_path}")

    if not args.analysis_only:
        driver.make_plots(results_dict, output_prefix=args.plots_prefix)
        print(f"Plots written with prefix '{args.plots_prefix}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
