#!/usr/bin/env python3
"""
Compare current stress (from CP2K GeoOpt output) to target stress.
Exit 0 if within tolerance, else 1.

Usage: check_stress_converged.py <cp2k_output> <xx> <yy> <zz> <xy> <xz> <yz> [--tolerance 500]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_parser import parse_stress_from_output


def main():
    parser = argparse.ArgumentParser(description="Check if stress matches target")
    parser.add_argument("cp2k_output", help="CP2K output file (e.g. run_geo.out)")
    parser.add_argument("xx", type=float, help="Target stress xx [bar]")
    parser.add_argument("yy", type=float, help="Target stress yy [bar]")
    parser.add_argument("zz", type=float, help="Target stress zz [bar]")
    parser.add_argument("xy", type=float, help="Target stress xy [bar]")
    parser.add_argument("xz", type=float, help="Target stress xz [bar]")
    parser.add_argument("yz", type=float, help="Target stress yz [bar]")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=500,
        help="Max allowed |sigma - target| per component [bar] (default 500)",
    )
    args = parser.parse_args()

    target = [args.xx, args.yy, args.zz, args.xy, args.xz, args.yz]
    stress = parse_stress_from_output(args.cp2k_output)
    if stress is None:
        print(f"Error: Could not parse stress from {args.cp2k_output}", file=sys.stderr)
        sys.exit(2)

    diff = [abs(s - t) for s, t in zip(stress, target)]
    max_diff = max(diff)
    if max_diff < args.tolerance:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
