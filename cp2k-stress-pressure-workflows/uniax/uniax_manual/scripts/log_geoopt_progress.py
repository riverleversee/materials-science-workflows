#!/usr/bin/env python3
"""
Append GeoOpt progress to a log file after each cell-optimization cycle.
Writes: step number, cell, stress tensor, atomic coordinates.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_parser import parse_stress_from_output


def read_cell(path):
    """Read CP2K cell file (A/B/C rows)."""
    lines = Path(path).read_text().splitlines()
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() in ("A", "B", "C"):
            rows.append(f"{parts[0]}   {parts[1]} {parts[2]} {parts[3]}")
    return "\n".join(rows) if rows else "(no cell)"


def read_xyz(path):
    """Read last frame from XYZ file."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        return "(empty)"
    # Find last frame: scan backwards for natoms line
    i = len(lines) - 1
    n = 0
    while i >= 0:
        try:
            n = int(lines[i].split()[0])
            break
        except (ValueError, IndexError):
            i -= 1
    if n <= 0:
        return "(no coords)"
    # Frame: natoms at i, comment at i+1, coords at i+2..i+1+n
    start = i
    end = min(i + 2 + n, len(lines))
    return "\n".join(lines[start:end])


def main():
    parser = argparse.ArgumentParser(description="Log GeoOpt progress")
    parser.add_argument("--step", type=int, required=True, help="Cycle/step number")
    parser.add_argument("--geo-out", required=True, help="run_geo.out path")
    parser.add_argument("--cell", required=True, help="init_cell.cell path")
    parser.add_argument("--coords", required=True, help="QM_geo-POS-pos-1.xyz path")
    parser.add_argument("--output", default="geoopt_progress.txt", help="Progress log file")
    args = parser.parse_args()

    stress = parse_stress_from_output(args.geo_out, as_matrix=True)
    if stress is None:
        print(f"Warning: Could not parse stress from {args.geo_out}", file=sys.stderr)
        stress_str = "(parse failed)"
    else:
        stress_str = "\n".join(
            f"  {stress[i][0]:18.6f} {stress[i][1]:18.6f} {stress[i][2]:18.6f}"
            for i in range(3)
        )

    cell_str = read_cell(args.cell)
    coords_str = read_xyz(args.coords)

    out_path = Path(args.output)
    with open(out_path, "a") as f:
        f.write(f"Step {args.step}\n")
        f.write("Cell:\n")
        f.write(cell_str + "\n")
        f.write("Stress tensor [bar]:\n")
        f.write(stress_str + "\n")
        f.write("Atomic coordinates (XYZ, last frame):\n")
        f.write(coords_str + "\n")
        f.write("===========\n")


if __name__ == "__main__":
    main()
