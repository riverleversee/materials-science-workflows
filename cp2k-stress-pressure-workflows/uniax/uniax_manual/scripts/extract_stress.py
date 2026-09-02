#!/usr/bin/env python3
"""
Extract stress tensor from CP2K output and write to text file.
Usage: extract_stress.py <cp2k_output> <output_file>
Writes 3x3 stress tensor (bar) and 6-component vector (xx yy zz xy xz yz).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_parser import parse_stress_from_output


def main():
    if len(sys.argv) != 3:
        print("Usage: extract_stress.py <cp2k_output> <output_file>", file=sys.stderr)
        sys.exit(2)
    out_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])
    tensor = parse_stress_from_output(out_path, as_matrix=True)
    if tensor is None:
        print(f"Error: Could not parse stress tensor from {out_path}", file=sys.stderr)
        sys.exit(1)
    lines = [
        "# Stress tensor [bar] from CP2K single-point",
        "# xx  xy  xz",
        "# yx  yy  yz",
        "# zx  zy  zz",
        "",
        f"{tensor[0][0]:18.6f} {tensor[0][1]:18.6f} {tensor[0][2]:18.6f}",
        f"{tensor[1][0]:18.6f} {tensor[1][1]:18.6f} {tensor[1][2]:18.6f}",
        f"{tensor[2][0]:18.6f} {tensor[2][1]:18.6f} {tensor[2][2]:18.6f}",
        "",
        "# 6-component (xx yy zz xy xz yz):",
        f"{tensor[0][0]:18.6f} {tensor[1][1]:18.6f} {tensor[2][2]:18.6f} {tensor[0][1]:18.6f} {tensor[0][2]:18.6f} {tensor[1][2]:18.6f}",
    ]
    dst_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
