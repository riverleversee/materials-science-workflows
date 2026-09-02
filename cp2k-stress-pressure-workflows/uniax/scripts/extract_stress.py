#!/usr/bin/env python3
"""
Extract stress tensor from CP2K output and write to text file.
Usage: extract_stress.py <cp2k_output> <output_file>
Writes 3x3 stress tensor (bar) and 6-component vector (xx yy zz xy xz yz).
"""
import re
import sys
from pathlib import Path


def parse_stress(out_path):
    path = Path(out_path)
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    last_tensor = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if "STRESS" in line.upper() and "TENSOR" in line.upper():
            block = []
            for k in range(1, 8):
                if i + k >= len(lines):
                    break
                nums = re.findall(r"-?\d+\.?\d*(?:[Ee][-+]?\d+)?", lines[i + k])
                if len(nums) >= 3:
                    try:
                        row = [float(nums[0]), float(nums[1]), float(nums[2])]
                        if all(-1e7 < x < 1e7 for x in row):
                            block.append(row)
                    except (ValueError, IndexError):
                        break
                elif block:
                    break
            if len(block) >= 3:
                last_tensor = [block[0], block[1], block[2]]
                i += 3
        i += 1
    return last_tensor


def main():
    if len(sys.argv) != 3:
        print("Usage: extract_stress.py <cp2k_output> <output_file>", file=sys.stderr)
        sys.exit(2)
    out_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])
    tensor = parse_stress(out_path)
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
