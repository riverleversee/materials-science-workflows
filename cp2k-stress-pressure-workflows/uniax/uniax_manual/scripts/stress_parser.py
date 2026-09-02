#!/usr/bin/env python3
"""
Parse stress tensor from CP2K output.
Returns 6-component vector (xx, yy, zz, xy, xz, yz) or 3x3 array.
"""
import re
from pathlib import Path


def parse_stress_from_output(out_path, as_matrix=False):
    """
    Parse the last stress tensor from CP2K output.
    Returns 6-component vector (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) in bar,
    or 3x3 symmetric matrix if as_matrix=True.
    """
    path = Path(out_path)
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    last_tensor = None
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match stress tensor block but not eigenvalues/eigenvectors block
        if ("STRESS" in line.upper() and "TENSOR" in line.upper() and
                "EIGENVALUE" not in line.upper() and "EIGENVECTOR" not in line.upper()):
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

    if last_tensor is None:
        return None

    # Symmetric: sigma_ij = sigma_ji. CP2K typically outputs full 3x3.
    mat = [
        [last_tensor[0][0], last_tensor[0][1], last_tensor[0][2]],
        [last_tensor[1][0], last_tensor[1][1], last_tensor[1][2]],
        [last_tensor[2][0], last_tensor[2][1], last_tensor[2][2]],
    ]
    if as_matrix:
        return mat
    # Flatten to 6 components: xx, yy, zz, xy, xz, yz
    vec = [
        mat[0][0], mat[1][1], mat[2][2],
        mat[0][1], mat[0][2], mat[1][2],
    ]
    return vec
